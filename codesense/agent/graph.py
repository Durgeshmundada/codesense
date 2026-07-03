"""ReasoningGraph — LangGraph StateGraph orchestrating the 5-node reasoning loop.

Builds and compiles a StateGraph with nodes: explore → hypothesize → verify →
check_contradictions → (conditional router) → synthesize/loop-back.

The graph uses a TypedDict state representation for LangGraph compatibility,
converting to/from the AgentState dataclass at node boundaries.

Requirements: 1.1, 1.5, 1.6, 1.7, 1.8, 1.9, 9.1, 9.2
"""

import logging
import uuid
from typing import Any, Optional

from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from codesense.agent.check_contradictions import CheckContradictionsNode
from codesense.agent.nodes import (
    ExploreNode,
    HypothesizeNode,
    SynthesizeNode,
    VerifyNode,
)
from codesense.agent.router import CONFIDENCE_THRESHOLD, MAX_LOOPS, Router
from codesense.llm.gemini_service import GeminiService
from codesense.memory.embedder import HuggingFaceEmbedder
from codesense.memory.vector_store import VectorStore
from codesense.models.state import (
    AgentState,
    Conflict,
    ConflictSource,
    Evidence,
    Hypothesis,
    NodeType,
    SynthesisResult,
)

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """TypedDict state for LangGraph StateGraph compatibility.

    Mirrors AgentState dataclass fields but uses plain types that
    LangGraph can serialize and manage through its state channels.
    """

    query: str
    code_path: str
    loop_counter: int
    remaining_iterations: int
    evidence: list[dict[str, Any]]
    hypotheses: list[dict[str, Any]]
    confidence_score: float
    conflicts: list[dict[str, Any]]
    synthesis: Optional[dict[str, Any]]
    current_node: str
    is_incomplete: bool


def _agent_state_to_graph_state(state: AgentState) -> GraphState:
    """Convert an AgentState dataclass to the GraphState TypedDict."""
    return GraphState(
        query=state.query,
        code_path=state.code_path,
        loop_counter=state.loop_counter,
        remaining_iterations=state.remaining_iterations,
        evidence=[
            {
                "source_type": e.source_type,
                "source_id": e.source_id,
                "content": e.content,
                "timestamp": e.timestamp,
                "metadata": e.metadata,
            }
            for e in state.evidence
        ],
        hypotheses=[
            {
                "id": h.id,
                "explanation": h.explanation,
                "supporting_evidence": h.supporting_evidence,
                "confidence": h.confidence,
            }
            for h in state.hypotheses
        ],
        confidence_score=state.confidence_score,
        conflicts=[
            {
                "id": c.id,
                "sources": [
                    {"source_id": s.source_id, "claim": s.claim} for s in c.sources
                ],
                "description": c.description,
            }
            for c in state.conflicts
        ],
        synthesis=(
            {
                "answer": state.synthesis.answer,
                "confidence": state.synthesis.confidence,
                "supporting_evidence": state.synthesis.supporting_evidence,
                "conflicts": [
                    {
                        "id": c.id,
                        "sources": [
                            {"source_id": s.source_id, "claim": s.claim}
                            for s in c.sources
                        ],
                        "description": c.description,
                    }
                    for c in state.synthesis.conflicts
                ],
                "reasoning_path": [n.value for n in state.synthesis.reasoning_path],
                "is_incomplete": state.synthesis.is_incomplete,
            }
            if state.synthesis
            else None
        ),
        current_node=state.current_node.value,
        is_incomplete=state.is_incomplete,
    )


def _graph_state_to_agent_state(state: GraphState) -> AgentState:
    """Convert a GraphState TypedDict back to an AgentState dataclass."""
    evidence = [
        Evidence(
            source_type=e["source_type"],
            source_id=e["source_id"],
            content=e["content"],
            timestamp=e.get("timestamp"),
            metadata=e.get("metadata", {}),
        )
        for e in state.get("evidence", [])
    ]

    hypotheses = [
        Hypothesis(
            id=h["id"],
            explanation=h["explanation"],
            supporting_evidence=h.get("supporting_evidence", []),
            confidence=h.get("confidence", 0.0),
        )
        for h in state.get("hypotheses", [])
    ]

    conflicts = [
        Conflict(
            id=c["id"],
            sources=[
                ConflictSource(source_id=s["source_id"], claim=s["claim"])
                for s in c.get("sources", [])
            ],
            description=c["description"],
        )
        for c in state.get("conflicts", [])
    ]

    synthesis_data = state.get("synthesis")
    synthesis = None
    if synthesis_data:
        synthesis = SynthesisResult(
            answer=synthesis_data["answer"],
            confidence=synthesis_data["confidence"],
            supporting_evidence=synthesis_data.get("supporting_evidence", []),
            conflicts=[
                Conflict(
                    id=c["id"],
                    sources=[
                        ConflictSource(source_id=s["source_id"], claim=s["claim"])
                        for s in c.get("sources", [])
                    ],
                    description=c["description"],
                )
                for c in synthesis_data.get("conflicts", [])
            ],
            reasoning_path=[
                NodeType(n) for n in synthesis_data.get("reasoning_path", [])
            ],
            is_incomplete=synthesis_data.get("is_incomplete", False),
        )

    current_node_str = state.get("current_node", "explore")
    try:
        current_node = NodeType(current_node_str)
    except ValueError:
        current_node = NodeType.EXPLORE

    return AgentState(
        query=state.get("query", ""),
        code_path=state.get("code_path", ""),
        loop_counter=state.get("loop_counter", 0),
        remaining_iterations=state.get("remaining_iterations", 3),
        evidence=evidence,
        hypotheses=hypotheses,
        confidence_score=state.get("confidence_score", 0.0),
        conflicts=conflicts,
        synthesis=synthesis,
        current_node=current_node,
        is_incomplete=state.get("is_incomplete", False),
    )


class ReasoningGraph:
    """Orchestrates the 5-node LangGraph reasoning loop.

    Builds a StateGraph with nodes: explore, hypothesize, verify,
    check_contradictions, and synthesize. Uses conditional routing
    after check_contradictions to decide whether to loop back or
    proceed to synthesis.

    The loop_counter is incremented at the start of each full cycle
    (in the explore node wrapper). MAX_LOOPS=3 ensures termination.

    Node failures are caught and routed to synthesize with
    is_incomplete=True, preserving all evidence gathered so far.

    Args:
        gemini_service: GeminiService instance for LLM calls.
        mock: Whether to use mock data sources.
        vector_store: Optional VectorStore for Decision Memory.
        embedder: Optional HuggingFaceEmbedder for query embedding.
    """

    def __init__(
        self,
        gemini_service: GeminiService,
        mock: bool = False,
        vector_store: Optional[VectorStore] = None,
        embedder: Optional[HuggingFaceEmbedder] = None,
    ) -> None:
        self._gemini_service = gemini_service
        self._mock = mock
        self._vector_store = vector_store
        self._embedder = embedder

        # Instantiate node objects
        self._explore_node = ExploreNode(
            mock=mock, vector_store=vector_store, embedder=embedder
        )
        self._hypothesize_node = HypothesizeNode(gemini_service=gemini_service)
        self._verify_node = VerifyNode(
            gemini_service=gemini_service,
            vector_store=vector_store,
            embedder=embedder,
        )
        self._check_contradictions_node = CheckContradictionsNode(
            gemini_service=gemini_service
        )
        self._synthesize_node = SynthesizeNode(gemini_service=gemini_service)

        # Build and compile the graph
        self._checkpointer = MemorySaver()
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Construct and compile the 5-node StateGraph.

        Graph structure:
            START → explore → hypothesize → verify → check_contradictions
            check_contradictions → (conditional router):
                - "hypothesize" → hypothesize
                - "explore" → explore
                - "synthesize" → synthesize
            synthesize → END

        Returns:
            Compiled LangGraph graph ready for invocation.
        """
        graph = StateGraph(GraphState)

        # Add nodes with error-handling wrappers
        graph.add_node("explore", self._explore_wrapper)
        graph.add_node("hypothesize", self._hypothesize_wrapper)
        graph.add_node("verify", self._verify_wrapper)
        graph.add_node("check_contradictions", self._check_contradictions_wrapper)
        graph.add_node("synthesize", self._synthesize_wrapper)

        # Wire edges: START → explore
        graph.add_edge(START, "explore")

        # Linear edges through the cycle
        graph.add_edge("explore", "hypothesize")
        graph.add_edge("hypothesize", "verify")
        graph.add_edge("verify", "check_contradictions")

        # Conditional edge from check_contradictions using the router
        graph.add_conditional_edges(
            "check_contradictions",
            self._route_after_check,
            {
                "hypothesize": "hypothesize",
                "explore": "explore",
                "synthesize": "synthesize",
            },
        )

        # Synthesize → END
        graph.add_edge("synthesize", END)

        # Compile with checkpointer
        return graph.compile(checkpointer=self._checkpointer)

    def _route_after_check(self, state: GraphState) -> str:
        """Route after check_contradictions using the Router logic.

        This is a pure decision function that does NOT modify state.
        The remaining_iterations decrement is already applied in
        _check_contradictions_wrapper, so we use the state as-is
        to determine the next node.

        Args:
            state: Current graph state as TypedDict.

        Returns:
            Next node name: "hypothesize", "explore", or "synthesize".
        """
        loop_counter = state.get("loop_counter", 0)
        remaining_iterations = state.get("remaining_iterations", 3)
        conflicts = state.get("conflicts", [])
        confidence_score = state.get("confidence_score", 0.0)

        # Rule 1: Termination conditions
        if loop_counter >= MAX_LOOPS or remaining_iterations <= 0:
            return "synthesize"

        # Rule 2: Conflicts detected → re-hypothesize
        if conflicts:
            return "hypothesize"

        # Rule 3: Low confidence → re-explore
        if confidence_score < CONFIDENCE_THRESHOLD:
            return "explore"

        # Rule 4: Converged
        return "synthesize"

    def _explore_wrapper(self, state: GraphState) -> GraphState:
        """Wrapper for ExploreNode that increments loop_counter.

        The loop_counter is incremented at the start of each full cycle,
        which begins with the explore node. This ensures termination
        after MAX_LOOPS iterations.

        On failure, routes to synthesize with is_incomplete=True.
        """
        agent_state = _graph_state_to_agent_state(state)

        # Increment loop_counter at the start of each full cycle
        agent_state.loop_counter += 1

        try:
            result = self._explore_node.execute(agent_state)
            return _agent_state_to_graph_state(result)
        except Exception as e:
            logger.error("ExploreNode failed: %s", e)
            agent_state.is_incomplete = True
            agent_state.current_node = NodeType.SYNTHESIZE
            return _agent_state_to_graph_state(agent_state)

    def _hypothesize_wrapper(self, state: GraphState) -> GraphState:
        """Wrapper for HypothesizeNode with error handling.

        On failure, marks state as incomplete for synthesis.
        """
        agent_state = _graph_state_to_agent_state(state)

        try:
            result = self._hypothesize_node.execute(agent_state)
            return _agent_state_to_graph_state(result)
        except Exception as e:
            logger.error("HypothesizeNode failed: %s", e)
            agent_state.is_incomplete = True
            agent_state.current_node = NodeType.SYNTHESIZE
            return _agent_state_to_graph_state(agent_state)

    def _verify_wrapper(self, state: GraphState) -> GraphState:
        """Wrapper for VerifyNode with error handling.

        On failure, marks state as incomplete for synthesis.
        """
        agent_state = _graph_state_to_agent_state(state)

        try:
            result = self._verify_node.execute(agent_state)
            return _agent_state_to_graph_state(result)
        except Exception as e:
            logger.error("VerifyNode failed: %s", e)
            agent_state.is_incomplete = True
            agent_state.current_node = NodeType.SYNTHESIZE
            return _agent_state_to_graph_state(agent_state)

    def _check_contradictions_wrapper(self, state: GraphState) -> GraphState:
        """Wrapper for CheckContradictionsNode with error handling.

        After running the contradiction check, this wrapper applies the
        remaining_iterations decrement that the Router logic requires.
        This ensures the decrement persists in the graph state (since
        conditional edge functions cannot modify state in LangGraph).

        On failure, marks state as incomplete for synthesis.
        """
        agent_state = _graph_state_to_agent_state(state)

        try:
            result = self._check_contradictions_node.execute(agent_state)

            # Apply remaining_iterations decrement based on routing decision.
            # The Router decrements when it routes to hypothesize (conflicts)
            # or explore (low confidence). We apply the same logic here so
            # the state persists.
            if result.loop_counter < MAX_LOOPS and result.remaining_iterations > 0:
                if result.conflicts:
                    result.remaining_iterations -= 1
                elif result.confidence_score < CONFIDENCE_THRESHOLD:
                    result.remaining_iterations -= 1

            return _agent_state_to_graph_state(result)
        except Exception as e:
            logger.error("CheckContradictionsNode failed: %s", e)
            agent_state.is_incomplete = True
            agent_state.current_node = NodeType.SYNTHESIZE
            return _agent_state_to_graph_state(agent_state)

    def _synthesize_wrapper(self, state: GraphState) -> GraphState:
        """Wrapper for SynthesizeNode with error handling.

        On failure, produces a minimal SynthesisResult indicating
        incomplete reasoning.
        """
        agent_state = _graph_state_to_agent_state(state)

        try:
            result = self._synthesize_node.execute(agent_state)
            return _agent_state_to_graph_state(result)
        except Exception as e:
            logger.error("SynthesizeNode failed: %s", e)
            # Produce a minimal fallback synthesis
            agent_state.synthesis = SynthesisResult(
                answer="Reasoning could not be completed due to an internal error.",
                confidence=0.0,
                supporting_evidence=[ev.source_id for ev in agent_state.evidence],
                conflicts=agent_state.conflicts,
                reasoning_path=[agent_state.current_node],
                is_incomplete=True,
            )
            agent_state.is_incomplete = True
            return _agent_state_to_graph_state(agent_state)

    def run(self, query: str, code_path: str, mock: bool = False) -> AgentState:
        """Execute the reasoning graph for a given query and code path.

        Invokes the compiled LangGraph StateGraph, passing the initial
        state through all nodes until reaching END. Returns the final
        AgentState with synthesis results.

        Args:
            query: The user's question about the code.
            code_path: Path to the code being analyzed.
            mock: Whether to use mock data sources (overrides instance setting).

        Returns:
            Final AgentState containing the synthesis result, evidence,
            hypotheses, conflicts, and reasoning metadata.
        """
        # Build initial state
        initial_state: GraphState = GraphState(
            query=query,
            code_path=code_path,
            loop_counter=0,
            remaining_iterations=3,
            evidence=[],
            hypotheses=[],
            confidence_score=0.0,
            conflicts=[],
            synthesis=None,
            current_node=NodeType.EXPLORE.value,
            is_incomplete=False,
        )

        # Generate a unique thread_id for this invocation
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        # Invoke the compiled graph
        final_state = self._graph.invoke(initial_state, config=config)

        # Convert back to AgentState dataclass
        return _graph_state_to_agent_state(final_state)
