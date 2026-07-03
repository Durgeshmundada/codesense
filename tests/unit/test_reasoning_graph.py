"""Unit tests for ReasoningGraph state conversion, graph structure, and node wiring.

Tests the GraphState TypedDict conversion functions that bridge
the AgentState dataclass with LangGraph's state management,
plus integration tests for the graph structure itself.
"""

from unittest.mock import MagicMock, patch

import pytest

from codesense.models.state import (
    AgentState,
    Conflict,
    ConflictSource,
    Evidence,
    Hypothesis,
    NodeType,
    SynthesisResult,
)


# Import graph module - may take time loading dependencies
from codesense.agent.graph import (
    GraphState,
    ReasoningGraph,
    _agent_state_to_graph_state,
    _graph_state_to_agent_state,
)
from codesense.agent.router import CONFIDENCE_THRESHOLD, MAX_LOOPS


class TestAgentStateToGraphState:
    """Tests for converting AgentState dataclass to GraphState TypedDict."""

    def test_converts_empty_state(self):
        state = AgentState(query="why?", code_path="src/main.py")
        result = _agent_state_to_graph_state(state)

        assert result["query"] == "why?"
        assert result["code_path"] == "src/main.py"
        assert result["loop_counter"] == 0
        assert result["remaining_iterations"] == 3
        assert result["evidence"] == []
        assert result["hypotheses"] == []
        assert result["confidence_score"] == 0.0
        assert result["conflicts"] == []
        assert result["synthesis"] is None
        assert result["current_node"] == "explore"
        assert result["is_incomplete"] is False

    def test_converts_state_with_evidence(self):
        evidence = [
            Evidence(
                source_type="git_commit",
                source_id="abc123",
                content="Fixed auth bug",
                timestamp="2024-01-01",
                metadata={"author": "dev"},
            )
        ]
        state = AgentState(
            query="why?", code_path="src/auth.py", evidence=evidence
        )
        result = _agent_state_to_graph_state(state)

        assert len(result["evidence"]) == 1
        assert result["evidence"][0]["source_type"] == "git_commit"
        assert result["evidence"][0]["source_id"] == "abc123"
        assert result["evidence"][0]["content"] == "Fixed auth bug"
        assert result["evidence"][0]["timestamp"] == "2024-01-01"
        assert result["evidence"][0]["metadata"] == {"author": "dev"}

    def test_converts_state_with_conflicts(self):
        conflicts = [
            Conflict(
                id="c1",
                sources=[
                    ConflictSource(source_id="s1", claim="A is true"),
                    ConflictSource(source_id="s2", claim="A is false"),
                ],
                description="Conflicting claims about A",
            )
        ]
        state = AgentState(
            query="why?", code_path="src/a.py", conflicts=conflicts
        )
        result = _agent_state_to_graph_state(state)

        assert len(result["conflicts"]) == 1
        assert result["conflicts"][0]["id"] == "c1"
        assert len(result["conflicts"][0]["sources"]) == 2
        assert result["conflicts"][0]["sources"][0]["source_id"] == "s1"
        assert result["conflicts"][0]["sources"][0]["claim"] == "A is true"

    def test_converts_state_with_synthesis(self):
        synthesis = SynthesisResult(
            answer="Code exists for auth",
            confidence=0.85,
            supporting_evidence=["s1", "s2"],
            conflicts=[],
            reasoning_path=[NodeType.EXPLORE, NodeType.HYPOTHESIZE, NodeType.SYNTHESIZE],
            is_incomplete=False,
        )
        state = AgentState(
            query="why?", code_path="src/a.py", synthesis=synthesis
        )
        result = _agent_state_to_graph_state(state)

        assert result["synthesis"] is not None
        assert result["synthesis"]["answer"] == "Code exists for auth"
        assert result["synthesis"]["confidence"] == 0.85
        assert result["synthesis"]["reasoning_path"] == [
            "explore", "hypothesize", "synthesize"
        ]


class TestGraphStateToAgentState:
    """Tests for converting GraphState TypedDict back to AgentState dataclass."""

    def test_converts_empty_graph_state(self):
        graph_state = {
            "query": "why does this exist?",
            "code_path": "src/main.py",
            "loop_counter": 0,
            "remaining_iterations": 3,
            "evidence": [],
            "hypotheses": [],
            "confidence_score": 0.0,
            "conflicts": [],
            "synthesis": None,
            "current_node": "explore",
            "is_incomplete": False,
        }
        result = _graph_state_to_agent_state(graph_state)

        assert isinstance(result, AgentState)
        assert result.query == "why does this exist?"
        assert result.code_path == "src/main.py"
        assert result.loop_counter == 0
        assert result.remaining_iterations == 3
        assert result.current_node == NodeType.EXPLORE

    def test_converts_graph_state_with_evidence(self):
        graph_state = {
            "query": "why?",
            "code_path": "src/a.py",
            "loop_counter": 1,
            "remaining_iterations": 2,
            "evidence": [
                {
                    "source_type": "github_issue",
                    "source_id": "issue-42",
                    "content": "Add caching",
                    "timestamp": None,
                    "metadata": {"labels": ["enhancement"]},
                }
            ],
            "hypotheses": [
                {
                    "id": "h1",
                    "explanation": "Code exists for caching",
                    "supporting_evidence": ["issue-42"],
                    "confidence": 0.7,
                }
            ],
            "confidence_score": 0.7,
            "conflicts": [],
            "synthesis": None,
            "current_node": "verify",
            "is_incomplete": False,
        }
        result = _graph_state_to_agent_state(graph_state)

        assert len(result.evidence) == 1
        assert result.evidence[0].source_type == "github_issue"
        assert result.evidence[0].source_id == "issue-42"
        assert len(result.hypotheses) == 1
        assert result.hypotheses[0].explanation == "Code exists for caching"
        assert result.current_node == NodeType.VERIFY

    def test_roundtrip_conversion(self):
        """Converting AgentState -> GraphState -> AgentState preserves data."""
        original = AgentState(
            query="test query",
            code_path="src/test.py",
            loop_counter=2,
            remaining_iterations=1,
            evidence=[
                Evidence(
                    source_type="git_commit",
                    source_id="sha1",
                    content="commit msg",
                    timestamp="2024-01-01",
                    metadata={"author": "dev"},
                )
            ],
            hypotheses=[
                Hypothesis(
                    id="h1",
                    explanation="because reasons",
                    supporting_evidence=["sha1"],
                    confidence=0.8,
                )
            ],
            confidence_score=0.8,
            conflicts=[
                Conflict(
                    id="c1",
                    sources=[
                        ConflictSource(source_id="s1", claim="yes"),
                        ConflictSource(source_id="s2", claim="no"),
                    ],
                    description="contradiction",
                )
            ],
            synthesis=None,
            current_node=NodeType.CHECK_CONTRADICTIONS,
            is_incomplete=False,
        )

        graph_state = _agent_state_to_graph_state(original)
        roundtrip = _graph_state_to_agent_state(graph_state)

        assert roundtrip.query == original.query
        assert roundtrip.code_path == original.code_path
        assert roundtrip.loop_counter == original.loop_counter
        assert roundtrip.remaining_iterations == original.remaining_iterations
        assert roundtrip.confidence_score == original.confidence_score
        assert roundtrip.current_node == original.current_node
        assert roundtrip.is_incomplete == original.is_incomplete
        assert len(roundtrip.evidence) == len(original.evidence)
        assert roundtrip.evidence[0].source_id == original.evidence[0].source_id
        assert len(roundtrip.hypotheses) == len(original.hypotheses)
        assert roundtrip.hypotheses[0].id == original.hypotheses[0].id
        assert len(roundtrip.conflicts) == len(original.conflicts)
        assert roundtrip.conflicts[0].id == original.conflicts[0].id

    def test_handles_invalid_node_type_gracefully(self):
        graph_state = {
            "query": "why?",
            "code_path": "src/a.py",
            "loop_counter": 0,
            "remaining_iterations": 3,
            "evidence": [],
            "hypotheses": [],
            "confidence_score": 0.0,
            "conflicts": [],
            "synthesis": None,
            "current_node": "invalid_node",
            "is_incomplete": False,
        }
        result = _graph_state_to_agent_state(graph_state)
        # Should default to EXPLORE on invalid node type
        assert result.current_node == NodeType.EXPLORE



class TestReasoningGraphStructure:
    """Tests for the ReasoningGraph class structure and wiring."""

    def _make_graph(self):
        """Create a ReasoningGraph with a mocked GeminiService."""
        mock_gemini = MagicMock()
        mock_gemini.generate.return_value = '[]'
        return ReasoningGraph(
            gemini_service=mock_gemini,
            mock=True,
            vector_store=None,
            embedder=None,
        )

    def test_constructor_stores_dependencies(self):
        mock_gemini = MagicMock()
        graph = ReasoningGraph(
            gemini_service=mock_gemini,
            mock=True,
            vector_store=None,
            embedder=None,
        )
        assert graph._gemini_service is mock_gemini
        assert graph._mock is True
        assert graph._vector_store is None
        assert graph._embedder is None

    def test_graph_compiles_successfully(self):
        """The graph should compile without error during construction."""
        graph = self._make_graph()
        assert graph._graph is not None

    def test_checkpointer_is_memory_saver(self):
        """Graph uses MemorySaver checkpointer for persistence."""
        from langgraph.checkpoint.memory import MemorySaver
        graph = self._make_graph()
        assert isinstance(graph._checkpointer, MemorySaver)

    def test_graph_has_all_five_nodes(self):
        """The compiled graph should contain all 5 reasoning nodes."""
        graph = self._make_graph()
        # Access the graph's nodes via the internal structure
        node_names = set(graph._graph.get_graph().nodes.keys())
        expected = {"explore", "hypothesize", "verify", "check_contradictions", "synthesize"}
        assert expected.issubset(node_names)


class TestReasoningGraphRouting:
    """Tests for the conditional routing logic in the graph."""

    def _make_graph(self):
        mock_gemini = MagicMock()
        mock_gemini.generate.return_value = '[]'
        return ReasoningGraph(
            gemini_service=mock_gemini,
            mock=True,
            vector_store=None,
            embedder=None,
        )

    def test_route_synthesize_when_max_loops_reached(self):
        """Router returns 'synthesize' when loop_counter >= MAX_LOOPS."""
        graph = self._make_graph()
        state: GraphState = {
            "query": "why?",
            "code_path": "src/a.py",
            "loop_counter": MAX_LOOPS,
            "remaining_iterations": 3,
            "evidence": [],
            "hypotheses": [],
            "confidence_score": 0.5,
            "conflicts": [],
            "synthesis": None,
            "current_node": "check_contradictions",
            "is_incomplete": False,
        }
        result = graph._route_after_check(state)
        assert result == "synthesize"

    def test_route_synthesize_when_remaining_iterations_zero(self):
        """Router returns 'synthesize' when remaining_iterations <= 0."""
        graph = self._make_graph()
        state: GraphState = {
            "query": "why?",
            "code_path": "src/a.py",
            "loop_counter": 1,
            "remaining_iterations": 0,
            "evidence": [],
            "hypotheses": [],
            "confidence_score": 0.5,
            "conflicts": [],
            "synthesis": None,
            "current_node": "check_contradictions",
            "is_incomplete": False,
        }
        result = graph._route_after_check(state)
        assert result == "synthesize"

    def test_route_hypothesize_when_conflicts_exist(self):
        """Router returns 'hypothesize' when conflicts are detected."""
        graph = self._make_graph()
        state: GraphState = {
            "query": "why?",
            "code_path": "src/a.py",
            "loop_counter": 1,
            "remaining_iterations": 2,
            "evidence": [],
            "hypotheses": [],
            "confidence_score": 0.8,
            "conflicts": [
                {
                    "id": "c1",
                    "sources": [
                        {"source_id": "s1", "claim": "A"},
                        {"source_id": "s2", "claim": "B"},
                    ],
                    "description": "contradiction",
                }
            ],
            "synthesis": None,
            "current_node": "check_contradictions",
            "is_incomplete": False,
        }
        result = graph._route_after_check(state)
        assert result == "hypothesize"

    def test_route_explore_when_low_confidence(self):
        """Router returns 'explore' when confidence < threshold and no conflicts."""
        graph = self._make_graph()
        state: GraphState = {
            "query": "why?",
            "code_path": "src/a.py",
            "loop_counter": 1,
            "remaining_iterations": 2,
            "evidence": [],
            "hypotheses": [],
            "confidence_score": CONFIDENCE_THRESHOLD - 0.1,
            "conflicts": [],
            "synthesis": None,
            "current_node": "check_contradictions",
            "is_incomplete": False,
        }
        result = graph._route_after_check(state)
        assert result == "explore"

    def test_route_synthesize_when_converged(self):
        """Router returns 'synthesize' when confidence >= threshold and no conflicts."""
        graph = self._make_graph()
        state: GraphState = {
            "query": "why?",
            "code_path": "src/a.py",
            "loop_counter": 1,
            "remaining_iterations": 2,
            "evidence": [],
            "hypotheses": [],
            "confidence_score": CONFIDENCE_THRESHOLD,
            "conflicts": [],
            "synthesis": None,
            "current_node": "check_contradictions",
            "is_incomplete": False,
        }
        result = graph._route_after_check(state)
        assert result == "synthesize"


class TestReasoningGraphNodeFailureHandling:
    """Tests that node wrappers catch exceptions and set is_incomplete=True."""

    def _make_graph(self):
        mock_gemini = MagicMock()
        mock_gemini.generate.return_value = '[]'
        return ReasoningGraph(
            gemini_service=mock_gemini,
            mock=True,
            vector_store=None,
            embedder=None,
        )

    def test_explore_wrapper_increments_loop_counter(self):
        """Explore wrapper should increment loop_counter at cycle start."""
        graph = self._make_graph()
        state: GraphState = {
            "query": "why?",
            "code_path": "src/a.py",
            "loop_counter": 0,
            "remaining_iterations": 3,
            "evidence": [],
            "hypotheses": [],
            "confidence_score": 0.0,
            "conflicts": [],
            "synthesis": None,
            "current_node": "explore",
            "is_incomplete": False,
        }
        # Mock the explore node's execute to just return state
        with patch.object(graph._explore_node, "execute", side_effect=lambda s: s):
            result = graph._explore_wrapper(state)
        assert result["loop_counter"] == 1

    def test_explore_wrapper_handles_failure(self):
        """Explore wrapper sets is_incomplete=True on failure."""
        graph = self._make_graph()
        state: GraphState = {
            "query": "why?",
            "code_path": "src/a.py",
            "loop_counter": 0,
            "remaining_iterations": 3,
            "evidence": [],
            "hypotheses": [],
            "confidence_score": 0.0,
            "conflicts": [],
            "synthesis": None,
            "current_node": "explore",
            "is_incomplete": False,
        }
        with patch.object(
            graph._explore_node, "execute", side_effect=RuntimeError("fail")
        ):
            result = graph._explore_wrapper(state)
        assert result["is_incomplete"] is True

    def test_hypothesize_wrapper_handles_failure(self):
        """Hypothesize wrapper sets is_incomplete=True on failure."""
        graph = self._make_graph()
        state: GraphState = {
            "query": "why?",
            "code_path": "src/a.py",
            "loop_counter": 1,
            "remaining_iterations": 3,
            "evidence": [],
            "hypotheses": [],
            "confidence_score": 0.0,
            "conflicts": [],
            "synthesis": None,
            "current_node": "hypothesize",
            "is_incomplete": False,
        }
        with patch.object(
            graph._hypothesize_node, "execute", side_effect=RuntimeError("fail")
        ):
            result = graph._hypothesize_wrapper(state)
        assert result["is_incomplete"] is True

    def test_verify_wrapper_handles_failure(self):
        """Verify wrapper sets is_incomplete=True on failure."""
        graph = self._make_graph()
        state: GraphState = {
            "query": "why?",
            "code_path": "src/a.py",
            "loop_counter": 1,
            "remaining_iterations": 3,
            "evidence": [],
            "hypotheses": [],
            "confidence_score": 0.0,
            "conflicts": [],
            "synthesis": None,
            "current_node": "verify",
            "is_incomplete": False,
        }
        with patch.object(
            graph._verify_node, "execute", side_effect=RuntimeError("fail")
        ):
            result = graph._verify_wrapper(state)
        assert result["is_incomplete"] is True

    def test_check_contradictions_wrapper_handles_failure(self):
        """Check contradictions wrapper sets is_incomplete=True on failure."""
        graph = self._make_graph()
        state: GraphState = {
            "query": "why?",
            "code_path": "src/a.py",
            "loop_counter": 1,
            "remaining_iterations": 3,
            "evidence": [],
            "hypotheses": [],
            "confidence_score": 0.0,
            "conflicts": [],
            "synthesis": None,
            "current_node": "check_contradictions",
            "is_incomplete": False,
        }
        with patch.object(
            graph._check_contradictions_node,
            "execute",
            side_effect=RuntimeError("fail"),
        ):
            result = graph._check_contradictions_wrapper(state)
        assert result["is_incomplete"] is True

    def test_synthesize_wrapper_handles_failure(self):
        """Synthesize wrapper produces fallback synthesis on failure."""
        graph = self._make_graph()
        state: GraphState = {
            "query": "why?",
            "code_path": "src/a.py",
            "loop_counter": 1,
            "remaining_iterations": 3,
            "evidence": [
                {
                    "source_type": "git_commit",
                    "source_id": "abc",
                    "content": "test",
                    "timestamp": None,
                    "metadata": {},
                }
            ],
            "hypotheses": [],
            "confidence_score": 0.0,
            "conflicts": [],
            "synthesis": None,
            "current_node": "synthesize",
            "is_incomplete": False,
        }
        with patch.object(
            graph._synthesize_node, "execute", side_effect=RuntimeError("fail")
        ):
            result = graph._synthesize_wrapper(state)
        assert result["is_incomplete"] is True
        assert result["synthesis"] is not None
        assert result["synthesis"]["is_incomplete"] is True
        assert "abc" in result["synthesis"]["supporting_evidence"]


class TestReasoningGraphRun:
    """Tests for the run() method of ReasoningGraph."""

    def test_run_returns_agent_state(self):
        """The run method should return a valid AgentState with synthesis."""
        mock_gemini = MagicMock()
        # Mock generate to return valid JSON responses for each node
        mock_gemini.generate.return_value = (
            '[{"explanation": "test hypothesis", "confidence": 0.9, "supporting_evidence": []}]'
        )

        graph = ReasoningGraph(
            gemini_service=mock_gemini,
            mock=True,
            vector_store=None,
            embedder=None,
        )

        result = graph.run(query="why does auth.py exist?", code_path="src/auth.py")

        assert isinstance(result, AgentState)
        assert result.query == "why does auth.py exist?"
        assert result.code_path == "src/auth.py"
        # Loop counter should be >= 1 since explore increments it
        assert result.loop_counter >= 1

    def test_run_produces_synthesis_result(self):
        """The run method should end with a non-None synthesis result."""
        mock_gemini = MagicMock()
        # First call (hypothesize) returns hypotheses
        # Second call (verify) returns confidence
        # Third call (check_contradictions) returns no conflicts
        # Fourth call (synthesize) returns final answer
        mock_gemini.generate.side_effect = [
            '[{"explanation": "test", "confidence": 0.85, "supporting_evidence": []}]',
            '{"hypothesis_scores": [], "overall_confidence": 0.85, "verification_summary": "good"}',
            '[]',
            '{"answer": "Code exists for authentication", "confidence": 0.85}',
        ]

        graph = ReasoningGraph(
            gemini_service=mock_gemini,
            mock=True,
            vector_store=None,
            embedder=None,
        )

        result = graph.run(query="why?", code_path="src/auth.py")

        assert isinstance(result, AgentState)
        assert result.synthesis is not None
        assert result.synthesis.answer != ""
