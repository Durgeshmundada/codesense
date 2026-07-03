"""Property-based tests for reasoning agent node output invariants.

Tests Properties 1, 2, 5 from the design document using Hypothesis.

Validates: Requirements 1.2, 1.3, 1.9
"""

import json
import uuid
from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from codesense.agent.nodes import HypothesizeNode, VerifyNode
from codesense.models.state import (
    AgentState,
    Evidence,
    Hypothesis,
    NodeType,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def evidence_strategy() -> st.SearchStrategy[Evidence]:
    """Strategy to generate varied Evidence objects."""
    return st.builds(
        Evidence,
        source_type=st.sampled_from(
            ["git_commit", "github_issue", "pr_comment", "decision_unit", "related_change"]
        ),
        source_id=st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=("L", "N", "P"))),
        content=st.text(min_size=1, max_size=200),
        timestamp=st.one_of(st.none(), st.text(min_size=10, max_size=25)),
        metadata=st.just({}),
    )


def hypothesis_strategy() -> st.SearchStrategy[Hypothesis]:
    """Strategy to generate varied Hypothesis objects."""
    return st.builds(
        Hypothesis,
        id=st.from_type(type).map(lambda _: str(uuid.uuid4())),
        explanation=st.text(min_size=5, max_size=200),
        supporting_evidence=st.lists(
            st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=("L", "N", "P"))),
            min_size=0,
            max_size=5,
        ),
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )


def mock_hypothesize_response_strategy(evidence_ids: list[str]) -> st.SearchStrategy[str]:
    """Strategy to generate varied JSON responses from the LLM for hypothesize.

    Generates 1-8 hypotheses in JSON to test that HypothesizeNode enforces bounds.
    """
    def build_response(num_hypotheses: int, confidences: list[float]) -> str:
        hypotheses = []
        for i in range(num_hypotheses):
            conf = confidences[i] if i < len(confidences) else 0.5
            supporting = evidence_ids[:2] if evidence_ids else []
            hypotheses.append({
                "explanation": f"Hypothesis {i+1}: This code exists because of reason {i+1}.",
                "confidence": conf,
                "supporting_evidence": supporting,
            })
        return json.dumps(hypotheses)

    return st.builds(
        build_response,
        num_hypotheses=st.integers(min_value=1, max_value=8),
        confidences=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=8,
            max_size=8,
        ),
    )


def mock_verify_response_strategy() -> st.SearchStrategy[str]:
    """Strategy to generate varied JSON responses from the LLM for verify.

    Generates confidence scores that may be outside [0, 1] to test clamping.
    """
    return st.builds(
        lambda confidence: json.dumps({
            "hypothesis_scores": [
                {"hypothesis_id": "test-id", "confidence": confidence, "reasoning": "test"}
            ],
            "overall_confidence": confidence,
            "verification_summary": "Assessment complete.",
        }),
        confidence=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )


# ---------------------------------------------------------------------------
# Property 1: Hypothesis count bounds
# ---------------------------------------------------------------------------

# Feature: codesense, Property 1: Hypothesis count bounds
@settings(max_examples=100)
@given(
    evidence_list=st.lists(evidence_strategy(), min_size=0, max_size=20),
    data=st.data(),
)
def test_hypothesis_count_bounds(evidence_list: list[Evidence], data: st.DataObject) -> None:
    """For any evidence set (0-20 items), the HypothesizeNode produces 1-5 hypotheses.

    **Validates: Requirements 1.2**

    Uses a mock GeminiService that returns varied JSON with different numbers
    of hypotheses. The node must enforce bounds regardless of what the LLM returns.
    """
    # Build evidence IDs for the mock response
    evidence_ids = [e.source_id for e in evidence_list]

    # Generate a varied LLM response using Hypothesis
    mock_response = data.draw(mock_hypothesize_response_strategy(evidence_ids))

    # Create mock GeminiService
    mock_gemini = MagicMock()
    mock_gemini.generate.return_value = mock_response

    # Create node and execute
    node = HypothesizeNode(gemini_service=mock_gemini)
    state = AgentState(
        query="Why does this code exist?",
        code_path="src/example.py",
        evidence=evidence_list,
    )

    result = node.execute(state)

    # Property: always produces 1-5 hypotheses
    assert 1 <= len(result.hypotheses) <= 5, (
        f"Expected 1-5 hypotheses, got {len(result.hypotheses)}"
    )

    # Verify node sets current_node correctly
    assert result.current_node == NodeType.HYPOTHESIZE


# ---------------------------------------------------------------------------
# Property 2: Confidence score range invariant
# ---------------------------------------------------------------------------

# Feature: codesense, Property 2: Confidence score range invariant
@settings(max_examples=100)
@given(
    hypotheses_list=st.lists(hypothesis_strategy(), min_size=1, max_size=5),
    evidence_list=st.lists(evidence_strategy(), min_size=0, max_size=10),
    data=st.data(),
)
def test_confidence_score_range_invariant(
    hypotheses_list: list[Hypothesis],
    evidence_list: list[Evidence],
    data: st.DataObject,
) -> None:
    """For any set of hypotheses and evidence, the VerifyNode produces confidence in [0.0, 1.0].

    **Validates: Requirements 1.3**

    Uses a mock GeminiService returning varied confidence values (including
    out-of-range values) to ensure the node always clamps to [0.0, 1.0].
    """
    # Generate a varied LLM response (may contain out-of-range confidence)
    mock_response = data.draw(mock_verify_response_strategy())

    # Create mock GeminiService
    mock_gemini = MagicMock()
    mock_gemini.generate.return_value = mock_response

    # Create node and execute
    node = VerifyNode(gemini_service=mock_gemini)
    state = AgentState(
        query="Why does this code exist?",
        code_path="src/example.py",
        evidence=evidence_list,
        hypotheses=hypotheses_list,
    )

    result = node.execute(state)

    # Property: confidence_score is always in [0.0, 1.0]
    assert 0.0 <= result.confidence_score <= 1.0, (
        f"Expected confidence in [0.0, 1.0], got {result.confidence_score}"
    )

    # Verify node sets current_node correctly
    assert result.current_node == NodeType.VERIFY


# ---------------------------------------------------------------------------
# Property 5: Graceful degradation on node failure
# ---------------------------------------------------------------------------

# Feature: codesense, Property 5: Graceful degradation on node failure
@settings(max_examples=100, deadline=None)
@given(
    evidence_list=st.lists(evidence_strategy(), min_size=0, max_size=10),
    hypotheses_list=st.lists(hypothesis_strategy(), min_size=0, max_size=5),
    confidence_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    loop_counter=st.integers(min_value=0, max_value=3),
    failing_node=st.sampled_from(["explore", "hypothesize", "verify", "check_contradictions"]),
)
def test_graceful_degradation_on_node_failure(
    evidence_list: list[Evidence],
    hypotheses_list: list[Hypothesis],
    confidence_score: float,
    loop_counter: int,
    failing_node: str,
) -> None:
    """For any AgentState, when a node raises an exception during execution,
    routing should proceed to synthesize with is_incomplete=True.

    **Validates: Requirements 1.9**

    Tests that the ReasoningGraph node wrappers handle exceptions gracefully
    by setting is_incomplete=True, preserving all evidence gathered prior to
    the failure.
    """
    from codesense.agent.graph import (
        ReasoningGraph,
        _agent_state_to_graph_state,
        _graph_state_to_agent_state,
    )

    # Create a mock GeminiService that raises for the target node
    mock_gemini = MagicMock()

    # Set up the GeminiService to raise an exception
    mock_gemini.generate.side_effect = RuntimeError("Simulated LLM failure")

    # Build the state
    state = AgentState(
        query="Why does this code exist?",
        code_path="src/example.py",
        loop_counter=loop_counter,
        remaining_iterations=3,
        evidence=evidence_list,
        hypotheses=hypotheses_list,
        confidence_score=confidence_score,
    )

    graph_state = _agent_state_to_graph_state(state)

    # Create the ReasoningGraph with the failing mock
    with patch("codesense.agent.graph.ExploreNode") as MockExplore:
        with patch("codesense.agent.graph.CheckContradictionsNode") as MockCheck:
            # Build a real ReasoningGraph but override specific nodes to fail
            reasoning_graph = ReasoningGraph(gemini_service=mock_gemini, mock=True)

            # Test individual node wrappers directly
            if failing_node == "explore":
                # Make explore raise
                reasoning_graph._explore_node.execute = MagicMock(
                    side_effect=RuntimeError("Simulated explore failure")
                )
                result_graph_state = reasoning_graph._explore_wrapper(graph_state)
            elif failing_node == "hypothesize":
                # Make hypothesize raise
                reasoning_graph._hypothesize_node.execute = MagicMock(
                    side_effect=RuntimeError("Simulated hypothesize failure")
                )
                result_graph_state = reasoning_graph._hypothesize_wrapper(graph_state)
            elif failing_node == "verify":
                # Make verify raise
                reasoning_graph._verify_node.execute = MagicMock(
                    side_effect=RuntimeError("Simulated verify failure")
                )
                result_graph_state = reasoning_graph._verify_wrapper(graph_state)
            elif failing_node == "check_contradictions":
                # Make check_contradictions raise
                reasoning_graph._check_contradictions_node.execute = MagicMock(
                    side_effect=RuntimeError("Simulated check_contradictions failure")
                )
                result_graph_state = reasoning_graph._check_contradictions_wrapper(
                    graph_state
                )

            # Convert back to AgentState for assertions
            result_state = _graph_state_to_agent_state(result_graph_state)

            # Property: is_incomplete must be True after a node failure
            assert result_state.is_incomplete is True, (
                f"Expected is_incomplete=True after {failing_node} failure, "
                f"got is_incomplete={result_state.is_incomplete}"
            )

            # Property: evidence gathered prior to failure is preserved
            assert len(result_state.evidence) == len(evidence_list), (
                f"Expected {len(evidence_list)} evidence items preserved, "
                f"got {len(result_state.evidence)}"
            )
