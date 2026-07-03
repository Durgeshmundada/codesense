"""Property-based tests for routing and termination invariants.

Tests Properties 3, 4, 20 from the design document using Hypothesis.

Validates: Requirements 1.5, 1.6, 1.7, 1.8, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6
"""

import copy

from hypothesis import given, settings
from hypothesis import strategies as st

from codesense.agent.router import CONFIDENCE_THRESHOLD, MAX_LOOPS, Router
from codesense.models.state import (
    AgentState,
    Conflict,
    ConflictSource,
    SynthesisResult,
    NodeType,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def conflict_source_strategy() -> st.SearchStrategy[ConflictSource]:
    """Strategy to generate varied ConflictSource objects."""
    return st.builds(
        ConflictSource,
        source_id=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))),
        claim=st.text(min_size=1, max_size=100),
    )


def conflict_strategy() -> st.SearchStrategy[Conflict]:
    """Strategy to generate varied Conflict objects with 2+ sources."""
    return st.builds(
        Conflict,
        id=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))),
        sources=st.lists(conflict_source_strategy(), min_size=2, max_size=5),
        description=st.text(min_size=1, max_size=100),
    )


def agent_state_strategy() -> st.SearchStrategy[AgentState]:
    """Strategy to generate varied AgentState objects for routing tests."""
    return st.builds(
        AgentState,
        query=st.just("test query"),
        code_path=st.just("src/test.py"),
        loop_counter=st.integers(min_value=0, max_value=5),
        remaining_iterations=st.integers(min_value=-1, max_value=5),
        confidence_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        conflicts=st.lists(conflict_strategy(), min_size=0, max_size=3),
    )


# ---------------------------------------------------------------------------
# Property 3: Routing determinism
# ---------------------------------------------------------------------------

# Feature: codesense, Property 3: Routing determinism
@settings(max_examples=100)
@given(state=agent_state_strategy())
def test_routing_determinism(state: AgentState) -> None:
    """For any AgentState, the router output is deterministic given
    (has_conflicts, confidence_score, loop_counter, remaining_iterations).
    Same inputs always produce the same routing decision.

    **Validates: Requirements 1.5, 1.6, 1.7**

    Calls the router twice with identical state and verifies the same
    result is returned each time.
    """
    router = Router()

    # Make two independent copies of the state
    state_copy1 = AgentState(
        query=state.query,
        code_path=state.code_path,
        loop_counter=state.loop_counter,
        remaining_iterations=state.remaining_iterations,
        confidence_score=state.confidence_score,
        conflicts=list(state.conflicts),
    )
    state_copy2 = AgentState(
        query=state.query,
        code_path=state.code_path,
        loop_counter=state.loop_counter,
        remaining_iterations=state.remaining_iterations,
        confidence_score=state.confidence_score,
        conflicts=list(state.conflicts),
    )

    # Route both copies
    result1 = router.route(state_copy1)
    result2 = router.route(state_copy2)

    # Property: identical inputs produce identical outputs
    assert result1 == result2, (
        f"Non-deterministic routing: got '{result1}' and '{result2}' for same state "
        f"(loop_counter={state.loop_counter}, remaining_iterations={state.remaining_iterations}, "
        f"confidence={state.confidence_score}, has_conflicts={len(state.conflicts) > 0})"
    )

    # Also verify the result is a valid routing target
    assert result1 in {"synthesize", "hypothesize", "explore"}, (
        f"Invalid routing target: '{result1}'"
    )


# Feature: codesense, Property 3: Routing determinism (exhaustive rules)
@settings(max_examples=100)
@given(state=agent_state_strategy())
def test_routing_rules_exhaustive(state: AgentState) -> None:
    """Verify that routing decisions match the specified rules exhaustively.

    **Validates: Requirements 1.5, 1.6, 1.7**

    Rule 1: loop_counter >= MAX_LOOPS OR remaining_iterations <= 0 → "synthesize"
    Rule 2: conflicts exist → "hypothesize"
    Rule 3: confidence < threshold → "explore"
    Rule 4: else → "synthesize"
    """
    router = Router()

    # Preserve original state values before routing (router may mutate remaining_iterations)
    original_loop_counter = state.loop_counter
    original_remaining_iterations = state.remaining_iterations
    has_conflicts = len(state.conflicts) > 0
    confidence = state.confidence_score

    result = router.route(state)

    # Verify rules in priority order
    if original_loop_counter >= MAX_LOOPS or original_remaining_iterations <= 0:
        assert result == "synthesize", (
            f"Expected 'synthesize' for termination condition "
            f"(loop_counter={original_loop_counter}, remaining_iterations={original_remaining_iterations}), "
            f"got '{result}'"
        )
    elif has_conflicts:
        assert result == "hypothesize", (
            f"Expected 'hypothesize' when conflicts exist, got '{result}'"
        )
    elif confidence < CONFIDENCE_THRESHOLD:
        assert result == "explore", (
            f"Expected 'explore' for low confidence ({confidence} < {CONFIDENCE_THRESHOLD}), "
            f"got '{result}'"
        )
    else:
        assert result == "synthesize", (
            f"Expected 'synthesize' when converged "
            f"(no conflicts, confidence={confidence} >= {CONFIDENCE_THRESHOLD}), "
            f"got '{result}'"
        )


# ---------------------------------------------------------------------------
# Property 4: Loop termination guarantee
# ---------------------------------------------------------------------------

# Feature: codesense, Property 4: Loop termination guarantee
@settings(max_examples=100)
@given(
    confidence_scores=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        min_size=10,
        max_size=10,
    ),
    conflict_patterns=st.lists(
        st.booleans(),
        min_size=10,
        max_size=10,
    ),
)
def test_loop_termination_guarantee(
    confidence_scores: list[float],
    conflict_patterns: list[bool],
) -> None:
    """For any sequence of routing decisions starting from loop_counter=0,
    remaining_iterations=3, the reasoning loop reaches "synthesize" in at most
    3 full cycles regardless of confidence/conflict states.

    **Validates: Requirements 1.8**

    Simulates the routing loop by iterating: each time the router says
    "hypothesize" or "explore", we update the state as the loop would
    (incrementing loop_counter at cycle boundaries) and re-route.
    """
    router = Router()

    state = AgentState(
        query="test query",
        code_path="src/test.py",
        loop_counter=0,
        remaining_iterations=3,
        confidence_score=confidence_scores[0],
        conflicts=[],
    )

    max_routing_decisions = MAX_LOOPS * 4 + 5  # generous upper bound
    decisions_made = 0
    cycle_index = 0

    for i in range(max_routing_decisions):
        # Apply varying conditions from generated data
        idx = min(i, len(confidence_scores) - 1)
        state.confidence_score = confidence_scores[idx]

        # Toggle conflicts based on pattern
        cidx = min(i, len(conflict_patterns) - 1)
        if conflict_patterns[cidx]:
            state.conflicts = [
                Conflict(
                    id=f"conflict-{i}",
                    sources=[
                        ConflictSource(source_id="src1", claim="claim A"),
                        ConflictSource(source_id="src2", claim="claim B"),
                    ],
                    description="test conflict",
                )
            ]
        else:
            state.conflicts = []

        result = router.route(state)
        decisions_made += 1

        if result == "synthesize":
            break

        # Simulate the graph advancing: after a full cycle
        # (explore → hypothesize → verify → check_contradictions),
        # loop_counter increments
        if result == "explore":
            # A full cycle will run, increment loop_counter
            state.loop_counter += 1

    # Property: must have reached synthesize
    assert result == "synthesize", (
        f"Loop did not terminate after {decisions_made} routing decisions. "
        f"Final state: loop_counter={state.loop_counter}, "
        f"remaining_iterations={state.remaining_iterations}"
    )

    # Property: should terminate within a reasonable number of decisions
    # With MAX_LOOPS=3 and remaining_iterations=3, worst case is 3 cycles
    assert decisions_made <= MAX_LOOPS * 2 + 1, (
        f"Loop took too many routing decisions ({decisions_made}) to terminate. "
        f"Expected at most {MAX_LOOPS * 2 + 1} decisions."
    )


# Feature: codesense, Property 4: Loop termination guarantee (remaining_iterations path)
@settings(max_examples=100)
@given(
    confidence_scores=st.lists(
        st.floats(min_value=0.0, max_value=0.69, allow_nan=False),
        min_size=5,
        max_size=5,
    ),
)
def test_loop_termination_via_remaining_iterations(
    confidence_scores: list[float],
) -> None:
    """The loop terminates when remaining_iterations reaches 0, even if
    loop_counter hasn't reached MAX_LOOPS.

    **Validates: Requirements 1.8**

    With always-low confidence and no conflicts, the router decrements
    remaining_iterations each time it routes to "explore". After 3 decrements,
    remaining_iterations=0 forces "synthesize".
    """
    router = Router()

    state = AgentState(
        query="test query",
        code_path="src/test.py",
        loop_counter=0,
        remaining_iterations=3,
        confidence_score=confidence_scores[0],
        conflicts=[],
    )

    decisions = []
    for i in range(10):  # generous upper bound
        state.confidence_score = confidence_scores[min(i, len(confidence_scores) - 1)]
        state.conflicts = []

        result = router.route(state)
        decisions.append(result)

        if result == "synthesize":
            break

    # Property: must terminate
    assert decisions[-1] == "synthesize", (
        f"Loop did not terminate: decisions={decisions}, "
        f"remaining_iterations={state.remaining_iterations}"
    )

    # Property: terminated within remaining_iterations + 1 decisions
    assert len(decisions) <= 4, (
        f"Expected at most 4 decisions (3 explores + 1 synthesize), got {len(decisions)}"
    )


# ---------------------------------------------------------------------------
# Property 20: Terminal conflicts included at termination
# ---------------------------------------------------------------------------

# Feature: codesense, Property 20: Terminal conflicts included at termination
@settings(max_examples=100)
@given(
    conflicts=st.lists(conflict_strategy(), min_size=1, max_size=5),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_terminal_conflicts_route_to_synthesize(
    conflicts: list[Conflict],
    confidence: float,
) -> None:
    """When loop_counter reaches MAX_LOOPS (3) with unresolved conflicts,
    the router returns "synthesize" — conflicts are then passed through
    to SynthesisResult.

    **Validates: Requirements 1.5, 1.6, 1.7, 1.8**

    This verifies the routing decision at termination; the conflicts remain
    in state and are available to the SynthesizeNode for inclusion in output.
    """
    router = Router()

    state = AgentState(
        query="test query",
        code_path="src/test.py",
        loop_counter=MAX_LOOPS,  # At max loops
        remaining_iterations=3,
        confidence_score=confidence,
        conflicts=conflicts,
    )

    result = router.route(state)

    # Property: at MAX_LOOPS, always routes to synthesize regardless of conflicts
    assert result == "synthesize", (
        f"Expected 'synthesize' at MAX_LOOPS with conflicts, got '{result}'"
    )

    # Property: conflicts remain in state (unmodified) for SynthesizeNode
    assert len(state.conflicts) == len(conflicts), (
        f"Conflicts were modified during routing: expected {len(conflicts)}, "
        f"got {len(state.conflicts)}"
    )

    # Verify each conflict is preserved
    for i, conflict in enumerate(state.conflicts):
        assert len(conflict.sources) >= 2, (
            f"Conflict {i} lost sources during routing"
        )


# Feature: codesense, Property 20: Terminal conflicts included at termination
@settings(max_examples=100)
@given(
    conflicts=st.lists(conflict_strategy(), min_size=1, max_size=5),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    remaining_iterations=st.integers(min_value=-1, max_value=0),
)
def test_terminal_conflicts_with_exhausted_iterations(
    conflicts: list[Conflict],
    confidence: float,
    remaining_iterations: int,
) -> None:
    """When remaining_iterations <= 0 with unresolved conflicts,
    the router returns "synthesize" — conflicts are passed through
    to SynthesisResult.

    **Validates: Requirements 1.5, 1.7**

    Tests the other termination path (exhausted iterations) with conflicts.
    """
    router = Router()

    state = AgentState(
        query="test query",
        code_path="src/test.py",
        loop_counter=1,  # Not at max loops but iterations exhausted
        remaining_iterations=remaining_iterations,
        confidence_score=confidence,
        conflicts=conflicts,
    )

    result = router.route(state)

    # Property: with exhausted iterations, always routes to synthesize
    assert result == "synthesize", (
        f"Expected 'synthesize' with remaining_iterations={remaining_iterations}, got '{result}'"
    )

    # Property: conflicts remain intact for synthesis
    assert len(state.conflicts) == len(conflicts), (
        f"Conflicts modified during routing at termination"
    )
