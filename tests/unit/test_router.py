"""Unit tests for the conditional Router."""

import pytest

from codesense.agent.router import (
    CONFIDENCE_THRESHOLD,
    MAX_LOOPS,
    Router,
    route_after_check,
)
from codesense.models.state import AgentState, Conflict, ConflictSource


@pytest.fixture
def router() -> Router:
    """Create a default Router instance."""
    return Router()


@pytest.fixture
def base_state() -> AgentState:
    """Create a base agent state for testing."""
    return AgentState(query="why does this code exist?", code_path="src/auth.py")


class TestRouterConstants:
    """Test module-level constants."""

    def test_max_loops_default(self):
        assert MAX_LOOPS == 3

    def test_confidence_threshold_default(self):
        assert CONFIDENCE_THRESHOLD == 0.7


class TestRouterTermination:
    """Test Rule 1: loop_counter >= MAX_LOOPS OR remaining_iterations <= 0 → synthesize."""

    def test_max_loops_reached(self, router: Router, base_state: AgentState):
        base_state.loop_counter = 3
        assert router.route(base_state) == "synthesize"

    def test_max_loops_exceeded(self, router: Router, base_state: AgentState):
        base_state.loop_counter = 5
        assert router.route(base_state) == "synthesize"

    def test_remaining_iterations_zero(self, router: Router, base_state: AgentState):
        base_state.remaining_iterations = 0
        assert router.route(base_state) == "synthesize"

    def test_remaining_iterations_negative(self, router: Router, base_state: AgentState):
        base_state.remaining_iterations = -1
        assert router.route(base_state) == "synthesize"

    def test_max_loops_overrides_conflicts(self, router: Router, base_state: AgentState):
        """Termination takes priority even if conflicts exist."""
        base_state.loop_counter = 3
        base_state.conflicts = [
            Conflict(
                id="c1",
                sources=[
                    ConflictSource(source_id="s1", claim="claim A"),
                    ConflictSource(source_id="s2", claim="claim B"),
                ],
                description="test conflict",
            )
        ]
        assert router.route(base_state) == "synthesize"

    def test_max_loops_overrides_low_confidence(self, router: Router, base_state: AgentState):
        """Termination takes priority even if confidence is low."""
        base_state.loop_counter = 3
        base_state.confidence_score = 0.2
        assert router.route(base_state) == "synthesize"

    def test_zero_remaining_overrides_conflicts(self, router: Router, base_state: AgentState):
        """remaining_iterations <= 0 takes priority over conflicts."""
        base_state.remaining_iterations = 0
        base_state.conflicts = [
            Conflict(
                id="c1",
                sources=[
                    ConflictSource(source_id="s1", claim="claim A"),
                    ConflictSource(source_id="s2", claim="claim B"),
                ],
                description="test conflict",
            )
        ]
        assert router.route(base_state) == "synthesize"


class TestRouterConflicts:
    """Test Rule 2: conflicts exist → decrement remaining_iterations, return 'hypothesize'."""

    def test_conflicts_route_to_hypothesize(self, router: Router, base_state: AgentState):
        base_state.conflicts = [
            Conflict(
                id="c1",
                sources=[
                    ConflictSource(source_id="s1", claim="claim A"),
                    ConflictSource(source_id="s2", claim="claim B"),
                ],
                description="test conflict",
            )
        ]
        assert router.route(base_state) == "hypothesize"

    def test_conflicts_decrement_remaining_iterations(self, router: Router, base_state: AgentState):
        base_state.remaining_iterations = 3
        base_state.conflicts = [
            Conflict(
                id="c1",
                sources=[
                    ConflictSource(source_id="s1", claim="claim A"),
                    ConflictSource(source_id="s2", claim="claim B"),
                ],
                description="test conflict",
            )
        ]
        router.route(base_state)
        assert base_state.remaining_iterations == 2

    def test_conflicts_take_priority_over_low_confidence(self, router: Router, base_state: AgentState):
        """Conflicts are checked before confidence."""
        base_state.confidence_score = 0.3
        base_state.conflicts = [
            Conflict(
                id="c1",
                sources=[
                    ConflictSource(source_id="s1", claim="claim A"),
                    ConflictSource(source_id="s2", claim="claim B"),
                ],
                description="test conflict",
            )
        ]
        assert router.route(base_state) == "hypothesize"


class TestRouterLowConfidence:
    """Test Rule 3: confidence < threshold → decrement remaining_iterations, return 'explore'."""

    def test_low_confidence_routes_to_explore(self, router: Router, base_state: AgentState):
        base_state.confidence_score = 0.5
        assert router.route(base_state) == "explore"

    def test_zero_confidence_routes_to_explore(self, router: Router, base_state: AgentState):
        base_state.confidence_score = 0.0
        assert router.route(base_state) == "explore"

    def test_just_below_threshold_routes_to_explore(self, router: Router, base_state: AgentState):
        base_state.confidence_score = 0.69
        assert router.route(base_state) == "explore"

    def test_low_confidence_decrements_remaining_iterations(self, router: Router, base_state: AgentState):
        base_state.remaining_iterations = 3
        base_state.confidence_score = 0.5
        router.route(base_state)
        assert base_state.remaining_iterations == 2


class TestRouterConvergence:
    """Test Rule 4: no conflicts AND confidence >= threshold → synthesize."""

    def test_converged_routes_to_synthesize(self, router: Router, base_state: AgentState):
        base_state.confidence_score = 0.8
        base_state.conflicts = []
        assert router.route(base_state) == "synthesize"

    def test_exactly_at_threshold_routes_to_synthesize(self, router: Router, base_state: AgentState):
        base_state.confidence_score = 0.7
        base_state.conflicts = []
        assert router.route(base_state) == "synthesize"

    def test_high_confidence_routes_to_synthesize(self, router: Router, base_state: AgentState):
        base_state.confidence_score = 1.0
        base_state.conflicts = []
        assert router.route(base_state) == "synthesize"

    def test_converged_does_not_decrement(self, router: Router, base_state: AgentState):
        base_state.remaining_iterations = 3
        base_state.confidence_score = 0.8
        base_state.conflicts = []
        router.route(base_state)
        assert base_state.remaining_iterations == 3


class TestRouterDeterminism:
    """Test that the router is deterministic: same inputs always produce same output."""

    def test_same_state_same_result(self, router: Router):
        state1 = AgentState(query="q", code_path="p", confidence_score=0.5)
        state2 = AgentState(query="q", code_path="p", confidence_score=0.5)
        assert router.route(state1) == router.route(state2)

    def test_repeated_calls_same_result(self, router: Router, base_state: AgentState):
        base_state.confidence_score = 0.8
        result1 = router.route(base_state)
        # Reset state for second call
        base_state_copy = AgentState(
            query=base_state.query,
            code_path=base_state.code_path,
            confidence_score=0.8,
        )
        result2 = router.route(base_state_copy)
        assert result1 == result2


class TestRouterCustomThresholds:
    """Test configurable constants."""

    def test_custom_max_loops(self):
        router = Router(max_loops=5)
        state = AgentState(query="q", code_path="p", loop_counter=4, confidence_score=0.5)
        assert router.route(state) == "explore"

    def test_custom_confidence_threshold(self):
        router = Router(confidence_threshold=0.5)
        state = AgentState(query="q", code_path="p", confidence_score=0.6)
        assert router.route(state) == "synthesize"


class TestRouteAfterCheck:
    """Test the standalone route_after_check function."""

    def test_function_routes_correctly(self):
        state = AgentState(query="q", code_path="p", confidence_score=0.8)
        assert route_after_check(state) == "synthesize"

    def test_function_routes_conflicts(self):
        state = AgentState(
            query="q",
            code_path="p",
            conflicts=[
                Conflict(
                    id="c1",
                    sources=[
                        ConflictSource(source_id="s1", claim="A"),
                        ConflictSource(source_id="s2", claim="B"),
                    ],
                    description="conflict",
                )
            ],
        )
        assert route_after_check(state) == "hypothesize"

    def test_function_routes_low_confidence(self):
        state = AgentState(query="q", code_path="p", confidence_score=0.3)
        assert route_after_check(state) == "explore"

    def test_function_routes_max_loops(self):
        state = AgentState(query="q", code_path="p", loop_counter=3)
        assert route_after_check(state) == "synthesize"
