"""Conditional router for the reasoning loop.

Implements deterministic routing based on agent state to decide
the next node in the LangGraph reasoning cycle.
"""

from codesense.models.state import AgentState

# Configurable constants at module level
MAX_LOOPS: int = 3
CONFIDENCE_THRESHOLD: float = 0.7


class Router:
    """Deterministic router for the reasoning loop.

    Routes based on the tuple (has_conflicts, confidence_score, loop_counter,
    remaining_iterations) to decide whether to continue reasoning or synthesize.
    """

    def __init__(
        self,
        max_loops: int = MAX_LOOPS,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self.max_loops = max_loops
        self.confidence_threshold = confidence_threshold

    def route(self, state: AgentState) -> str:
        """Route to the next node based on current agent state.

        Routing logic (evaluated in order):
        1. If loop_counter >= MAX_LOOPS OR remaining_iterations <= 0 → "synthesize"
        2. Else if conflicts exist → decrement remaining_iterations, return "hypothesize"
        3. Else if confidence < threshold → decrement remaining_iterations, return "explore"
        4. Else (no conflicts AND confidence >= threshold) → "synthesize"

        Args:
            state: The current agent state.

        Returns:
            The name of the next node: "synthesize", "hypothesize", or "explore".
        """
        # Rule 1: Termination conditions
        if state.loop_counter >= self.max_loops or state.remaining_iterations <= 0:
            return "synthesize"

        # Rule 2: Conflicts detected → re-hypothesize
        if state.conflicts:
            state.remaining_iterations -= 1
            return "hypothesize"

        # Rule 3: Low confidence → re-explore
        if state.confidence_score < self.confidence_threshold:
            state.remaining_iterations -= 1
            return "explore"

        # Rule 4: Converged — no conflicts and confidence >= threshold
        return "synthesize"


def route_after_check(state: AgentState) -> str:
    """Standalone routing function for use as a LangGraph conditional edge.

    This function can be passed directly to `StateGraph.add_conditional_edges`
    as the routing function after the check_contradictions node.

    Args:
        state: The current agent state.

    Returns:
        The name of the next node: "synthesize", "hypothesize", or "explore".
    """
    router = Router()
    return router.route(state)
