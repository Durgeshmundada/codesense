"""Reasoning agent components: nodes, router, and LangGraph graph.

Lazy imports are used to avoid circular import issues between graph.py,
nodes.py, and check_contradictions.py — all of which reside in this package.
"""


def __getattr__(name: str):
    """Lazy import of agent submodules to prevent circular import deadlocks."""
    if name == "CheckContradictionsNode":
        from codesense.agent.check_contradictions import CheckContradictionsNode
        return CheckContradictionsNode
    if name == "ReasoningGraph":
        from codesense.agent.graph import ReasoningGraph
        return ReasoningGraph
    if name in ("ExploreNode", "HypothesizeNode", "SynthesizeNode", "VerifyNode"):
        from codesense.agent import nodes as _nodes
        return getattr(_nodes, name)
    if name == "Router":
        from codesense.agent.router import Router
        return Router
    if name == "route_after_check":
        from codesense.agent.router import route_after_check
        return route_after_check
    raise AttributeError(f"module 'codesense.agent' has no attribute {name!r}")


__all__ = [
    "CheckContradictionsNode",
    "ExploreNode",
    "HypothesizeNode",
    "ReasoningGraph",
    "Router",
    "SynthesizeNode",
    "VerifyNode",
    "route_after_check",
]
