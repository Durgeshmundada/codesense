"""Static analysis models."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FunctionInfo:
    """Information about a function extracted from AST analysis."""

    name: str
    parameters: list[str]
    return_type: Optional[str]
    calls: list[str]
    line_number: int
    is_async: bool = False


@dataclass
class ClassInfo:
    """Information about a class extracted from AST analysis."""

    name: str
    bases: list[str]
    methods: list[FunctionInfo]
    attributes: list[str]
    line_number: int


@dataclass
class ModuleInfo:
    """Information about a module extracted from AST analysis."""

    path: str
    classes: list[ClassInfo]
    functions: list[FunctionInfo]
    imports: list[str]


@dataclass
class CallGraph:
    """A call graph tracing execution from a root function."""

    root: str
    edges: list[tuple[str, str]]  # (caller, callee)
    max_depth_reached: bool = False
    depth: int = 0


@dataclass
class ImportGraph:
    """Import dependency graph for a module."""

    module: str
    internal_deps: list[str]
    external_deps: list[str]
    env_vars: list[str]
    external_apis: list[str]


@dataclass
class RiskAssessment:
    """Risk assessment for a code path. Score is in range [0.0, 10.0]."""

    path: str
    score: float  # 0.0 to 10.0
    signals: dict[str, float] = field(default_factory=dict)
    # Signals: author_turnover, staleness, dependency_count, test_coverage, hack_markers
