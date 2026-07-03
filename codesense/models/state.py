"""Core state models and enums for the reasoning loop."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class NodeType(Enum):
    """Node types in the reasoning graph."""

    EXPLORE = "explore"
    HYPOTHESIZE = "hypothesize"
    VERIFY = "verify"
    CHECK_CONTRADICTIONS = "check_contradictions"
    SYNTHESIZE = "synthesize"


@dataclass
class ConflictSource:
    """One side of a contradiction."""

    source_id: str
    claim: str


@dataclass
class Conflict:
    """A detected contradiction between sources. No winner field by design."""

    id: str
    sources: list[ConflictSource]  # Always 2 or more
    description: str


@dataclass
class Evidence:
    """A piece of evidence gathered during exploration."""

    source_type: str  # "git_commit", "github_issue", "pr_comment", "decision_unit"
    source_id: str
    content: str
    timestamp: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Hypothesis:
    """A candidate explanation generated during hypothesize phase."""

    id: str
    explanation: str
    supporting_evidence: list[str]  # Evidence source_ids
    confidence: float = 0.0


@dataclass
class SynthesisResult:
    """Final output from the synthesize node."""

    answer: str
    confidence: float
    supporting_evidence: list[str]
    conflicts: list[Conflict]
    reasoning_path: list[NodeType]
    is_incomplete: bool = False


@dataclass
class AgentState:
    """Immutable state passed through the LangGraph reasoning loop."""

    query: str
    code_path: str
    loop_counter: int = 0
    remaining_iterations: int = 3
    evidence: list[Evidence] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    confidence_score: float = 0.0
    conflicts: list[Conflict] = field(default_factory=list)
    synthesis: Optional[SynthesisResult] = None
    current_node: NodeType = NodeType.EXPLORE
    is_incomplete: bool = False
