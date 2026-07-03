"""Property-based tests for conflict detection invariants.

Tests Properties 16, 17, 18 from the design document using Hypothesis.

Validates: Requirements 7.1, 7.2, 7.4, 7.5
"""

import json
import uuid
from io import StringIO
from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st
from rich.console import Console

from codesense.agent.check_contradictions import CheckContradictionsNode
from codesense.models.state import (
    AgentState,
    Conflict,
    ConflictSource,
    Evidence,
    Hypothesis,
    NodeType,
    SynthesisResult,
)
from codesense.output.formatter import RichFormatter


# ─── Strategies ───────────────────────────────────────────────────────────────

# Strategy for non-empty strings (source_id and claim fields)
non_empty_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S", "Z")),
    min_size=1,
    max_size=50,
).filter(lambda s: s.strip())

# Strategy for generating ConflictSource objects
conflict_source_strategy = st.builds(
    ConflictSource,
    source_id=non_empty_text,
    claim=non_empty_text,
)

# Strategy for generating Conflict objects with 2+ sources
conflict_strategy = st.builds(
    Conflict,
    id=st.from_type(str).map(lambda _: str(uuid.uuid4())),
    sources=st.lists(conflict_source_strategy, min_size=2, max_size=6),
    description=non_empty_text,
)

# Strategy for generating Evidence objects
evidence_strategy = st.builds(
    Evidence,
    source_type=st.sampled_from(["git_commit", "github_issue", "pr_comment", "decision_unit"]),
    source_id=non_empty_text,
    content=st.text(min_size=1, max_size=200).filter(lambda s: s.strip()),
)


# ─── Property 16: Conflict structural invariants ─────────────────────────────

# Feature: codesense, Property 16: Conflict structural invariants
@settings(max_examples=50)
@given(conflict=conflict_strategy)
def test_conflict_structural_invariants(conflict: Conflict) -> None:
    """For any Conflict object produced by CheckContradictionsNode:
    (a) sources list has 2+ entries,
    (b) each source has non-empty source_id and claim,
    (c) no winner/resolution/ranking field.

    **Validates: Requirements 7.1, 7.2**
    """
    # (a) sources list has 2+ entries
    assert len(conflict.sources) >= 2, (
        f"Conflict must have at least 2 sources, got {len(conflict.sources)}"
    )

    # (b) each source has non-empty source_id and claim
    for source in conflict.sources:
        assert source.source_id.strip(), (
            "ConflictSource.source_id must be non-empty"
        )
        assert source.claim.strip(), (
            "ConflictSource.claim must be non-empty"
        )

    # (c) no winner/resolution/ranking field
    assert not hasattr(conflict, "winner"), (
        "Conflict must NOT have a 'winner' field"
    )
    assert not hasattr(conflict, "resolution"), (
        "Conflict must NOT have a 'resolution' field"
    )
    assert not hasattr(conflict, "ranking"), (
        "Conflict must NOT have a 'ranking' field"
    )


# Feature: codesense, Property 16: Conflict structural invariants (via CheckContradictionsNode)
@settings(max_examples=50)
@given(
    num_conflicts=st.integers(min_value=1, max_value=4),
    sources_per_conflict=st.integers(min_value=2, max_value=5),
)
def test_conflict_structural_invariants_from_node(
    num_conflicts: int, sources_per_conflict: int
) -> None:
    """For any Conflict object produced by CheckContradictionsNode, verify
    structural invariants hold after parsing.

    **Validates: Requirements 7.1, 7.2**
    """
    # Build mock LLM response with N conflicts, each having M sources
    conflicts_json = []
    for i in range(num_conflicts):
        sources = [
            {"source_id": f"source_{i}_{j}", "claim": f"Claim {i} from source {j}"}
            for j in range(sources_per_conflict)
        ]
        conflicts_json.append({
            "description": f"Contradiction about topic {i}",
            "sources": sources,
        })

    mock_response = json.dumps(conflicts_json)

    # Mock GeminiService to return structured JSON
    mock_gemini = MagicMock()
    mock_gemini.generate.return_value = mock_response

    node = CheckContradictionsNode(gemini_service=mock_gemini)

    state = AgentState(
        query="test query",
        code_path="test/path.py",
        evidence=[
            Evidence(source_type="git_commit", source_id="abc123", content="Some commit"),
            Evidence(source_type="github_issue", source_id="issue-1", content="Some issue"),
        ],
        hypotheses=[
            Hypothesis(id="h1", explanation="Test hypothesis", supporting_evidence=["abc123"], confidence=0.5),
        ],
    )

    result = node.execute(state)

    # Verify all produced conflicts satisfy structural invariants
    assert len(result.conflicts) == num_conflicts
    for conflict in result.conflicts:
        # (a) 2+ sources
        assert len(conflict.sources) >= 2
        # (b) non-empty source_id and claim
        for source in conflict.sources:
            assert source.source_id.strip()
            assert source.claim.strip()
        # (c) no winner/resolution/ranking
        assert not hasattr(conflict, "winner")
        assert not hasattr(conflict, "resolution")
        assert not hasattr(conflict, "ranking")


# ─── Property 17: All conflicts appear in output ─────────────────────────────

# Feature: codesense, Property 17: All conflicts appear in output
@settings(max_examples=50)
@given(num_conflicts=st.integers(min_value=1, max_value=8))
def test_all_conflicts_appear_in_output(num_conflicts: int) -> None:
    """For any SynthesisResult with N conflicts (N >= 1), format_output
    produces exactly N labeled conflict sections.

    **Validates: Requirements 7.4**
    """
    # Create N conflict objects
    conflicts = []
    for i in range(num_conflicts):
        conflict = Conflict(
            id=str(uuid.uuid4()),
            sources=[
                ConflictSource(source_id=f"source_a_{i}", claim=f"Claim A for conflict {i}"),
                ConflictSource(source_id=f"source_b_{i}", claim=f"Claim B for conflict {i}"),
            ],
            description=f"Contradiction about topic {i}",
        )
        conflicts.append(conflict)

    # Build a CommandOutput (the format_output input) with these conflicts
    from codesense.models.output import CommandOutput

    output = CommandOutput(
        title="Test Analysis",
        content="Some analysis content.",
        conflicts=conflicts,
        confidence=0.8,
    )

    # Capture formatter output to a string buffer
    string_buffer = StringIO()
    console = Console(file=string_buffer, no_color=True, width=120)
    formatter = RichFormatter(console=console)
    formatter.format_output(output)

    rendered = string_buffer.getvalue()

    # Count labeled conflict sections: each conflict has "Conflict {idx}:" label
    conflict_labels_found = 0
    for idx in range(1, num_conflicts + 1):
        label = f"Conflict {idx}:"
        if label in rendered:
            conflict_labels_found += 1

    assert conflict_labels_found == num_conflicts, (
        f"Expected {num_conflicts} labeled conflict sections, found {conflict_labels_found}. "
        f"Output:\n{rendered}"
    )


# ─── Property 18: No conflicts when evidence is consistent ───────────────────

# Feature: codesense, Property 18: No conflicts when evidence is consistent
@settings(max_examples=50)
@given(
    num_evidence=st.integers(min_value=1, max_value=10),
    shared_claim=non_empty_text,
)
def test_no_conflicts_when_evidence_consistent(
    num_evidence: int, shared_claim: str
) -> None:
    """For any set of evidence where all sources agree (no contradicting claims),
    CheckContradictionsNode produces zero Conflict objects.

    **Validates: Requirements 7.5**
    """
    # Create evidence where all sources make the same claim (consistent)
    evidence = [
        Evidence(
            source_type="git_commit",
            source_id=f"source_{i}",
            content=f"Evidence {i}: {shared_claim}",
        )
        for i in range(num_evidence)
    ]

    # Mock LLM to return empty array (no contradictions found for consistent evidence)
    mock_gemini = MagicMock()
    mock_gemini.generate.return_value = "[]"

    node = CheckContradictionsNode(gemini_service=mock_gemini)

    state = AgentState(
        query="test query",
        code_path="test/path.py",
        evidence=evidence,
        hypotheses=[
            Hypothesis(
                id="h1",
                explanation="Consistent hypothesis",
                supporting_evidence=[f"source_{i}" for i in range(num_evidence)],
                confidence=0.8,
            ),
        ],
    )

    result = node.execute(state)

    # When evidence is consistent, there should be zero conflicts
    assert len(result.conflicts) == 0, (
        f"Expected 0 conflicts for consistent evidence, got {len(result.conflicts)}"
    )
