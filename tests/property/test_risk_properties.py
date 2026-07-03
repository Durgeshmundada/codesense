"""Property-based tests for risk score range invariant.

Tests Property 14 from the design document using Hypothesis.

Validates: Requirements 5.9
"""

from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

from codesense.capabilities.risk import (
    _clamp,
    _compute_author_turnover,
    _compute_hack_markers,
    _compute_staleness,
    compute_risk_score,
)
from codesense.models.mcp import CommitRecord


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def commit_record_strategy() -> st.SearchStrategy[CommitRecord]:
    """Strategy to generate varied CommitRecord objects."""
    return st.builds(
        CommitRecord,
        sha=st.text(min_size=7, max_size=40, alphabet="0123456789abcdef"),
        message=st.text(min_size=1, max_size=100),
        author=st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=("L",))),
        timestamp=st.sampled_from([
            "2024-01-15T10:30:00Z",
            "2023-06-20T08:00:00+00:00",
            "2022-03-01T14:45:00Z",
            "2021-11-10T22:15:00+00:00",
            "2020-07-04T06:00:00Z",
            "2019-01-01T00:00:00Z",
        ]),
        diff=st.text(min_size=0, max_size=50),
        files_changed=st.lists(st.text(min_size=1, max_size=30), min_size=0, max_size=5),
    )


def hack_marker_content_strategy() -> st.SearchStrategy[str]:
    """Strategy to generate file content with varied hack markers."""
    markers = ["TODO", "HACK", "FIXME", "temporary", "workaround"]
    marker_lines = st.lists(
        st.sampled_from(markers),
        min_size=0,
        max_size=20,
    )
    normal_lines = st.lists(
        st.text(min_size=0, max_size=80, alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"))),
        min_size=0,
        max_size=20,
    )
    return st.tuples(marker_lines, normal_lines).map(
        lambda t: "\n".join(
            [f"# {m}: fix this" for m in t[0]] + t[1]
        )
    )


# ---------------------------------------------------------------------------
# Property 14: Risk score range invariant
# ---------------------------------------------------------------------------


# Feature: codesense, Property 14: Risk score range invariant
@settings(max_examples=100)
@given(
    commits=st.lists(commit_record_strategy(), min_size=0, max_size=30),
    file_content=hack_marker_content_strategy(),
)
def test_risk_score_range_invariant(commits: list[CommitRecord], file_content: str) -> None:
    """For any combination of risk signal inputs, the computed risk score is in [0.0, 10.0].

    **Validates: Requirements 5.9**

    Tests that compute_risk_score always produces a score within the valid range
    regardless of input variation in commit history, file content, and project structure.
    """
    test_path = "test_file.py"
    project_root = "."

    # Mock the internal functions that require filesystem/git access
    with patch("codesense.capabilities.risk._get_commits", return_value=commits):
        with patch("codesense.capabilities.risk._compute_dependency_count", return_value=1.0):
            with patch("codesense.capabilities.risk._compute_test_coverage", return_value=1.0):
                with patch("codesense.capabilities.risk._compute_hack_markers") as mock_hack:
                    # Compute hack markers from generated content
                    import re
                    _HACK_MARKERS = ["TODO", "HACK", "FIXME", "temporary", "workaround"]
                    total_markers = 0
                    for marker in _HACK_MARKERS:
                        total_markers += len(re.findall(re.escape(marker), file_content, re.IGNORECASE))
                    hack_score = min(total_markers / 5.0, 1.0) * 2.0
                    hack_score = max(0.0, min(2.0, hack_score))
                    mock_hack.return_value = hack_score

                    result = compute_risk_score(
                        path=test_path,
                        project_root=project_root,
                    )

    # Property: risk score is always in [0.0, 10.0]
    assert 0.0 <= result.score <= 10.0, (
        f"Expected risk score in [0.0, 10.0], got {result.score}"
    )

    # Property: all individual signals are in their valid range
    for signal_name, signal_value in result.signals.items():
        assert 0.0 <= signal_value <= 2.0, (
            f"Expected signal '{signal_name}' in [0.0, 2.0], got {signal_value}"
        )


# ---------------------------------------------------------------------------
# Individual signal range tests
# ---------------------------------------------------------------------------


# Feature: codesense, Property 14: Risk score range - author_turnover signal
@settings(max_examples=100)
@given(
    commits=st.lists(commit_record_strategy(), min_size=0, max_size=30),
)
def test_author_turnover_signal_range(commits: list[CommitRecord]) -> None:
    """For any commit history, _compute_author_turnover returns a value in [0.0, 2.0].

    **Validates: Requirements 5.9**
    """
    score = _compute_author_turnover(commits)

    assert 0.0 <= score <= 2.0, (
        f"Expected author_turnover in [0.0, 2.0], got {score}"
    )


# Feature: codesense, Property 14: Risk score range - staleness signal
@settings(max_examples=100)
@given(
    commits=st.lists(commit_record_strategy(), min_size=0, max_size=30),
)
def test_staleness_signal_range(commits: list[CommitRecord]) -> None:
    """For any commit history, _compute_staleness returns a value in [0.0, 2.0].

    **Validates: Requirements 5.9**
    """
    score = _compute_staleness(commits)

    assert 0.0 <= score <= 2.0, (
        f"Expected staleness in [0.0, 2.0], got {score}"
    )


# Feature: codesense, Property 14: Risk score range - hack_markers signal
@settings(max_examples=100)
@given(
    file_content=hack_marker_content_strategy(),
)
def test_hack_markers_signal_range(file_content: str) -> None:
    """For any file content, _compute_hack_markers returns a value in [0.0, 2.0].

    **Validates: Requirements 5.9**

    Uses a temporary file to test the actual hack markers computation.
    """
    import tempfile
    import os

    # Write content to a temporary file for _compute_hack_markers to read
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(file_content)
        tmp_path = f.name

    try:
        score = _compute_hack_markers(tmp_path)
        assert 0.0 <= score <= 2.0, (
            f"Expected hack_markers in [0.0, 2.0], got {score}"
        )
    finally:
        os.unlink(tmp_path)


# Feature: codesense, Property 14: Risk score range - clamp function
@settings(max_examples=100)
@given(
    value=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
)
def test_clamp_always_within_bounds(value: float) -> None:
    """For any input value, _clamp returns a value in [0.0, 10.0].

    **Validates: Requirements 5.9**
    """
    result = _clamp(value, 0.0, 10.0)
    assert 0.0 <= result <= 10.0, (
        f"Expected clamped value in [0.0, 10.0], got {result}"
    )

    # Also test signal-level clamping
    signal_result = _clamp(value, 0.0, 2.0)
    assert 0.0 <= signal_result <= 2.0, (
        f"Expected signal-clamped value in [0.0, 2.0], got {signal_result}"
    )
