"""Property-based tests for MCP result limits.

Tests Properties 6 and 7 from the design document using Hypothesis
to verify that MockSource respects configured limits and thresholds.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4**
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from codesense.sources.mock_source import MockSource


# Feature: codesense, Property 6: MCP result count limits
# For any query to MockSource with any limit value (1 <= limit <= 100),
# the number of results returned never exceeds the configured limit
# (50 for commits, 20 for issues, 20 for PR comments, 20 for related files).


@settings(max_examples=100)
@given(
    code_path=st.text(min_size=1, max_size=50),
    limit=st.integers(min_value=1, max_value=100),
)
def test_commits_never_exceed_limit(code_path: str, limit: int) -> None:
    """Property 6a: get_commits results never exceed the requested limit."""
    source = MockSource()
    results = source.get_commits(code_path, limit=limit)
    assert len(results) <= limit


@settings(max_examples=100)
@given(
    search_term=st.text(min_size=1, max_size=50),
    limit=st.integers(min_value=1, max_value=100),
)
def test_issues_never_exceed_limit(search_term: str, limit: int) -> None:
    """Property 6b: get_issues results never exceed the requested limit."""
    source = MockSource()
    results = source.get_issues(search_term, limit=limit)
    assert len(results) <= limit


@settings(max_examples=100)
@given(
    code_path=st.text(min_size=1, max_size=50),
    limit=st.integers(min_value=1, max_value=100),
)
def test_pr_comments_never_exceed_limit(code_path: str, limit: int) -> None:
    """Property 6c: get_pr_comments results never exceed the requested limit."""
    source = MockSource()
    results = source.get_pr_comments(code_path, limit=limit)
    assert len(results) <= limit


@settings(max_examples=100)
@given(
    code_path=st.text(min_size=1, max_size=50),
    limit=st.integers(min_value=1, max_value=100),
)
def test_related_files_never_exceed_limit(code_path: str, limit: int) -> None:
    """Property 6d: get_related_files results never exceed the requested limit."""
    source = MockSource()
    results = source.get_related_files(code_path, limit=limit)
    assert len(results) <= limit


# Feature: codesense, Property 7: Related files co-modification threshold
# For any result returned by MockSource.get_related_files() with any
# min_co_commits value (1 <= min_co_commits <= 20), every RelatedFile
# in the result has co_commit_count >= min_co_commits.


@settings(max_examples=100)
@given(
    code_path=st.text(min_size=1, max_size=50),
    min_co_commits=st.integers(min_value=1, max_value=20),
)
def test_related_files_respect_co_modification_threshold(
    code_path: str, min_co_commits: int
) -> None:
    """Property 7: Every RelatedFile has co_commit_count >= min_co_commits."""
    source = MockSource()
    results = source.get_related_files(code_path, min_co_commits=min_co_commits)
    for related_file in results:
        assert related_file.co_commit_count >= min_co_commits, (
            f"RelatedFile '{related_file.path}' has co_commit_count="
            f"{related_file.co_commit_count} which is less than "
            f"min_co_commits={min_co_commits}"
        )
