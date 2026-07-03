"""Unit tests for data sources: MockSource, GitSource, and GitHubSource.

Tests cover:
- MockSource returns valid fixture data (proper dataclass instances)
- MockSource respects limit parameters
- MockSource filters get_related_files by min_co_commits
- GitSource with mocked gitpython
- GitHubSource timeout and error handling
- Empty result set handling
- Invalid code path error handling

Requirements: 2.1-2.8, 8.1, 8.2
"""

import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from codesense.models.mcp import (
    CommitRecord,
    IssueComment,
    IssueRecord,
    PRCommentRecord,
    RelatedFile,
)


# =============================================================================
# MockSource Tests
# =============================================================================


class TestMockSourceFixtureData:
    """MockSource returns valid fixture data as proper dataclass instances."""

    def test_get_commits_returns_commit_records(self):
        """get_commits returns a list of CommitRecord dataclass instances."""
        from codesense.sources.mock_source import MockSource

        source = MockSource()
        results = source.get_commits("src/payments/gateway.py")

        assert len(results) > 0
        for record in results:
            assert isinstance(record, CommitRecord)

    def test_commit_records_have_required_fields(self):
        """Each CommitRecord has non-empty sha, message, author, timestamp, diff."""
        from codesense.sources.mock_source import MockSource

        source = MockSource()
        results = source.get_commits("src/payments/gateway.py")

        for record in results:
            assert record.sha, "sha must be non-empty"
            assert record.message, "message must be non-empty"
            assert record.author, "author must be non-empty"
            assert record.timestamp, "timestamp must be non-empty"
            assert isinstance(record.files_changed, list)
            assert len(record.files_changed) > 0

    def test_get_issues_returns_issue_records(self):
        """get_issues returns a list of IssueRecord dataclass instances."""
        from codesense.sources.mock_source import MockSource

        source = MockSource()
        results = source.get_issues("payment")

        assert len(results) > 0
        for record in results:
            assert isinstance(record, IssueRecord)

    def test_issue_records_have_comments(self):
        """IssueRecords include IssueComment dataclass instances."""
        from codesense.sources.mock_source import MockSource

        source = MockSource()
        results = source.get_issues("payment")

        # At least one issue should have comments
        has_comments = any(len(r.comments) > 0 for r in results)
        assert has_comments, "At least one issue should have comments"

        for record in results:
            for comment in record.comments:
                assert isinstance(comment, IssueComment)
                assert comment.author
                assert comment.body
                assert comment.timestamp

    def test_get_pr_comments_returns_pr_comment_records(self):
        """get_pr_comments returns a list of PRCommentRecord instances."""
        from codesense.sources.mock_source import MockSource

        source = MockSource()
        results = source.get_pr_comments("src/payments/gateway.py")

        assert len(results) > 0
        for record in results:
            assert isinstance(record, PRCommentRecord)
            assert record.pr_number > 0
            assert record.file_path
            assert record.body
            assert record.author
            assert record.timestamp

    def test_get_related_files_returns_related_file_records(self):
        """get_related_files returns a list of RelatedFile instances."""
        from codesense.sources.mock_source import MockSource

        source = MockSource()
        results = source.get_related_files("src/payments/gateway.py")

        assert len(results) > 0
        for record in results:
            assert isinstance(record, RelatedFile)
            assert record.path
            assert record.co_commit_count > 0
            assert record.last_co_modified


class TestMockSourceLimits:
    """MockSource respects limit parameters."""

    def test_get_commits_respects_limit(self):
        """get_commits truncates results to the limit parameter."""
        from codesense.sources.mock_source import MockSource

        source = MockSource()
        results = source.get_commits("src/payments/gateway.py", limit=2)
        assert len(results) <= 2

    def test_get_commits_limit_one(self):
        """get_commits with limit=1 returns at most 1 result."""
        from codesense.sources.mock_source import MockSource

        source = MockSource()
        results = source.get_commits("src/payments/gateway.py", limit=1)
        assert len(results) <= 1

    def test_get_issues_respects_limit(self):
        """get_issues truncates results to the limit parameter."""
        from codesense.sources.mock_source import MockSource

        source = MockSource()
        results = source.get_issues("payment", limit=2)
        assert len(results) <= 2

    def test_get_pr_comments_respects_limit(self):
        """get_pr_comments truncates results to the limit parameter."""
        from codesense.sources.mock_source import MockSource

        source = MockSource()
        results = source.get_pr_comments("src/payments/gateway.py", limit=3)
        assert len(results) <= 3

    def test_get_related_files_respects_limit(self):
        """get_related_files truncates results to the limit parameter."""
        from codesense.sources.mock_source import MockSource

        source = MockSource()
        results = source.get_related_files("src/payments/gateway.py", limit=2)
        assert len(results) <= 2


class TestMockSourceFiltering:
    """MockSource filters correctly by parameters."""

    def test_get_related_files_filters_by_min_co_commits(self):
        """Only files with co_commit_count >= min_co_commits are returned."""
        from codesense.sources.mock_source import MockSource

        source = MockSource()
        results = source.get_related_files(
            "src/payments/gateway.py", min_co_commits=5
        )

        for record in results:
            assert record.co_commit_count >= 5

    def test_get_related_files_high_threshold_returns_fewer(self):
        """Higher min_co_commits threshold returns fewer results."""
        from codesense.sources.mock_source import MockSource

        source = MockSource()
        low_threshold = source.get_related_files(
            "src/payments/gateway.py", min_co_commits=3
        )
        high_threshold = source.get_related_files(
            "src/payments/gateway.py", min_co_commits=7
        )

        assert len(high_threshold) <= len(low_threshold)

    def test_get_related_files_very_high_threshold_returns_empty(self):
        """Threshold higher than any fixture data returns empty list."""
        from codesense.sources.mock_source import MockSource

        source = MockSource()
        results = source.get_related_files(
            "src/payments/gateway.py", min_co_commits=100
        )
        assert results == []

    def test_get_commits_filters_by_code_path(self):
        """get_commits returns commits containing the code path in files_changed."""
        from codesense.sources.mock_source import MockSource

        source = MockSource()
        results = source.get_commits("src/payments/gateway.py")

        # All results should have the path in files_changed
        for record in results:
            assert any(
                "src/payments/gateway.py" in f for f in record.files_changed
            )

    def test_get_issues_filters_by_search_term(self):
        """get_issues returns issues matching the search term in title/body."""
        from codesense.sources.mock_source import MockSource

        source = MockSource()
        results = source.get_issues("race condition")

        # At least one issue should be about race conditions
        assert len(results) > 0
        has_match = any(
            "race condition" in r.title.lower() or "race condition" in r.body.lower()
            for r in results
        )
        assert has_match

    def test_get_pr_comments_filters_by_code_path(self):
        """get_pr_comments returns comments matching the code path."""
        from codesense.sources.mock_source import MockSource

        source = MockSource()
        results = source.get_pr_comments("src/payments/gateway.py")

        for record in results:
            assert "src/payments/gateway.py" in record.file_path


class TestMockSourceEmptyResults:
    """MockSource handles no-match scenarios gracefully."""

    def test_get_commits_no_match_returns_all(self):
        """When no commits match the code path, returns all commits (demo-friendly)."""
        from codesense.sources.mock_source import MockSource

        source = MockSource()
        results = source.get_commits("nonexistent/file.py")
        # Demo-friendly behavior: returns all commits when no path matches
        assert len(results) > 0

    def test_get_issues_no_match_returns_all(self):
        """When no issues match the search term, returns all issues (demo-friendly)."""
        from codesense.sources.mock_source import MockSource

        source = MockSource()
        results = source.get_issues("zzz_nonexistent_term_zzz")
        # Demo-friendly behavior: returns all issues when no term matches
        assert len(results) > 0

    def test_get_pr_comments_no_match_returns_all(self):
        """When no PR comments match the path, returns all comments (demo-friendly)."""
        from codesense.sources.mock_source import MockSource

        source = MockSource()
        results = source.get_pr_comments("nonexistent/file.py")
        # Demo-friendly behavior: returns all PR comments when no path matches
        assert len(results) > 0


# =============================================================================
# GitSource Tests (with mocked gitpython)
# =============================================================================


class TestGitSourceInit:
    """GitSource constructor validation."""

    @patch("codesense.mcp_server.git_source.Repo")
    def test_valid_repo_path_initializes(self, mock_repo_class):
        """Valid git repository path initializes successfully."""
        from codesense.mcp_server.git_source import GitSource

        mock_repo_class.return_value = MagicMock()
        source = GitSource("/path/to/valid/repo")
        assert source._repo is not None

    def test_invalid_repo_path_raises_value_error(self):
        """Non-existent or non-git path raises ValueError."""
        from codesense.mcp_server.git_source import GitSource

        with pytest.raises(ValueError, match="Invalid git repository path"):
            GitSource("/nonexistent/path/that/does/not/exist")

    @patch("codesense.mcp_server.git_source.Repo")
    def test_invalid_git_repo_raises_value_error(self, mock_repo_class):
        """Directory that is not a git repo raises ValueError."""
        from git.exc import InvalidGitRepositoryError
        from codesense.mcp_server.git_source import GitSource

        mock_repo_class.side_effect = InvalidGitRepositoryError("not a git repo")

        with pytest.raises(ValueError, match="Invalid git repository path"):
            GitSource("/some/directory")


class TestGitSourceGetCommits:
    """GitSource.get_commits tests with mocked gitpython."""

    @patch("codesense.mcp_server.git_source.Repo")
    def test_returns_commit_records(self, mock_repo_class):
        """get_commits returns properly populated CommitRecord instances."""
        from codesense.mcp_server.git_source import GitSource

        mock_repo = MagicMock()
        mock_repo_class.return_value = mock_repo

        # Mock path validation
        source = GitSource("/repo")
        source._repo_path = Path("/repo")

        # Create a mock commit
        mock_commit = MagicMock()
        mock_commit.hexsha = "abc123"
        mock_commit.message = "fix: something\n"
        mock_commit.author = "dev"
        mock_commit.committed_date = 1700000000
        mock_commit.parents = []
        mock_commit.tree.__truediv__ = MagicMock(side_effect=KeyError)

        # Mock _validate_code_path to pass
        source._validate_code_path = MagicMock()
        source._iter_commits_for_path = MagicMock(return_value=[mock_commit])
        source._get_diff_summary = MagicMock(return_value="diff content")
        source._get_files_changed = MagicMock(return_value=["src/main.py"])

        results = source.get_commits("src/main.py", limit=10)

        assert len(results) == 1
        assert isinstance(results[0], CommitRecord)
        assert results[0].sha == "abc123"
        assert results[0].message == "fix: something"
        assert results[0].author == "dev"
        assert results[0].files_changed == ["src/main.py"]

    @patch("codesense.mcp_server.git_source.Repo")
    def test_respects_limit(self, mock_repo_class):
        """get_commits respects the limit parameter."""
        from codesense.mcp_server.git_source import GitSource

        mock_repo = MagicMock()
        mock_repo_class.return_value = mock_repo

        source = GitSource("/repo")
        source._repo_path = Path("/repo")
        source._validate_code_path = MagicMock()

        # Create 5 mock commits
        mock_commits = []
        for i in range(5):
            c = MagicMock()
            c.hexsha = f"sha{i}"
            c.message = f"commit {i}"
            c.author = "dev"
            c.committed_date = 1700000000 + i
            c.parents = []
            mock_commits.append(c)

        source._iter_commits_for_path = MagicMock(return_value=mock_commits)
        source._get_diff_summary = MagicMock(return_value="")
        source._get_files_changed = MagicMock(return_value=["f.py"])

        results = source.get_commits("f.py", limit=3)
        assert len(results) == 3

    @patch("codesense.mcp_server.git_source.Repo")
    def test_invalid_code_path_raises_value_error(self, mock_repo_class):
        """get_commits raises ValueError for non-existent code path."""
        from codesense.mcp_server.git_source import GitSource
        from git.exc import GitCommandError

        mock_repo = MagicMock()
        mock_repo_class.return_value = mock_repo

        source = GitSource("/repo")
        source._repo_path = Path("/repo")

        # Mock that the path doesn't exist in working tree or git history
        with patch.object(Path, "exists", return_value=False):
            mock_repo.git.log.return_value = ""
            with pytest.raises(ValueError, match="does not exist"):
                source.get_commits("nonexistent/file.py")

    @patch("codesense.mcp_server.git_source.Repo")
    def test_timeout_raises_runtime_error(self, mock_repo_class):
        """Timeout during git operation raises RuntimeError with source name."""
        from codesense.mcp_server.git_source import GitSource, GitOperationTimeout

        mock_repo = MagicMock()
        mock_repo_class.return_value = mock_repo

        source = GitSource("/repo")
        source._repo_path = Path("/repo")
        source._validate_code_path = MagicMock()
        source._iter_commits_for_path = MagicMock(
            side_effect=GitOperationTimeout("timed out")
        )

        with pytest.raises(RuntimeError, match="timed out"):
            source.get_commits("src/main.py")

    @patch("codesense.mcp_server.git_source.Repo")
    def test_empty_repo_returns_empty_list(self, mock_repo_class):
        """Repository with no commits returns empty list."""
        from codesense.mcp_server.git_source import GitSource

        mock_repo = MagicMock()
        mock_repo_class.return_value = mock_repo

        source = GitSource("/repo")
        source._repo_path = Path("/repo")
        source._validate_code_path = MagicMock()
        source._iter_commits_for_path = MagicMock(return_value=[])

        results = source.get_commits("src/main.py")
        assert results == []


class TestGitSourceGetRelatedFiles:
    """GitSource.get_related_files tests."""

    @patch("codesense.mcp_server.git_source.Repo")
    def test_filters_by_min_co_commits(self, mock_repo_class):
        """Only files with co_commit_count >= min_co_commits are returned."""
        from codesense.mcp_server.git_source import GitSource

        mock_repo = MagicMock()
        mock_repo_class.return_value = mock_repo

        source = GitSource("/repo")
        source._repo_path = Path("/repo")
        source._validate_code_path = MagicMock()

        # Create mock commits where file_a co-appears 4 times and file_b 2 times
        commits = []
        for i in range(4):
            c = MagicMock()
            c.committed_date = 1700000000 + i
            c.parents = [MagicMock()]
            commits.append(c)

        source._iter_commits_for_path = MagicMock(return_value=commits)

        # For the first 4 commits, target and file_a appear; only 2 have file_b
        call_count = [0]

        def mock_get_files(commit):
            idx = call_count[0]
            call_count[0] += 1
            if idx < 4:
                files = ["target.py", "file_a.py"]
                if idx < 2:
                    files.append("file_b.py")
                return files
            return []

        source._get_files_changed = mock_get_files

        results = source.get_related_files("target.py", min_co_commits=3)

        # file_a has 4 co-commits (>= 3), file_b has 2 (< 3)
        assert all(r.co_commit_count >= 3 for r in results)
        paths = [r.path for r in results]
        assert "file_a.py" in paths
        assert "file_b.py" not in paths

    @patch("codesense.mcp_server.git_source.Repo")
    def test_respects_limit(self, mock_repo_class):
        """get_related_files respects the limit parameter."""
        from codesense.mcp_server.git_source import GitSource

        mock_repo = MagicMock()
        mock_repo_class.return_value = mock_repo

        source = GitSource("/repo")
        source._repo_path = Path("/repo")
        source._validate_code_path = MagicMock()

        # Create enough commits to generate many related files
        commits = []
        for i in range(10):
            c = MagicMock()
            c.committed_date = 1700000000 + i
            c.parents = [MagicMock()]
            commits.append(c)

        source._iter_commits_for_path = MagicMock(return_value=commits)
        source._get_files_changed = MagicMock(
            return_value=["target.py", "a.py", "b.py", "c.py", "d.py"]
        )

        results = source.get_related_files("target.py", min_co_commits=1, limit=2)
        assert len(results) <= 2

    @patch("codesense.mcp_server.git_source.Repo")
    def test_invalid_code_path_raises_value_error(self, mock_repo_class):
        """get_related_files raises ValueError for non-existent code path."""
        from codesense.mcp_server.git_source import GitSource

        mock_repo = MagicMock()
        mock_repo_class.return_value = mock_repo

        source = GitSource("/repo")
        source._repo_path = Path("/repo")

        with patch.object(Path, "exists", return_value=False):
            mock_repo.git.log.return_value = ""
            with pytest.raises(ValueError, match="does not exist"):
                source.get_related_files("nonexistent/file.py")

    @patch("codesense.mcp_server.git_source.Repo")
    def test_timeout_raises_runtime_error(self, mock_repo_class):
        """Timeout during related files analysis raises RuntimeError."""
        from codesense.mcp_server.git_source import GitSource, GitOperationTimeout

        mock_repo = MagicMock()
        mock_repo_class.return_value = mock_repo

        source = GitSource("/repo")
        source._repo_path = Path("/repo")
        source._validate_code_path = MagicMock()
        source._iter_commits_for_path = MagicMock(
            side_effect=GitOperationTimeout("timed out")
        )

        with pytest.raises(RuntimeError, match="timed out"):
            source.get_related_files("src/main.py")


class TestGitSourceNotImplemented:
    """GitSource methods not applicable to git repos."""

    @patch("codesense.mcp_server.git_source.Repo")
    def test_get_issues_raises_not_implemented(self, mock_repo_class):
        """get_issues raises NotImplementedError for GitSource."""
        from codesense.mcp_server.git_source import GitSource

        mock_repo_class.return_value = MagicMock()
        source = GitSource("/repo")

        with pytest.raises(NotImplementedError, match="GitSource"):
            source.get_issues("search term")

    @patch("codesense.mcp_server.git_source.Repo")
    def test_get_pr_comments_raises_not_implemented(self, mock_repo_class):
        """get_pr_comments raises NotImplementedError for GitSource."""
        from codesense.mcp_server.git_source import GitSource

        mock_repo_class.return_value = MagicMock()
        source = GitSource("/repo")

        with pytest.raises(NotImplementedError, match="GitSource"):
            source.get_pr_comments("src/main.py")


# =============================================================================
# GitHubSource Additional Tests (timeout and error handling)
# =============================================================================


class TestGitHubSourceTimeoutHandling:
    """GitHubSource timeout and connection error handling."""

    @patch("codesense.sources.github_source.Github")
    def test_get_issues_timeout_raises_connection_error(self, mock_github_class):
        """Connection timeout during get_issues raises ConnectionError."""
        from codesense.sources.github_source import GitHubSource

        mock_client = MagicMock()
        mock_github_class.return_value = mock_client
        mock_user = MagicMock()
        mock_user.login = "testuser"
        mock_client.get_user.return_value = mock_user
        mock_client.get_repo.return_value = MagicMock()

        source = GitHubSource(token="valid-token", repo_name="owner/repo")

        # Mock rate limit as OK
        mock_rate_limit = MagicMock()
        mock_rate_limit.core.remaining = 100
        mock_client.get_rate_limit.return_value = mock_rate_limit

        # Simulate timeout during search
        mock_client.search_issues.side_effect = Exception(
            "Connection timed out after 30s"
        )

        with pytest.raises(ConnectionError, match="GitHubSource"):
            source.get_issues("test")

    @patch("codesense.sources.github_source.Github")
    def test_get_pr_comments_timeout_raises_connection_error(self, mock_github_class):
        """Connection timeout during get_pr_comments raises ConnectionError."""
        from codesense.sources.github_source import GitHubSource

        mock_client = MagicMock()
        mock_github_class.return_value = mock_client
        mock_user = MagicMock()
        mock_user.login = "testuser"
        mock_client.get_user.return_value = mock_user
        mock_repo = MagicMock()
        mock_client.get_repo.return_value = mock_repo

        source = GitHubSource(token="valid-token", repo_name="owner/repo")

        # Mock rate limit as OK
        mock_rate_limit = MagicMock()
        mock_rate_limit.core.remaining = 100
        mock_client.get_rate_limit.return_value = mock_rate_limit

        # Simulate timeout during PR fetch
        mock_repo.get_pulls.side_effect = Exception("connect timed out")

        with pytest.raises(ConnectionError, match="GitHubSource"):
            source.get_pr_comments("src/main.py")


class TestGitHubSourceErrorHandling:
    """GitHubSource error message handling."""

    @patch("codesense.sources.github_source.Github")
    def test_get_issues_non_timeout_error_returns_empty(self, mock_github_class):
        """Non-timeout errors in get_issues return empty list gracefully."""
        from codesense.sources.github_source import GitHubSource

        mock_client = MagicMock()
        mock_github_class.return_value = mock_client
        mock_user = MagicMock()
        mock_user.login = "testuser"
        mock_client.get_user.return_value = mock_user
        mock_client.get_repo.return_value = MagicMock()

        source = GitHubSource(token="valid-token", repo_name="owner/repo")

        mock_rate_limit = MagicMock()
        mock_rate_limit.core.remaining = 100
        mock_client.get_rate_limit.return_value = mock_rate_limit

        # Non-timeout error
        mock_client.search_issues.side_effect = Exception("some API error")

        result = source.get_issues("test")
        assert result == []

    @patch("codesense.sources.github_source.Github")
    def test_get_pr_comments_non_timeout_error_returns_empty(self, mock_github_class):
        """Non-timeout errors in get_pr_comments return empty list gracefully."""
        from codesense.sources.github_source import GitHubSource

        mock_client = MagicMock()
        mock_github_class.return_value = mock_client
        mock_user = MagicMock()
        mock_user.login = "testuser"
        mock_client.get_user.return_value = mock_user
        mock_repo = MagicMock()
        mock_client.get_repo.return_value = mock_repo

        source = GitHubSource(token="valid-token", repo_name="owner/repo")

        mock_rate_limit = MagicMock()
        mock_rate_limit.core.remaining = 100
        mock_client.get_rate_limit.return_value = mock_rate_limit

        # Non-timeout error
        mock_repo.get_pulls.side_effect = Exception("some API error")

        result = source.get_pr_comments("src/main.py")
        assert result == []

    @patch("codesense.sources.github_source.Github")
    def test_init_connection_timeout_includes_source_name(self, mock_github_class):
        """Connection timeout during init includes 'GitHubSource' in message."""
        from codesense.sources.github_source import GitHubSource

        mock_client = MagicMock()
        mock_github_class.return_value = mock_client
        mock_client.get_user.side_effect = Exception("Connection timed out")

        with pytest.raises(ConnectionError, match="GitHubSource"):
            GitHubSource(token="valid-token", repo_name="owner/repo")


class TestGitHubSourceEmptyResults:
    """GitHubSource empty result set handling."""

    @patch("codesense.sources.github_source.Github")
    def test_get_issues_empty_search_returns_empty_list(self, mock_github_class):
        """Empty search results return empty list without error."""
        from codesense.sources.github_source import GitHubSource

        mock_client = MagicMock()
        mock_github_class.return_value = mock_client
        mock_user = MagicMock()
        mock_user.login = "testuser"
        mock_client.get_user.return_value = mock_user
        mock_client.get_repo.return_value = MagicMock()

        source = GitHubSource(token="valid-token", repo_name="owner/repo")

        mock_rate_limit = MagicMock()
        mock_rate_limit.core.remaining = 100
        mock_client.get_rate_limit.return_value = mock_rate_limit

        mock_client.search_issues.return_value = iter([])

        result = source.get_issues("zzz_nonexistent_zzz")
        assert result == []

    @patch("codesense.sources.github_source.Github")
    def test_get_pr_comments_no_prs_returns_empty_list(self, mock_github_class):
        """No pull requests returns empty list without error."""
        from codesense.sources.github_source import GitHubSource

        mock_client = MagicMock()
        mock_github_class.return_value = mock_client
        mock_user = MagicMock()
        mock_user.login = "testuser"
        mock_client.get_user.return_value = mock_user
        mock_repo = MagicMock()
        mock_client.get_repo.return_value = mock_repo

        source = GitHubSource(token="valid-token", repo_name="owner/repo")

        mock_rate_limit = MagicMock()
        mock_rate_limit.core.remaining = 100
        mock_client.get_rate_limit.return_value = mock_rate_limit

        mock_repo.get_pulls.return_value = iter([])

        result = source.get_pr_comments("nonexistent/file.py")
        assert result == []
