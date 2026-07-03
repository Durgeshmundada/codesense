"""Unit tests for GitHubSource implementation."""

import sys
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

import pytest

# Ensure the sources subpackage is importable for patch resolution
import codesense.sources.github_source  # noqa: F401


class TestGitHubSourceInit:
    """Tests for GitHubSource constructor validation."""

    @patch("codesense.sources.github_source.Github")
    def test_invalid_token_raises_value_error(self, mock_github_class):
        """Invalid token raises ValueError with descriptive message."""
        from github.GithubException import BadCredentialsException
        from codesense.sources.github_source import GitHubSource

        mock_client = MagicMock()
        mock_github_class.return_value = mock_client
        mock_client.get_user.side_effect = BadCredentialsException(
            401, {"message": "Bad credentials"}, None
        )

        with pytest.raises(ValueError, match="Invalid GitHub token"):
            GitHubSource(token="bad-token", repo_name="owner/repo")

    @patch("codesense.sources.github_source.Github")
    def test_repo_not_found_raises_value_error(self, mock_github_class):
        """Non-existent repository raises ValueError."""
        from github import GithubException
        from codesense.sources.github_source import GitHubSource

        mock_client = MagicMock()
        mock_github_class.return_value = mock_client
        mock_user = MagicMock()
        mock_user.login = "testuser"
        mock_client.get_user.return_value = mock_user
        mock_client.get_repo.side_effect = GithubException(
            404, {"message": "Not Found"}, None
        )

        with pytest.raises(ValueError, match="not found"):
            GitHubSource(token="valid-token", repo_name="owner/nonexistent")

    @patch("codesense.sources.github_source.Github")
    def test_successful_init(self, mock_github_class):
        """Successful initialization with valid token and repo."""
        from codesense.sources.github_source import GitHubSource

        mock_client = MagicMock()
        mock_github_class.return_value = mock_client
        mock_user = MagicMock()
        mock_user.login = "testuser"
        mock_client.get_user.return_value = mock_user
        mock_repo = MagicMock()
        mock_client.get_repo.return_value = mock_repo

        source = GitHubSource(token="valid-token", repo_name="owner/repo")
        assert source._repo == mock_repo

    @patch("codesense.sources.github_source.Github")
    def test_connection_timeout_raises_connection_error(self, mock_github_class):
        """Connection timeout raises ConnectionError with source name."""
        from codesense.sources.github_source import GitHubSource

        mock_client = MagicMock()
        mock_github_class.return_value = mock_client
        mock_client.get_user.side_effect = Exception("Connection timed out")

        with pytest.raises(ConnectionError, match="GitHubSource"):
            GitHubSource(token="valid-token", repo_name="owner/repo")


def _create_source_with_mock(mock_github_class):
    """Helper to create a GitHubSource with mocked GitHub client."""
    from codesense.sources.github_source import GitHubSource

    mock_client = MagicMock()
    mock_github_class.return_value = mock_client
    mock_user = MagicMock()
    mock_user.login = "testuser"
    mock_client.get_user.return_value = mock_user
    mock_repo = MagicMock()
    mock_client.get_repo.return_value = mock_repo

    # Mock rate limit as sufficient
    mock_rate_limit = MagicMock()
    mock_rate_limit.core.remaining = 100
    mock_client.get_rate_limit.return_value = mock_rate_limit

    source = GitHubSource(token="valid-token", repo_name="owner/repo")
    return source, mock_client, mock_repo


class TestGitHubSourceGetIssues:
    """Tests for get_issues method."""

    @patch("codesense.sources.github_source.Github")
    def test_returns_empty_list_when_no_issues_match(self, mock_github_class):
        """Empty results return empty list without error."""
        source, mock_client, _ = _create_source_with_mock(mock_github_class)
        mock_client.search_issues.return_value = iter([])

        result = source.get_issues("nonexistent-term")
        assert result == []

    @patch("codesense.sources.github_source.Github")
    def test_respects_limit_parameter(self, mock_github_class):
        """Results are truncated to the limit parameter."""
        source, mock_client, _ = _create_source_with_mock(mock_github_class)

        # Create 5 mock issues
        mock_issues = []
        for i in range(5):
            issue = MagicMock()
            issue.pull_request = None  # Not a PR
            issue.number = i + 1
            issue.title = f"Issue {i + 1}"
            issue.body = "Test body"
            issue.user.login = "author"
            issue.state = "open"
            issue.labels = []
            issue.get_comments.return_value = iter([])
            mock_issues.append(issue)

        mock_client.search_issues.return_value = iter(mock_issues)

        result = source.get_issues("test", limit=3)
        assert len(result) == 3

    @patch("codesense.sources.github_source.Github")
    def test_includes_issue_comments(self, mock_github_class):
        """Issue comments are included in the returned IssueRecord."""
        source, mock_client, _ = _create_source_with_mock(mock_github_class)

        mock_comment = MagicMock()
        mock_comment.user.login = "commenter"
        mock_comment.body = "This is a comment"
        mock_comment.created_at = datetime(2024, 1, 15, tzinfo=timezone.utc)

        mock_issue = MagicMock()
        mock_issue.pull_request = None
        mock_issue.number = 42
        mock_issue.title = "Bug in auth"
        mock_issue.body = "Auth is broken"
        mock_issue.user.login = "reporter"
        mock_issue.state = "open"
        mock_issue.labels = []
        mock_issue.get_comments.return_value = iter([mock_comment])

        mock_client.search_issues.return_value = iter([mock_issue])

        result = source.get_issues("auth")
        assert len(result) == 1
        assert len(result[0].comments) == 1
        assert result[0].comments[0].author == "commenter"
        assert result[0].comments[0].body == "This is a comment"

    @patch("codesense.sources.github_source.Github")
    def test_skips_pull_requests(self, mock_github_class):
        """Pull requests returned by search are skipped."""
        source, mock_client, _ = _create_source_with_mock(mock_github_class)

        mock_pr = MagicMock()
        mock_pr.pull_request = MagicMock()  # Non-None means it's a PR

        mock_issue = MagicMock()
        mock_issue.pull_request = None
        mock_issue.number = 1
        mock_issue.title = "Real issue"
        mock_issue.body = "Body"
        mock_issue.user.login = "user"
        mock_issue.state = "open"
        mock_issue.labels = []
        mock_issue.get_comments.return_value = iter([])

        mock_client.search_issues.return_value = iter([mock_pr, mock_issue])

        result = source.get_issues("test")
        assert len(result) == 1
        assert result[0].number == 1

    @patch("codesense.sources.github_source.Github")
    def test_returns_correct_issue_fields(self, mock_github_class):
        """IssueRecord fields are populated correctly."""
        source, mock_client, _ = _create_source_with_mock(mock_github_class)

        mock_label = MagicMock()
        mock_label.name = "bug"

        mock_issue = MagicMock()
        mock_issue.pull_request = None
        mock_issue.number = 99
        mock_issue.title = "Critical bug"
        mock_issue.body = "Something is wrong"
        mock_issue.user.login = "developer"
        mock_issue.state = "closed"
        mock_issue.labels = [mock_label]
        mock_issue.get_comments.return_value = iter([])

        mock_client.search_issues.return_value = iter([mock_issue])

        result = source.get_issues("bug")
        assert result[0].number == 99
        assert result[0].title == "Critical bug"
        assert result[0].body == "Something is wrong"
        assert result[0].author == "developer"
        assert result[0].state == "closed"
        assert result[0].labels == ["bug"]


class TestGitHubSourceGetPRComments:
    """Tests for get_pr_comments method."""

    @patch("codesense.sources.github_source.Github")
    def test_returns_empty_list_when_no_comments_match(self, mock_github_class):
        """Empty results return empty list without error."""
        source, _, mock_repo = _create_source_with_mock(mock_github_class)
        mock_repo.get_pulls.return_value = iter([])

        result = source.get_pr_comments("src/auth.py")
        assert result == []

    @patch("codesense.sources.github_source.Github")
    def test_filters_comments_by_file_path(self, mock_github_class):
        """Only comments matching the file path are returned."""
        source, _, mock_repo = _create_source_with_mock(mock_github_class)

        mock_comment_match = MagicMock()
        mock_comment_match.path = "src/auth.py"
        mock_comment_match.line = 42
        mock_comment_match.body = "Review comment"
        mock_comment_match.user.login = "reviewer"
        mock_comment_match.created_at = datetime(2024, 1, 15, tzinfo=timezone.utc)

        mock_comment_no_match = MagicMock()
        mock_comment_no_match.path = "src/utils.py"

        mock_pr = MagicMock()
        mock_pr.number = 10
        mock_pr.get_review_comments.return_value = iter(
            [mock_comment_match, mock_comment_no_match]
        )

        mock_repo.get_pulls.return_value = iter([mock_pr])

        result = source.get_pr_comments("src/auth.py")
        assert len(result) == 1
        assert result[0].file_path == "src/auth.py"
        assert result[0].pr_number == 10

    @patch("codesense.sources.github_source.Github")
    def test_respects_limit_parameter(self, mock_github_class):
        """Results are truncated to the limit parameter."""
        source, _, mock_repo = _create_source_with_mock(mock_github_class)

        mock_comments = []
        for i in range(5):
            comment = MagicMock()
            comment.path = "src/auth.py"
            comment.line = i + 1
            comment.body = f"Comment {i}"
            comment.user.login = "reviewer"
            comment.created_at = datetime(2024, 1, 15, tzinfo=timezone.utc)
            mock_comments.append(comment)

        mock_pr = MagicMock()
        mock_pr.number = 1
        mock_pr.get_review_comments.return_value = iter(mock_comments)

        mock_repo.get_pulls.return_value = iter([mock_pr])

        result = source.get_pr_comments("src/auth.py", limit=2)
        assert len(result) == 2

    @patch("codesense.sources.github_source.Github")
    def test_returns_correct_pr_comment_fields(self, mock_github_class):
        """PRCommentRecord fields are populated correctly."""
        source, _, mock_repo = _create_source_with_mock(mock_github_class)

        ts = datetime(2024, 3, 10, 14, 30, 0, tzinfo=timezone.utc)
        mock_comment = MagicMock()
        mock_comment.path = "src/models/user.py"
        mock_comment.line = 55
        mock_comment.body = "Consider using a dataclass here"
        mock_comment.user.login = "senior-dev"
        mock_comment.created_at = ts

        mock_pr = MagicMock()
        mock_pr.number = 77
        mock_pr.get_review_comments.return_value = iter([mock_comment])

        mock_repo.get_pulls.return_value = iter([mock_pr])

        result = source.get_pr_comments("src/models/user.py")
        assert len(result) == 1
        assert result[0].pr_number == 77
        assert result[0].file_path == "src/models/user.py"
        assert result[0].line_number == 55
        assert result[0].body == "Consider using a dataclass here"
        assert result[0].author == "senior-dev"
        assert result[0].timestamp == ts.isoformat()


class TestGitHubSourceNotImplemented:
    """Tests for methods that delegate to GitSource."""

    @patch("codesense.sources.github_source.Github")
    def test_get_commits_raises_not_implemented(self, mock_github_class):
        """get_commits raises NotImplementedError."""
        from codesense.sources.github_source import GitHubSource

        mock_client = MagicMock()
        mock_github_class.return_value = mock_client
        mock_user = MagicMock()
        mock_user.login = "testuser"
        mock_client.get_user.return_value = mock_user
        mock_client.get_repo.return_value = MagicMock()

        source = GitHubSource(token="valid-token", repo_name="owner/repo")
        with pytest.raises(NotImplementedError):
            source.get_commits("src/main.py")

    @patch("codesense.sources.github_source.Github")
    def test_get_related_files_raises_not_implemented(self, mock_github_class):
        """get_related_files raises NotImplementedError."""
        from codesense.sources.github_source import GitHubSource

        mock_client = MagicMock()
        mock_github_class.return_value = mock_client
        mock_user = MagicMock()
        mock_user.login = "testuser"
        mock_client.get_user.return_value = mock_user
        mock_client.get_repo.return_value = MagicMock()

        source = GitHubSource(token="valid-token", repo_name="owner/repo")
        with pytest.raises(NotImplementedError):
            source.get_related_files("src/main.py")


class TestGitHubSourceRateLimit:
    """Tests for rate-limit handling."""

    @patch("codesense.sources.github_source.time.sleep")
    @patch("codesense.sources.github_source.time.time")
    @patch("codesense.sources.github_source.Github")
    def test_sleeps_when_rate_limit_low(self, mock_github_class, mock_time, mock_sleep):
        """When remaining rate limit < 10, sleeps until reset."""
        from codesense.sources.github_source import GitHubSource

        mock_client = MagicMock()
        mock_github_class.return_value = mock_client
        mock_user = MagicMock()
        mock_user.login = "testuser"
        mock_client.get_user.return_value = mock_user
        mock_client.get_repo.return_value = MagicMock()

        source = GitHubSource(token="valid-token", repo_name="owner/repo")

        # Set up rate limit below threshold
        mock_rate_limit = MagicMock()
        mock_rate_limit.core.remaining = 5
        mock_reset = MagicMock()
        mock_reset.timestamp.return_value = 1000.0
        mock_rate_limit.core.reset = mock_reset
        mock_client.get_rate_limit.return_value = mock_rate_limit

        mock_time.return_value = 990.0  # 10 seconds before reset

        # Make search return empty
        mock_client.search_issues.return_value = iter([])

        source.get_issues("test")
        mock_sleep.assert_called_once_with(11.0)  # (1000 - 990) + 1
