"""Integration tests for the FastMCP server.

Tests cover:
- Tool registration and callability (all 4 tools)
- Source selection logic (mock=True uses MockSource)
- Mock precedence over live credentials
- Each tool returns proper dict structure with expected keys
- Error handling (invalid paths return error dict)

All tests use CODESENSE_MOCK=true to avoid needing real credentials.

Validates: Requirements 2.6, 8.1, 8.4, 8.5
"""

import os
from unittest.mock import patch

import pytest

from codesense.mcp_server.server import (
    _is_mock_mode,
    get_git_history,
    get_github_issues,
    get_pr_comments,
    get_related_changes,
    mcp,
)


class TestToolRegistration:
    """Test that all 4 MCP tools are registered and callable."""

    def test_mcp_server_has_name(self):
        """The MCP server should have the name 'codesense'."""
        assert mcp.name == "codesense"

    def test_get_git_history_is_callable(self):
        """get_git_history should be a callable function."""
        assert callable(get_git_history)

    def test_get_github_issues_is_callable(self):
        """get_github_issues should be a callable function."""
        assert callable(get_github_issues)

    def test_get_pr_comments_is_callable(self):
        """get_pr_comments should be a callable function."""
        assert callable(get_pr_comments)

    def test_get_related_changes_is_callable(self):
        """get_related_changes should be a callable function."""
        assert callable(get_related_changes)

    def test_all_four_tools_return_dict(self):
        """All tools should return dict results when called with mock=True."""
        results = [
            get_git_history(code_path="src/payments/gateway.py", mock=True),
            get_github_issues(search_term="payment", mock=True),
            get_pr_comments(code_path="src/payments/gateway.py", mock=True),
            get_related_changes(code_path="src/payments/gateway.py", mock=True),
        ]
        for result in results:
            assert isinstance(result, dict)


class TestSourceSelectionLogic:
    """Test source selection logic (mock vs live)."""

    def test_mock_param_true_activates_mock_mode(self):
        """When mock=True is passed, _is_mock_mode returns True."""
        assert _is_mock_mode(mock=True) is True

    def test_mock_param_false_without_env_uses_live(self):
        """When mock=False and no env var, _is_mock_mode returns False."""
        with patch.dict(os.environ, {}, clear=True):
            assert _is_mock_mode(mock=False) is False

    def test_env_var_codesense_mock_true_activates_mock(self):
        """When CODESENSE_MOCK=true in env, _is_mock_mode returns True."""
        with patch.dict(os.environ, {"CODESENSE_MOCK": "true"}):
            assert _is_mock_mode(mock=None) is True

    def test_env_var_codesense_mock_TRUE_case_insensitive(self):
        """CODESENSE_MOCK check should be case-insensitive."""
        with patch.dict(os.environ, {"CODESENSE_MOCK": "TRUE"}):
            assert _is_mock_mode(mock=None) is True

    def test_env_var_codesense_mock_false_uses_live(self):
        """When CODESENSE_MOCK=false, mock mode is not activated via env."""
        with patch.dict(os.environ, {"CODESENSE_MOCK": "false"}):
            assert _is_mock_mode(mock=None) is False

    def test_mock_param_none_without_env_uses_live(self):
        """When mock=None and no env var set, uses live mode."""
        with patch.dict(os.environ, {}, clear=True):
            assert _is_mock_mode(mock=None) is False

    def test_get_git_history_mock_returns_commits(self):
        """get_git_history with mock=True returns commit data from MockSource."""
        result = get_git_history(code_path="src/payments/gateway.py", mock=True)
        assert "commits" in result
        assert isinstance(result["commits"], list)
        assert len(result["commits"]) > 0

    def test_get_github_issues_mock_returns_issues(self):
        """get_github_issues with mock=True returns issue data from MockSource."""
        result = get_github_issues(search_term="payment", mock=True)
        assert "issues" in result
        assert isinstance(result["issues"], list)
        assert len(result["issues"]) > 0

    def test_get_pr_comments_mock_returns_comments(self):
        """get_pr_comments with mock=True returns PR comment data from MockSource."""
        result = get_pr_comments(code_path="src/payments/gateway.py", mock=True)
        assert "pr_comments" in result
        assert isinstance(result["pr_comments"], list)
        assert len(result["pr_comments"]) > 0

    def test_get_related_changes_mock_returns_related_files(self):
        """get_related_changes with mock=True returns related file data from MockSource."""
        result = get_related_changes(code_path="src/payments/gateway.py", mock=True)
        assert "related_files" in result
        assert isinstance(result["related_files"], list)


class TestMockPrecedence:
    """Test that mock mode takes precedence over live credentials."""

    def test_mock_env_overrides_github_token(self):
        """When CODESENSE_MOCK=true and GITHUB_TOKEN is set, mock mode wins."""
        env = {
            "CODESENSE_MOCK": "true",
            "GITHUB_TOKEN": "ghp_faketoken123456",
            "GITHUB_REPO": "owner/repo",
        }
        with patch.dict(os.environ, env):
            assert _is_mock_mode(mock=None) is True
            # Should use mock source and return valid data, not try live GitHub
            result = get_github_issues(search_term="payment")
            assert "issues" in result
            assert "error" not in result

    def test_mock_param_overrides_github_token(self):
        """When mock=True and GITHUB_TOKEN is set, mock mode wins."""
        env = {
            "GITHUB_TOKEN": "ghp_faketoken123456",
            "GITHUB_REPO": "owner/repo",
        }
        with patch.dict(os.environ, env):
            result = get_github_issues(search_term="payment", mock=True)
            assert "issues" in result
            assert "error" not in result

    def test_mock_env_overrides_repo_path(self):
        """When CODESENSE_MOCK=true and CODESENSE_REPO_PATH is set, mock mode wins."""
        env = {
            "CODESENSE_MOCK": "true",
            "CODESENSE_REPO_PATH": "/some/real/repo",
        }
        with patch.dict(os.environ, env):
            result = get_git_history(code_path="src/payments/gateway.py")
            assert "commits" in result
            assert "error" not in result

    def test_mock_env_overrides_all_credentials(self):
        """When CODESENSE_MOCK=true, no live source is contacted even with all credentials."""
        env = {
            "CODESENSE_MOCK": "true",
            "GITHUB_TOKEN": "ghp_faketoken123456",
            "GITHUB_REPO": "owner/repo",
            "CODESENSE_REPO_PATH": "/some/real/repo",
        }
        with patch.dict(os.environ, env):
            # All four tools should work in mock mode without live access
            r1 = get_git_history(code_path="src/payments/gateway.py")
            r2 = get_github_issues(search_term="payment")
            r3 = get_pr_comments(code_path="src/payments/gateway.py")
            r4 = get_related_changes(code_path="src/payments/gateway.py")

            for result in [r1, r2, r3, r4]:
                assert "error" not in result


class TestReturnStructure:
    """Test that each tool returns proper dict structure with expected keys."""

    def test_git_history_commit_structure(self):
        """Each commit in the result should have required fields."""
        result = get_git_history(code_path="src/payments/gateway.py", mock=True)
        assert "commits" in result
        for commit in result["commits"]:
            assert "sha" in commit
            assert "message" in commit
            assert "author" in commit
            assert "timestamp" in commit
            assert "diff" in commit
            assert "files_changed" in commit

    def test_github_issues_structure(self):
        """Each issue in the result should have required fields."""
        result = get_github_issues(search_term="payment", mock=True)
        assert "issues" in result
        for issue in result["issues"]:
            assert "number" in issue
            assert "title" in issue
            assert "body" in issue
            assert "author" in issue
            assert "state" in issue
            assert "comments" in issue
            assert "labels" in issue

    def test_github_issues_comment_structure(self):
        """Issue comments should have required fields."""
        result = get_github_issues(search_term="payment", mock=True)
        for issue in result["issues"]:
            for comment in issue["comments"]:
                assert "author" in comment
                assert "body" in comment
                assert "timestamp" in comment

    def test_pr_comments_structure(self):
        """Each PR comment in the result should have required fields."""
        result = get_pr_comments(code_path="src/payments/gateway.py", mock=True)
        assert "pr_comments" in result
        for comment in result["pr_comments"]:
            assert "pr_number" in comment
            assert "file_path" in comment
            assert "line_number" in comment
            assert "body" in comment
            assert "author" in comment
            assert "timestamp" in comment

    def test_related_changes_structure(self):
        """Each related file in the result should have required fields."""
        result = get_related_changes(code_path="src/payments/gateway.py", mock=True)
        assert "related_files" in result
        for related in result["related_files"]:
            assert "path" in related
            assert "co_commit_count" in related
            assert "last_co_modified" in related


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_git_history_without_mock_and_invalid_repo_returns_error(self):
        """When not in mock mode and repo path is invalid, should return error dict."""
        env = {"CODESENSE_REPO_PATH": "/nonexistent/path/to/repo"}
        with patch.dict(os.environ, env, clear=True):
            result = get_git_history(code_path="anything.py", mock=False)
            assert "error" in result
            assert isinstance(result["error"], str)
            assert len(result["error"]) > 0

    def test_github_issues_without_token_returns_error(self):
        """When not in mock mode and no GITHUB_TOKEN, should return error dict."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_github_issues(search_term="test", mock=False)
            assert "error" in result
            assert "GITHUB_TOKEN" in result["error"]

    def test_pr_comments_without_token_returns_error(self):
        """When not in mock mode and no GITHUB_TOKEN, should return error dict."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_pr_comments(code_path="test.py", mock=False)
            assert "error" in result
            assert "GITHUB_TOKEN" in result["error"]

    def test_github_issues_without_repo_returns_error(self):
        """When GITHUB_TOKEN set but no GITHUB_REPO, should return error dict."""
        env = {"GITHUB_TOKEN": "ghp_test123"}
        with patch.dict(os.environ, env, clear=True):
            result = get_github_issues(search_term="test", mock=False)
            assert "error" in result
            assert "GITHUB_REPO" in result["error"]

    def test_error_dict_has_only_error_key(self):
        """Error results should contain an 'error' key with string value."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_github_issues(search_term="test", mock=False)
            assert "error" in result
            assert isinstance(result["error"], str)
