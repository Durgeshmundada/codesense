"""FastMCP server exposing CodeSense data retrieval tools.

Registers four MCP tools (get_git_history, get_github_issues, get_pr_comments,
get_related_changes) and handles source selection logic:
- Mock mode (CODESENSE_MOCK=true or mock=True parameter) → MockSource
- Live mode → GitSource + GitHubSource

Mock mode takes precedence over live credentials when both are present.
"""

import dataclasses
import os
from typing import Optional

from fastmcp import FastMCP

from codesense.interfaces import DataSource
from codesense.mcp_server.git_source import GitSource
from codesense.sources.github_source import GitHubSource
from codesense.sources.mock_source import MockSource

mcp = FastMCP("codesense")


def _is_mock_mode(mock: Optional[bool] = None) -> bool:
    """Determine if mock mode should be used.

    Mock mode is active when:
    - The `mock` parameter is explicitly True, OR
    - The CODESENSE_MOCK environment variable is set to "true"

    Mock mode takes precedence over live credentials (Requirement 8.5).
    """
    if mock is True:
        return True
    return os.environ.get("CODESENSE_MOCK", "").lower() == "true"


def _get_mock_source() -> MockSource:
    """Create and return a MockSource instance."""
    return MockSource()


def _get_git_source() -> GitSource:
    """Create a GitSource from the current working directory.

    Raises:
        ValueError: If the current directory is not a valid git repository.
    """
    repo_path = os.environ.get("CODESENSE_REPO_PATH", os.getcwd())
    return GitSource(repo_path=repo_path)


def _get_github_source() -> GitHubSource:
    """Create a GitHubSource using environment credentials.

    Raises:
        ValueError: If GITHUB_TOKEN or GITHUB_REPO environment variables are missing.
    """
    token = os.environ.get("GITHUB_TOKEN", "")
    repo_name = os.environ.get("GITHUB_REPO", "")
    if not token:
        raise ValueError(
            "GITHUB_TOKEN environment variable is required for live GitHub access."
        )
    if not repo_name:
        raise ValueError(
            "GITHUB_REPO environment variable is required (format: owner/repo)."
        )
    return GitHubSource(token=token, repo_name=repo_name)


def _dataclass_to_dict(obj: object) -> dict:
    """Recursively convert a dataclass instance to a dict for JSON transport."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        result = {}
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name)
            result[field.name] = _serialize(value)
        return result
    raise TypeError(f"Expected a dataclass instance, got {type(obj)}")


def _serialize(value: object) -> object:
    """Serialize a value for JSON transport, handling nested dataclasses and lists."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _dataclass_to_dict(value)
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    return value


@mcp.tool()
def get_git_history(code_path: str, limit: int = 50, mock: bool = False) -> dict:
    """Retrieve git commits that modified the specified code path.

    Args:
        code_path: File or directory path within the repository.
        limit: Maximum number of commits to return (default 50).
        mock: If True, use mock data source regardless of environment.

    Returns:
        Dict with 'commits' key containing list of commit records,
        or 'error' key with error message on failure.
    """
    try:
        if _is_mock_mode(mock):
            source: DataSource = _get_mock_source()
        else:
            source = _get_git_source()

        commits = source.get_commits(code_path=code_path, limit=limit)
        return {"commits": [_dataclass_to_dict(c) for c in commits]}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_github_issues(search_term: str, limit: int = 20, mock: bool = False) -> dict:
    """Retrieve GitHub issues matching the search term.

    Args:
        search_term: Code path or keyword to search in issue titles/bodies.
        limit: Maximum number of issues to return (default 20).
        mock: If True, use mock data source regardless of environment.

    Returns:
        Dict with 'issues' key containing list of issue records,
        or 'error' key with error message on failure.
    """
    try:
        if _is_mock_mode(mock):
            source: DataSource = _get_mock_source()
        else:
            source = _get_github_source()

        issues = source.get_issues(search_term=search_term, limit=limit)
        return {"issues": [_dataclass_to_dict(i) for i in issues]}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_pr_comments(code_path: str, limit: int = 20, mock: bool = False) -> dict:
    """Retrieve PR review comments from diffs that include the code path.

    Args:
        code_path: File path to search in PR diffs.
        limit: Maximum number of comments to return (default 20).
        mock: If True, use mock data source regardless of environment.

    Returns:
        Dict with 'pr_comments' key containing list of PR comment records,
        or 'error' key with error message on failure.
    """
    try:
        if _is_mock_mode(mock):
            source: DataSource = _get_mock_source()
        else:
            source = _get_github_source()

        comments = source.get_pr_comments(code_path=code_path, limit=limit)
        return {"pr_comments": [_dataclass_to_dict(c) for c in comments]}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_related_changes(
    code_path: str, min_co_commits: int = 3, limit: int = 20, mock: bool = False
) -> dict:
    """Find files frequently co-modified with the specified code path.

    Args:
        code_path: File path to find co-modification relationships for.
        min_co_commits: Minimum co-commit count threshold (default 3).
        limit: Maximum number of related files to return (default 20).
        mock: If True, use mock data source regardless of environment.

    Returns:
        Dict with 'related_files' key containing list of related file records,
        or 'error' key with error message on failure.
    """
    try:
        if _is_mock_mode(mock):
            source: DataSource = _get_mock_source()
        else:
            source = _get_git_source()

        related = source.get_related_files(
            code_path=code_path, min_co_commits=min_co_commits, limit=limit
        )
        return {"related_files": [_dataclass_to_dict(r) for r in related]}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
