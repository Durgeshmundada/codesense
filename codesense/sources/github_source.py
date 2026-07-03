"""GitHub data source using PyGithub for API access.

Implements the DataSource ABC for live GitHub issue and PR comment retrieval.
Provides rate-limit awareness, connection timeout handling, and descriptive
error messages for authentication or repository resolution failures.
"""

import time
from typing import Optional

from github import Auth, Github, GithubException
from github.GithubException import BadCredentialsException

from codesense.interfaces import DataSource
from codesense.models.mcp import (
    CommitRecord,
    IssueComment,
    IssueRecord,
    PRCommentRecord,
    RelatedFile,
)

# Connection timeout in seconds (Requirement 2.5)
_CONNECTION_TIMEOUT = 30

# Minimum remaining rate-limit calls before sleeping (Requirement 2.8)
_RATE_LIMIT_THRESHOLD = 10


class GitHubSource(DataSource):
    """Live GitHub data source using PyGithub.

    Retrieves issues and PR review comments from the GitHub API.
    Delegates get_commits() and get_related_files() to GitSource
    (raises NotImplementedError here).

    Args:
        token: GitHub personal access token for authentication.
        repo_name: Repository in "owner/repo" format.

    Raises:
        ValueError: If the token is invalid or the repository cannot be found.
        ConnectionError: If the connection to GitHub fails within the timeout.
    """

    def __init__(self, token: str, repo_name: str) -> None:
        self._repo_name = repo_name
        try:
            auth = Auth.Token(token)
            self._client = Github(auth=auth, timeout=_CONNECTION_TIMEOUT)
            # Validate credentials by fetching the authenticated user
            self._client.get_user().login
        except BadCredentialsException:
            raise ValueError(
                f"Invalid GitHub token: authentication failed for repository '{repo_name}'."
            )
        except Exception as e:
            if "timed out" in str(e).lower() or "connect" in str(e).lower():
                raise ConnectionError(
                    f"GitHubSource: failed to connect to GitHub within "
                    f"{_CONNECTION_TIMEOUT}s — {e}"
                )
            raise ValueError(
                f"Invalid GitHub token or connection issue: {e}"
            )

        try:
            self._repo = self._client.get_repo(repo_name)
        except GithubException as e:
            if e.status == 404:
                raise ValueError(
                    f"Repository '{repo_name}' not found. "
                    f"Verify the owner/repo format and access permissions."
                )
            raise ValueError(
                f"Failed to access repository '{repo_name}': {e.data.get('message', str(e))}"
            )
        except Exception as e:
            if "timed out" in str(e).lower() or "connect" in str(e).lower():
                raise ConnectionError(
                    f"GitHubSource: failed to connect to GitHub within "
                    f"{_CONNECTION_TIMEOUT}s — {e}"
                )
            raise ValueError(
                f"Failed to access repository '{repo_name}': {e}"
            )

    def _check_rate_limit(self) -> None:
        """Sleep until rate-limit resets if remaining calls are below threshold."""
        rate_limit = self._client.get_rate_limit()
        core = rate_limit.core
        if core.remaining < _RATE_LIMIT_THRESHOLD:
            reset_time = core.reset.timestamp()
            sleep_duration = max(0, reset_time - time.time()) + 1
            time.sleep(sleep_duration)

    def get_issues(self, search_term: str, limit: int = 20) -> list[IssueRecord]:
        """Retrieve GitHub issues matching the search term.

        Searches issue titles and bodies for the search term. Includes
        issue comments in the results.

        Args:
            search_term: Keyword to search in issue titles and bodies.
            limit: Maximum number of issues to return (default 20).

        Returns:
            List of IssueRecord objects, never exceeding `limit` entries.
            Returns an empty list if no issues match.
        """
        self._check_rate_limit()

        try:
            query = f"{search_term} repo:{self._repo_name}"
            results = self._client.search_issues(query=query)

            issues: list[IssueRecord] = []
            for issue in results:
                if len(issues) >= limit:
                    break

                # Skip pull requests (GitHub search_issues returns PRs too)
                if issue.pull_request is not None:
                    continue

                self._check_rate_limit()

                # Fetch comments for this issue
                comments: list[IssueComment] = []
                for comment in issue.get_comments():
                    comments.append(
                        IssueComment(
                            author=comment.user.login if comment.user else "unknown",
                            body=comment.body or "",
                            timestamp=comment.created_at.isoformat() if comment.created_at else "",
                        )
                    )

                issues.append(
                    IssueRecord(
                        number=issue.number,
                        title=issue.title or "",
                        body=issue.body or "",
                        author=issue.user.login if issue.user else "unknown",
                        state=issue.state or "open",
                        comments=comments,
                        labels=[label.name for label in issue.labels],
                    )
                )

            return issues

        except Exception as e:
            if "timed out" in str(e).lower() or "connect" in str(e).lower():
                raise ConnectionError(
                    f"GitHubSource: failed to connect to GitHub within "
                    f"{_CONNECTION_TIMEOUT}s — {e}"
                )
            # For other errors, return empty list (graceful handling)
            return []

    def get_pr_comments(
        self, code_path: str, limit: int = 20
    ) -> list[PRCommentRecord]:
        """Retrieve PR review comments from diffs that include the code path.

        Searches pull request review comments for those referencing the
        specified file path.

        Args:
            code_path: File path to search in PR diffs.
            limit: Maximum number of comments to return (default 20).

        Returns:
            List of PRCommentRecord objects, never exceeding `limit` entries.
            Returns an empty list if no comments match.
        """
        self._check_rate_limit()

        try:
            # Get pull request review comments for the repository
            # and filter by file path
            comments: list[PRCommentRecord] = []

            # Fetch recent pulls and check their review comments
            pulls = self._repo.get_pulls(state="all", sort="updated", direction="desc")

            for pr in pulls:
                if len(comments) >= limit:
                    break

                self._check_rate_limit()

                review_comments = pr.get_review_comments()
                for review_comment in review_comments:
                    if len(comments) >= limit:
                        break

                    # Filter by file path
                    comment_path = review_comment.path or ""
                    if code_path in comment_path or comment_path.endswith(code_path):
                        comments.append(
                            PRCommentRecord(
                                pr_number=pr.number,
                                file_path=comment_path,
                                line_number=review_comment.line,
                                body=review_comment.body or "",
                                author=(
                                    review_comment.user.login
                                    if review_comment.user
                                    else "unknown"
                                ),
                                timestamp=(
                                    review_comment.created_at.isoformat()
                                    if review_comment.created_at
                                    else ""
                                ),
                            )
                        )

            return comments

        except Exception as e:
            if "timed out" in str(e).lower() or "connect" in str(e).lower():
                raise ConnectionError(
                    f"GitHubSource: failed to connect to GitHub within "
                    f"{_CONNECTION_TIMEOUT}s — {e}"
                )
            # For other errors, return empty list (graceful handling)
            return []

    def get_commits(self, code_path: str, limit: int = 50) -> list[CommitRecord]:
        """Not implemented — use GitSource for commit history.

        Raises:
            NotImplementedError: Always. Git commit retrieval is handled by GitSource.
        """
        raise NotImplementedError(
            "GitHubSource does not provide commit history. Use GitSource instead."
        )

    def get_related_files(
        self, code_path: str, min_co_commits: int = 3, limit: int = 20
    ) -> list[RelatedFile]:
        """Not implemented — use GitSource for related file analysis.

        Raises:
            NotImplementedError: Always. Related file analysis is handled by GitSource.
        """
        raise NotImplementedError(
            "GitHubSource does not provide related file analysis. Use GitSource instead."
        )
