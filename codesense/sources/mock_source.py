"""Mock data source backed by fixture JSON files.

Implements the DataSource ABC for zero-credential demo mode.
Loads realistic sample data from tests/fixtures/ and returns
proper dataclass instances without any network access.
"""

import json
from pathlib import Path
from typing import Optional

from codesense.interfaces import DataSource
from codesense.models.mcp import (
    CommitRecord,
    IssueComment,
    IssueRecord,
    PRCommentRecord,
    RelatedFile,
)

# Default fixture directory relative to project root
_DEFAULT_FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures"


class MockSource(DataSource):
    """Fixture-backed data source for demo mode.

    Loads JSON fixture files and returns data as proper dataclass instances.
    Requires no credentials or network access. All methods respect limit
    parameters by truncating results.

    Args:
        fixtures_dir: Path to the directory containing fixture JSON files.
            Defaults to tests/fixtures/ relative to project root.
    """

    def __init__(self, fixtures_dir: Optional[Path] = None) -> None:
        self._fixtures_dir = fixtures_dir or _DEFAULT_FIXTURES_DIR
        self._commits: list[CommitRecord] = self._load_commits()
        self._issues: list[IssueRecord] = self._load_issues()
        self._pr_comments: list[PRCommentRecord] = self._load_pr_comments()
        self._related_files: list[RelatedFile] = self._load_related_files()

    def _load_commits(self) -> list[CommitRecord]:
        """Load commit records from mock_commits.json."""
        data = self._read_json("mock_commits.json")
        return [
            CommitRecord(
                sha=item["sha"],
                message=item["message"],
                author=item["author"],
                timestamp=item["timestamp"],
                diff=item["diff"],
                files_changed=item["files_changed"],
            )
            for item in data
        ]

    def _load_issues(self) -> list[IssueRecord]:
        """Load issue records from mock_issues.json."""
        data = self._read_json("mock_issues.json")
        return [
            IssueRecord(
                number=item["number"],
                title=item["title"],
                body=item["body"],
                author=item["author"],
                state=item["state"],
                comments=[
                    IssueComment(
                        author=c["author"],
                        body=c["body"],
                        timestamp=c["timestamp"],
                    )
                    for c in item["comments"]
                ],
                labels=item["labels"],
            )
            for item in data
        ]

    def _load_pr_comments(self) -> list[PRCommentRecord]:
        """Load PR comment records from mock_pr_comments.json."""
        data = self._read_json("mock_pr_comments.json")
        return [
            PRCommentRecord(
                pr_number=item["pr_number"],
                file_path=item["file_path"],
                line_number=item.get("line_number"),
                body=item["body"],
                author=item["author"],
                timestamp=item["timestamp"],
            )
            for item in data
        ]

    def _load_related_files(self) -> list[RelatedFile]:
        """Load related file records from mock_related_files.json."""
        data = self._read_json("mock_related_files.json")
        return [
            RelatedFile(
                path=item["path"],
                co_commit_count=item["co_commit_count"],
                last_co_modified=item["last_co_modified"],
            )
            for item in data
        ]

    def _read_json(self, filename: str) -> list[dict]:
        """Read and parse a JSON fixture file."""
        filepath = self._fixtures_dir / filename
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)

    def get_commits(self, code_path: str, limit: int = 50) -> list[CommitRecord]:
        """Retrieve mock commits, filtered by code_path presence in files_changed.

        Args:
            code_path: File or directory path to filter by.
            limit: Maximum number of commits to return.

        Returns:
            List of CommitRecord objects, truncated to limit.
        """
        matching = [
            commit
            for commit in self._commits
            if any(code_path in f for f in commit.files_changed)
        ]
        # If no matches on path, return all commits (demo-friendly behavior)
        results = matching if matching else self._commits
        return results[:limit]

    def get_issues(self, search_term: str, limit: int = 20) -> list[IssueRecord]:
        """Retrieve mock issues, filtered by search_term in title or body.

        Args:
            search_term: Keyword to search in issue titles and bodies.
            limit: Maximum number of issues to return.

        Returns:
            List of IssueRecord objects, truncated to limit.
        """
        term_lower = search_term.lower()
        matching = [
            issue
            for issue in self._issues
            if term_lower in issue.title.lower() or term_lower in issue.body.lower()
        ]
        # If no matches, return all issues (demo-friendly behavior)
        results = matching if matching else self._issues
        return results[:limit]

    def get_pr_comments(
        self, code_path: str, limit: int = 20
    ) -> list[PRCommentRecord]:
        """Retrieve mock PR comments, filtered by code_path in file_path.

        Args:
            code_path: File path to filter PR comments by.
            limit: Maximum number of comments to return.

        Returns:
            List of PRCommentRecord objects, truncated to limit.
        """
        matching = [
            comment
            for comment in self._pr_comments
            if code_path in comment.file_path
        ]
        # If no matches, return all PR comments (demo-friendly behavior)
        results = matching if matching else self._pr_comments
        return results[:limit]

    def get_related_files(
        self, code_path: str, min_co_commits: int = 3, limit: int = 20
    ) -> list[RelatedFile]:
        """Retrieve mock related files with co_commit_count >= min_co_commits.

        Args:
            code_path: File path to find co-modification relationships for.
            min_co_commits: Minimum co-commit count threshold.
            limit: Maximum number of related files to return.

        Returns:
            List of RelatedFile objects where co_commit_count >= min_co_commits,
            truncated to limit.
        """
        filtered = [
            rf
            for rf in self._related_files
            if rf.co_commit_count >= min_co_commits
        ]
        return filtered[:limit]
