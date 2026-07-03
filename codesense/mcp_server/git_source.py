"""Git data source using gitpython.

Implements the DataSource ABC for live git repository access.
Provides commit history and co-modification analysis for specified code paths.
"""

import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError, NoSuchPathError

from codesense.interfaces import DataSource
from codesense.models.mcp import (
    CommitRecord,
    IssueRecord,
    PRCommentRecord,
    RelatedFile,
)

_GIT_TIMEOUT_SECONDS = 30


class GitOperationTimeout(Exception):
    """Raised when a git operation exceeds the configured timeout."""

    pass


class GitSource(DataSource):
    """Live git repository data source using gitpython.

    Retrieves commit history and co-modification relationships from
    a local git repository. Applies a 30-second timeout to git operations.

    Args:
        repo_path: Path to the git repository root directory.

    Raises:
        ValueError: If repo_path is not a valid git repository.
    """

    def __init__(self, repo_path: str) -> None:
        try:
            self._repo = Repo(repo_path)
        except (InvalidGitRepositoryError, NoSuchPathError) as e:
            raise ValueError(
                f"Invalid git repository path: {repo_path}. {e}"
            ) from e
        self._repo_path = Path(repo_path).resolve()
        self._timeout = _GIT_TIMEOUT_SECONDS

    def get_commits(self, code_path: str, limit: int = 50) -> list[CommitRecord]:
        """Retrieve git commits that modified the specified code path.

        Args:
            code_path: File or directory path relative to the repository root.
            limit: Maximum number of commits to return (default 50).

        Returns:
            List of CommitRecord objects, never exceeding `limit` entries.
            Returns empty list if the repository has no history.

        Raises:
            ValueError: If code_path does not exist in the repository.
        """
        self._validate_code_path(code_path)

        try:
            commits = self._iter_commits_for_path(code_path, limit)
        except GitOperationTimeout:
            raise RuntimeError(
                f"GitSource: git operation timed out after {self._timeout}s "
                f"while retrieving commits for '{code_path}'"
            )
        except GitCommandError as e:
            raise RuntimeError(
                f"GitSource: failed to retrieve commits for '{code_path}'. {e}"
            )

        records: list[CommitRecord] = []
        for commit in commits:
            diff_summary = self._get_diff_summary(commit, code_path)
            files_changed = self._get_files_changed(commit)
            records.append(
                CommitRecord(
                    sha=commit.hexsha,
                    message=commit.message.strip(),
                    author=str(commit.author),
                    timestamp=datetime.fromtimestamp(
                        commit.committed_date, tz=timezone.utc
                    ).isoformat(),
                    diff=diff_summary,
                    files_changed=files_changed,
                )
            )

        return records[:limit]

    def get_issues(self, search_term: str, limit: int = 20) -> list[IssueRecord]:
        """Not implemented for git source — use GitHubSource instead.

        Raises:
            NotImplementedError: Always, as git repos don't provide issue data.
        """
        raise NotImplementedError(
            "GitSource does not provide issue data. Use GitHubSource for GitHub issues."
        )

    def get_pr_comments(
        self, code_path: str, limit: int = 20
    ) -> list[PRCommentRecord]:
        """Not implemented for git source — use GitHubSource instead.

        Raises:
            NotImplementedError: Always, as git repos don't provide PR comment data.
        """
        raise NotImplementedError(
            "GitSource does not provide PR comment data. Use GitHubSource for PR comments."
        )

    def get_related_files(
        self, code_path: str, min_co_commits: int = 3, limit: int = 20
    ) -> list[RelatedFile]:
        """Find files frequently co-modified with the specified code path.

        Examines the commit history for files that appear in the same commits
        as code_path, counts co-modifications, and returns those meeting the
        minimum threshold sorted by frequency.

        Args:
            code_path: File path relative to the repository root.
            min_co_commits: Minimum co-commit count threshold (default 3).
            limit: Maximum number of related files to return (default 20).

        Returns:
            List of RelatedFile objects where each has
            co_commit_count >= min_co_commits, sorted descending by count,
            never exceeding `limit` entries.

        Raises:
            ValueError: If code_path does not exist in the repository.
        """
        self._validate_code_path(code_path)

        try:
            commits = self._iter_commits_for_path(code_path, max_count=None)
        except GitOperationTimeout:
            raise RuntimeError(
                f"GitSource: git operation timed out after {self._timeout}s "
                f"while analyzing related files for '{code_path}'"
            )
        except GitCommandError as e:
            raise RuntimeError(
                f"GitSource: failed to analyze related files for '{code_path}'. {e}"
            )

        # Count co-modifications
        co_mod_counter: Counter[str] = Counter()
        last_co_modified: dict[str, str] = {}

        normalized_path = code_path.replace("\\", "/")

        for commit in commits:
            files_in_commit = self._get_files_changed(commit)
            # Check if target file is in this commit
            if not any(
                f.replace("\\", "/") == normalized_path or f.replace("\\", "/").endswith("/" + normalized_path)
                for f in files_in_commit
            ):
                continue

            commit_ts = datetime.fromtimestamp(
                commit.committed_date, tz=timezone.utc
            ).isoformat()

            for file_path in files_in_commit:
                norm_file = file_path.replace("\\", "/")
                # Skip the target file itself
                if norm_file == normalized_path or norm_file.endswith("/" + normalized_path):
                    continue
                co_mod_counter[norm_file] += 1
                # Track most recent co-modification timestamp
                if norm_file not in last_co_modified or commit_ts > last_co_modified[norm_file]:
                    last_co_modified[norm_file] = commit_ts

        # Filter by minimum co-commit count and build RelatedFile objects
        related: list[RelatedFile] = []
        for file_path, count in co_mod_counter.most_common():
            if count < min_co_commits:
                break
            related.append(
                RelatedFile(
                    path=file_path,
                    co_commit_count=count,
                    last_co_modified=last_co_modified[file_path],
                )
            )

        # Sort by co_commit_count descending (most_common already does this)
        return related[:limit]

    def _validate_code_path(self, code_path: str) -> None:
        """Check that code_path exists in the repository.

        Checks both the working tree and git's tracked files.

        Raises:
            ValueError: If the code path doesn't exist in the repo.
        """
        # Normalize the path
        normalized = code_path.replace("\\", "/")

        # Check if path exists in the working tree
        full_path = self._repo_path / normalized
        if full_path.exists():
            return

        # Check if path exists in git's tracked files (might have been deleted)
        try:
            # git log returns empty string if file was never tracked
            result = self._repo.git.log("--oneline", "-1", "--", normalized)
            if result.strip():
                return
        except GitCommandError:
            pass

        raise ValueError(
            f"Code path '{code_path}' does not exist in the repository at "
            f"'{self._repo_path}'. Please verify the path is correct."
        )

    def _iter_commits_for_path(self, code_path: str, max_count: int | None = 50) -> list:
        """Get commits that modified a specific path with timeout.

        Args:
            code_path: Relative file path within the repository.
            max_count: Maximum commits to retrieve, or None for all.

        Returns:
            List of git commit objects. Returns empty list if repo has no history.

        Raises:
            GitOperationTimeout: If operation exceeds timeout.
        """
        # Check if the repo has any commits (HEAD exists)
        try:
            self._repo.head.commit
        except (ValueError, TypeError):
            # No commits in the repo yet
            return []

        result: list = []
        error: list = []

        def _run():
            try:
                kwargs: dict = {"paths": code_path}
                if max_count is not None:
                    kwargs["max_count"] = max_count
                result.extend(list(self._repo.iter_commits(**kwargs)))
            except Exception as e:
                error.append(e)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=self._timeout)

        if thread.is_alive():
            raise GitOperationTimeout(
                f"Git operation timed out after {self._timeout} seconds"
            )

        if error:
            raise error[0]

        return result

    def _get_diff_summary(self, commit, code_path: str) -> str:
        """Get the diff summary for a specific file in a commit.

        Args:
            commit: A git commit object.
            code_path: The file path to extract diff for.

        Returns:
            String containing the diff content, or empty string if unavailable.
        """
        try:
            if commit.parents:
                parent = commit.parents[0]
                diffs = parent.diff(commit, paths=code_path, create_patch=True)
            else:
                # Initial commit — show the full file content as an addition
                try:
                    blob = commit.tree / code_path
                    content = blob.data_stream.read().decode("utf-8", errors="replace")
                    return f"@@ -0,0 +1 @@\n+{content}"
                except (KeyError, TypeError):
                    return ""

            parts = []
            for diff_item in diffs:
                if diff_item.diff:
                    decoded = diff_item.diff.decode("utf-8", errors="replace")
                    parts.append(decoded)

            return "\n".join(parts) if parts else ""
        except Exception:
            return ""

    def _get_files_changed(self, commit) -> list[str]:
        """Get list of files changed in a commit.

        Args:
            commit: A git commit object.

        Returns:
            List of file path strings changed in this commit.
        """
        try:
            if commit.parents:
                parent = commit.parents[0]
                diffs = parent.diff(commit)
            else:
                # Initial commit — list all files in the commit tree
                files: list[str] = []
                for blob in commit.tree.traverse():
                    if blob.type == "blob":
                        files.append(blob.path)
                return files

            files = []
            for diff_item in diffs:
                if diff_item.b_path:
                    files.append(diff_item.b_path)
                elif diff_item.a_path:
                    files.append(diff_item.a_path)
            return files
        except Exception:
            return []
