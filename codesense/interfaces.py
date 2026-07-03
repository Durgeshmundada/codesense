"""Abstract interfaces and protocols for CodeSense.

This module defines the core abstractions that decouple the reasoning pipeline
from concrete implementations:

- DataSource: ABC for MCP data retrieval (git, GitHub, mock sources)
- Node: Protocol for reasoning graph nodes
- CapabilityHandler: Protocol for CLI command handlers

Design constraint (Requirement 7.2):
    The Conflict dataclass (defined in codesense.models.state) intentionally has
    NO winner, resolution, or ranking fields. Conflicts surface ambiguity honestly
    so developers can decide which source to trust. Any implementation that detects
    contradictions MUST produce Conflict objects without attempting to resolve them.
"""

from abc import ABC, abstractmethod
from typing import Protocol

from codesense.models.mcp import (
    CommitRecord,
    IssueRecord,
    PRCommentRecord,
    RelatedFile,
)
from codesense.models.output import CommandOutput, CommandParams
from codesense.models.state import AgentState


class DataSource(ABC):
    """Abstract base class for MCP data retrieval sources.

    Implementations include GitSource (live git history), GitHubSource
    (live GitHub API), and MockSource (fixture-based demo data).

    All implementations must respect the configured result limits and
    return descriptive error messages on failure (Requirement 2.5, 2.7, 2.8).
    """

    @abstractmethod
    def get_commits(self, code_path: str, limit: int = 50) -> list[CommitRecord]:
        """Retrieve git commits that modified the specified code path.

        Args:
            code_path: File or directory path within the repository.
            limit: Maximum number of commits to return (default 50).

        Returns:
            List of CommitRecord objects, never exceeding `limit` entries.
        """
        ...

    @abstractmethod
    def get_issues(self, search_term: str, limit: int = 20) -> list[IssueRecord]:
        """Retrieve GitHub issues matching the search term.

        Args:
            search_term: Code path or keyword to search in issue titles/bodies.
            limit: Maximum number of issues to return (default 20).

        Returns:
            List of IssueRecord objects, never exceeding `limit` entries.
        """
        ...

    @abstractmethod
    def get_pr_comments(
        self, code_path: str, limit: int = 20
    ) -> list[PRCommentRecord]:
        """Retrieve PR review comments from diffs that include the code path.

        Args:
            code_path: File path to search in PR diffs.
            limit: Maximum number of comments to return (default 20).

        Returns:
            List of PRCommentRecord objects, never exceeding `limit` entries.
        """
        ...

    @abstractmethod
    def get_related_files(
        self, code_path: str, min_co_commits: int = 3, limit: int = 20
    ) -> list[RelatedFile]:
        """Find files frequently co-modified with the specified code path.

        Args:
            code_path: File path to find co-modification relationships for.
            min_co_commits: Minimum co-commit count threshold (default 3).
            limit: Maximum number of related files to return (default 20).

        Returns:
            List of RelatedFile objects where each has
            co_commit_count >= min_co_commits, never exceeding `limit` entries.
        """
        ...


class Node(Protocol):
    """Protocol for reasoning graph nodes.

    Each node in the LangGraph StateGraph implements this protocol,
    receiving the current agent state and returning an updated state.
    """

    def execute(self, state: AgentState) -> AgentState:
        """Execute this node's logic on the given state.

        Args:
            state: The current reasoning loop state.

        Returns:
            Updated AgentState after this node's processing.
        """
        ...


class CapabilityHandler(Protocol):
    """Protocol for CLI capability handlers.

    Each CLI command (explain, describe, tree, flow, etc.) delegates
    to a handler implementing this protocol.
    """

    def run(self, params: CommandParams) -> CommandOutput:
        """Execute the capability with the given parameters.

        Args:
            params: Parsed CLI arguments including path, flags, and options.

        Returns:
            Structured output consumed by the Rich formatter.
        """
        ...
