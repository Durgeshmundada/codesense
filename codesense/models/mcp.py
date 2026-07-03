"""MCP data retrieval models."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CommitRecord:
    """A git commit record with diff and metadata."""

    sha: str
    message: str
    author: str
    timestamp: str
    diff: str
    files_changed: list[str]


@dataclass
class IssueComment:
    """A comment on a GitHub issue."""

    author: str
    body: str
    timestamp: str


@dataclass
class IssueRecord:
    """A GitHub issue with comments and labels."""

    number: int
    title: str
    body: str
    author: str
    state: str
    comments: list[IssueComment]
    labels: list[str]


@dataclass
class PRCommentRecord:
    """A pull request review comment."""

    pr_number: int
    file_path: str
    line_number: Optional[int]
    body: str
    author: str
    timestamp: str


@dataclass
class RelatedFile:
    """A file that was co-modified with a target file."""

    path: str
    co_commit_count: int
    last_co_modified: str
