"""Trace capability handler — displays a decision timeline for code.

Builds a chronological timeline of commits, issues, and PR comments
that led to the specified code at a given file + line number, using
MCP tools (get_git_history, get_pr_comments) for data retrieval.

Requirements: 5.6
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from codesense.models.output import CommandOutput, CommandParams

logger = logging.getLogger(__name__)

TRACE_TITLE = "Decision Timeline"


class TraceHandler:
    """Capability handler for the 'trace' command.

    Retrieves git history and PR comments via MCP tools to build a
    chronological timeline of events that led to a specific line of code.

    Implements the CapabilityHandler protocol: run(params) -> CommandOutput.

    Args:
        mock: Optional flag to force demo mode for data sources.
            When True, MCP tools use MockSource regardless of environment.
    """

    def __init__(self, mock: Optional[bool] = None) -> None:
        self._mock = mock

    def run(self, params: CommandParams) -> CommandOutput:
        """Execute the trace capability with the given parameters.

        Uses MCP tools (get_git_history, get_pr_comments) to gather commits
        and PR comments related to the file at params.path (optionally
        filtered by params.line_number), then builds a chronological timeline.

        Args:
            params: Parsed CLI arguments. Uses:
                - params.path: File path to trace history for.
                - params.line_number: Optional line number to focus on.
                - params.mock: Whether demo mode is active.

        Returns:
            CommandOutput with:
                - title: "Decision Timeline"
                - content: formatted chronological timeline
                - is_demo_mode: from params.mock
        """
        code_path = params.path or ""
        line_number = params.line_number
        is_demo = params.mock or (self._mock is True)

        if not code_path:
            return CommandOutput(
                title=TRACE_TITLE,
                content="Error: No file path specified. Please provide a code path to trace.",
                is_demo_mode=is_demo,
            )

        # Gather data from MCP tools
        events = self._gather_timeline_events(code_path, line_number, is_demo)

        # Handle empty results gracefully
        if not events:
            location = f"`{code_path}`"
            if line_number is not None:
                location += f" (line {line_number})"
            return CommandOutput(
                title=TRACE_TITLE,
                content=(
                    f"No timeline events found for {location}.\n\n"
                    "This could mean the file has no recorded git history, "
                    "or no related issues/PR comments were found."
                ),
                is_demo_mode=is_demo,
            )

        # Sort events chronologically (earliest first)
        events.sort(key=lambda e: e["sort_key"])

        # Format as a timeline
        content = self._format_timeline(events, code_path, line_number)

        return CommandOutput(
            title=TRACE_TITLE,
            content=content,
            is_demo_mode=is_demo,
        )

    def _gather_timeline_events(
        self,
        code_path: str,
        line_number: Optional[int],
        mock: bool,
    ) -> list[dict]:
        """Gather timeline events from MCP tools.

        Calls get_git_history and get_pr_comments to collect commits and
        PR review comments related to the specified code path.

        Args:
            code_path: File path to query.
            line_number: Optional line number for filtering PR comments.
            mock: Whether to use mock data sources.

        Returns:
            List of event dicts with keys: sort_key, date, type, description.
        """
        from codesense.mcp_server.server import get_git_history, get_pr_comments

        events: list[dict] = []

        # Gather commits from git history
        try:
            git_result = get_git_history(code_path=code_path, limit=50, mock=mock)
            if "commits" in git_result:
                for commit in git_result["commits"]:
                    timestamp = commit.get("timestamp", "")
                    date_str = self._format_date(timestamp)
                    sort_key = self._parse_sort_key(timestamp)

                    author = commit.get("author", "unknown")
                    message = commit.get("message", "").split("\n")[0]  # first line
                    sha_short = commit.get("sha", "")[:7]

                    events.append({
                        "sort_key": sort_key,
                        "date": date_str,
                        "type": "commit",
                        "description": f"[{sha_short}] {message} — by {author}",
                    })
        except Exception as e:
            logger.warning("Failed to retrieve git history for '%s': %s", code_path, e)

        # Gather PR comments
        try:
            pr_result = get_pr_comments(code_path=code_path, limit=20, mock=mock)
            if "pr_comments" in pr_result:
                for comment in pr_result["pr_comments"]:
                    # Filter by line number if specified
                    if line_number is not None:
                        comment_line = comment.get("line_number")
                        if comment_line is not None and comment_line != line_number:
                            continue

                    timestamp = comment.get("timestamp", "")
                    date_str = self._format_date(timestamp)
                    sort_key = self._parse_sort_key(timestamp)

                    author = comment.get("author", "unknown")
                    body = comment.get("body", "").split("\n")[0]  # first line
                    pr_num = comment.get("pr_number", "?")

                    events.append({
                        "sort_key": sort_key,
                        "date": date_str,
                        "type": "pr_comment",
                        "description": f"PR #{pr_num} review: {body} — by {author}",
                    })
        except Exception as e:
            logger.warning("Failed to retrieve PR comments for '%s': %s", code_path, e)

        return events

    def _format_timeline(
        self,
        events: list[dict],
        code_path: str,
        line_number: Optional[int],
    ) -> str:
        """Format events as a markdown timeline list.

        Args:
            events: Sorted list of event dicts.
            code_path: Original code path being traced.
            line_number: Optional line number being traced.

        Returns:
            Formatted markdown string with timeline entries.
        """
        location = f"`{code_path}`"
        if line_number is not None:
            location += f" (line {line_number})"

        lines: list[str] = []
        lines.append(f"### Timeline for {location}")
        lines.append("")
        lines.append(f"Found {len(events)} event(s):")
        lines.append("")

        for event in events:
            event_type = event["type"]
            date = event["date"]
            description = event["description"]

            # Use different markers for different event types
            if event_type == "commit":
                marker = "🔨"
            elif event_type == "pr_comment":
                marker = "💬"
            elif event_type == "issue":
                marker = "📋"
            else:
                marker = "•"

            lines.append(f"- {marker} **{date}** — {description}")

        return "\n".join(lines)

    def _format_date(self, timestamp: str) -> str:
        """Format an ISO timestamp string into a readable date.

        Args:
            timestamp: ISO 8601 timestamp string.

        Returns:
            Human-readable date string (e.g., "2024-01-15").
            Returns the raw string if parsing fails.
        """
        if not timestamp:
            return "unknown date"

        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            # Return raw timestamp if parsing fails
            return timestamp[:10] if len(timestamp) >= 10 else timestamp

    def _parse_sort_key(self, timestamp: str) -> str:
        """Parse a timestamp into a sortable string key.

        Args:
            timestamp: ISO 8601 timestamp string.

        Returns:
            Sortable string representation. Empty timestamps sort last.
        """
        if not timestamp:
            return "9999-99-99"

        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            return dt.isoformat()
        except (ValueError, TypeError):
            # Return the raw string for best-effort sorting
            return timestamp
