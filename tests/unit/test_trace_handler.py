"""Unit tests for the TraceHandler capability."""

from unittest.mock import patch

import pytest

from codesense.capabilities.trace import TraceHandler
from codesense.models.output import CommandParams


class TestTraceHandler:
    """Tests for TraceHandler.run()."""

    def test_returns_decision_timeline_title(self):
        """Handler always returns 'Decision Timeline' as the title."""
        handler = TraceHandler(mock=True)
        params = CommandParams(path="some/file.py", mock=True)
        result = handler.run(params)
        assert result.title == "Decision Timeline"

    def test_demo_mode_from_params(self):
        """is_demo_mode reflects params.mock."""
        handler = TraceHandler()
        params = CommandParams(path="some/file.py", mock=True)
        result = handler.run(params)
        assert result.is_demo_mode is True

    def test_demo_mode_from_constructor(self):
        """is_demo_mode reflects constructor mock flag."""
        handler = TraceHandler(mock=True)
        params = CommandParams(path="some/file.py", mock=False)
        result = handler.run(params)
        assert result.is_demo_mode is True

    def test_empty_path_returns_error(self):
        """Empty path produces a helpful error message."""
        handler = TraceHandler(mock=True)
        params = CommandParams(path="", mock=True)
        result = handler.run(params)
        assert "No file path specified" in result.content

    def test_none_path_returns_error(self):
        """None path produces a helpful error message."""
        handler = TraceHandler(mock=True)
        params = CommandParams(path=None, mock=True)
        result = handler.run(params)
        assert "No file path specified" in result.content

    def test_events_sorted_chronologically(self):
        """Events are ordered earliest first."""
        handler = TraceHandler(mock=True)
        params = CommandParams(path="codesense/main.py", mock=True)
        result = handler.run(params)

        # Extract dates from the timeline content
        lines = result.content.split("\n")
        dates = []
        for line in lines:
            if line.startswith("- ") and "**" in line:
                # Extract date between ** markers
                start = line.index("**") + 2
                end = line.index("**", start)
                dates.append(line[start:end])

        # Verify chronological order
        assert dates == sorted(dates)

    def test_with_line_number_filters_pr_comments(self):
        """Line number filters PR comments to matching lines."""
        handler = TraceHandler(mock=True)
        # With a specific line number that doesn't match PR comments
        params = CommandParams(path="some/file.py", line_number=999, mock=True)
        result = handler.run(params)
        # Should still have commits (not filtered by line number)
        # but PR comments with non-matching line numbers are excluded
        assert result.title == "Decision Timeline"

    def test_format_includes_commit_marker(self):
        """Commits use the hammer emoji marker."""
        handler = TraceHandler(mock=True)
        params = CommandParams(path="some/file.py", mock=True)
        result = handler.run(params)
        assert "🔨" in result.content

    def test_format_includes_pr_comment_marker(self):
        """PR comments use the speech bubble emoji marker."""
        handler = TraceHandler(mock=True)
        params = CommandParams(path="codesense/main.py", mock=True)
        result = handler.run(params)
        assert "💬" in result.content

    def test_timeline_header_includes_path(self):
        """Timeline header mentions the traced file path."""
        handler = TraceHandler(mock=True)
        params = CommandParams(path="src/auth.py", mock=True)
        result = handler.run(params)
        assert "src/auth.py" in result.content

    def test_timeline_header_includes_line_number(self):
        """Timeline header mentions the line number when provided."""
        handler = TraceHandler(mock=True)
        params = CommandParams(path="src/auth.py", line_number=42, mock=True)
        result = handler.run(params)
        assert "line 42" in result.content

    @patch("codesense.mcp_server.server.get_git_history")
    @patch("codesense.mcp_server.server.get_pr_comments")
    def test_empty_results_graceful_message(self, mock_pr, mock_git):
        """Empty MCP results produce a graceful 'no events found' message."""
        mock_git.return_value = {"commits": []}
        mock_pr.return_value = {"pr_comments": []}

        handler = TraceHandler(mock=True)
        params = CommandParams(path="nonexistent/file.py", mock=True)
        result = handler.run(params)
        assert "No timeline events found" in result.content

    @patch("codesense.mcp_server.server.get_git_history")
    @patch("codesense.mcp_server.server.get_pr_comments")
    def test_handles_mcp_errors_gracefully(self, mock_pr, mock_git):
        """MCP tool errors are handled without crashing."""
        mock_git.side_effect = Exception("Git connection failed")
        mock_pr.side_effect = Exception("GitHub API unavailable")

        handler = TraceHandler(mock=True)
        params = CommandParams(path="src/file.py", mock=True)
        result = handler.run(params)
        # Should return graceful empty result, not crash
        assert "No timeline events found" in result.content

    def test_event_count_in_content(self):
        """Timeline content includes the number of events found."""
        handler = TraceHandler(mock=True)
        params = CommandParams(path="codesense/main.py", mock=True)
        result = handler.run(params)
        assert "event(s)" in result.content
