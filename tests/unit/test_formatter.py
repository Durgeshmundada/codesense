"""Unit tests for RichFormatter output formatting.

Validates Requirements 6.1-6.6:
- 6.1: Syntax highlighting using detected language (file extension or code fence)
- 6.2: Plain monospaced text fallback when language cannot be detected
- 6.3: Structured data formatted using Rich tables or panels
- 6.4: Markdown rendering (headings, bold, italic, code blocks, lists)
- 6.5: Pagination when output exceeds terminal height
- 6.6: Plain text fallback when TERM=dumb or NO_COLOR is set
"""

import os
from io import StringIO
from unittest.mock import patch

import pytest
from rich.console import Console

from codesense.models.output import CodeSnippet, CommandOutput, TableData
from codesense.output.formatter import (
    RichFormatter,
    _detect_language,
    _get_confidence_color,
    _is_plain_text_mode,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_formatter(width: int = 120, no_color: bool = True) -> RichFormatter:
    """Create a RichFormatter with a buffer console for testing."""
    console = Console(file=StringIO(), width=width, no_color=no_color, highlight=False)
    return RichFormatter(console=console)


def _capture_output(formatter: RichFormatter, output: CommandOutput) -> str:
    """Capture format_output result as a string."""
    buffer = StringIO()
    console = Console(file=buffer, width=120, no_color=True, highlight=False)
    formatter._console = console
    formatter.format_output(output)
    return buffer.getvalue()


def _capture_code_render(formatter: RichFormatter, snippet: CodeSnippet) -> str:
    """Capture render_code result as a string."""
    buffer = StringIO()
    console = Console(file=buffer, width=120, no_color=True, highlight=False)
    formatter._console = console
    formatter.render_code(snippet)
    return buffer.getvalue()


def _capture_table_render(formatter: RichFormatter, table: TableData) -> str:
    """Capture render_table result as a string."""
    buffer = StringIO()
    console = Console(file=buffer, width=120, no_color=True, highlight=False)
    formatter._console = console
    formatter.render_table(table)
    return buffer.getvalue()


# ─── Tests: Syntax Highlighting (Req 6.1) ────────────────────────────────────


class TestSyntaxHighlighting:
    """Test syntax highlighting with various languages."""

    def test_python_language_detected(self):
        """Python code snippet gets language detected correctly."""
        snippet = CodeSnippet(code="def hello(): pass", language="python")
        lang = _detect_language(snippet)
        assert lang == "python"

    def test_javascript_language_detected(self):
        """JavaScript code snippet gets language detected correctly."""
        snippet = CodeSnippet(code="const x = 1;", language="javascript")
        lang = _detect_language(snippet)
        assert lang == "javascript"

    def test_typescript_language_detected(self):
        """TypeScript detected from language field."""
        snippet = CodeSnippet(code="let x: number = 1;", language="typescript")
        lang = _detect_language(snippet)
        assert lang == "typescript"

    def test_language_from_label_extension(self):
        """Language inferred from file extension in label."""
        snippet = CodeSnippet(code="x = 1", language="", label="script.py")
        lang = _detect_language(snippet)
        assert lang == "python"

    def test_language_from_label_js_extension(self):
        """JavaScript inferred from .js file extension in label."""
        snippet = CodeSnippet(code="var x;", language="", label="app.js")
        lang = _detect_language(snippet)
        assert lang == "javascript"

    def test_language_from_bare_extension_label(self):
        """Language detected when label is just an extension (e.g., '.py')."""
        snippet = CodeSnippet(code="pass", language="", label=".py")
        lang = _detect_language(snippet)
        assert lang == "python"

    def test_language_name_as_label(self):
        """Language detected when label is a language name directly."""
        snippet = CodeSnippet(code="fn main() {}", language="", label="rust")
        lang = _detect_language(snippet)
        assert lang == "rust"

    def test_unknown_language_returns_none(self):
        """Unknown language string returns None."""
        snippet = CodeSnippet(code="some code", language="unknown")
        lang = _detect_language(snippet)
        assert lang is None

    def test_empty_language_no_label_returns_none(self):
        """No language info at all returns None."""
        snippet = CodeSnippet(code="some code", language="")
        lang = _detect_language(snippet)
        assert lang is None

    def test_render_python_code_has_line_numbers(self):
        """Rendered Python code includes line numbers (Rich Syntax feature)."""
        formatter = _make_formatter()
        snippet = CodeSnippet(
            code="def hello():\n    return 'world'",
            language="python",
            label="example.py",
        )
        rendered = _capture_code_render(formatter, snippet)
        # Line numbers are present (at least "1" should appear)
        assert "1" in rendered
        # Code content is present
        assert "hello" in rendered

    def test_render_javascript_code_contains_content(self):
        """JavaScript code renders with content visible."""
        formatter = _make_formatter()
        snippet = CodeSnippet(
            code="function greet() { return 'hi'; }",
            language="javascript",
        )
        rendered = _capture_code_render(formatter, snippet)
        assert "greet" in rendered


# ─── Tests: Plain Text Fallback (Req 6.2, 6.6) ──────────────────────────────


class TestPlainTextFallback:
    """Test plain text fallback when NO_COLOR or TERM=dumb is set."""

    def test_no_color_env_activates_plain_mode(self):
        """NO_COLOR environment variable triggers plain text mode."""
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            assert _is_plain_text_mode() is True

    def test_term_dumb_activates_plain_mode(self):
        """TERM=dumb triggers plain text mode."""
        with patch.dict(os.environ, {"TERM": "dumb"}, clear=False):
            # Remove NO_COLOR if set
            env = os.environ.copy()
            env.pop("NO_COLOR", None)
            env["TERM"] = "dumb"
            with patch.dict(os.environ, env, clear=True):
                assert _is_plain_text_mode() is True

    def test_normal_terminal_not_plain_mode(self):
        """Normal terminal without NO_COLOR/TERM=dumb is not plain mode."""
        env = {"TERM": "xterm-256color"}
        with patch.dict(os.environ, env, clear=True):
            assert _is_plain_text_mode() is False

    def test_plain_mode_code_renders_without_highlighting(self):
        """In plain mode, code is rendered as plain text with label."""
        formatter = _make_formatter()
        formatter._plain_mode = True
        snippet = CodeSnippet(
            code="def hello(): pass",
            language="python",
            label="example.py",
        )
        rendered = _capture_code_render(formatter, snippet)
        assert "--- example.py ---" in rendered
        assert "def hello(): pass" in rendered

    def test_plain_mode_unknown_language_renders_code(self):
        """Unknown language in plain mode still shows the code."""
        formatter = _make_formatter()
        formatter._plain_mode = True
        snippet = CodeSnippet(
            code="custom syntax here",
            language="unknown",
            label="snippet",
        )
        rendered = _capture_code_render(formatter, snippet)
        assert "custom syntax here" in rendered

    def test_plain_mode_full_output_rendering(self):
        """Full CommandOutput renders in plain text format."""
        formatter = _make_formatter()
        formatter._plain_mode = True
        output = CommandOutput(
            title="Analysis Result",
            content="This is the explanation.",
            confidence=0.85,
        )
        rendered = _capture_output(formatter, output)
        assert "=== Analysis Result ===" in rendered
        assert "This is the explanation." in rendered
        assert "Confidence: 85.0%" in rendered

    def test_undetectable_language_falls_back_to_panel(self):
        """When language is not detected, code renders in a panel as plain text."""
        formatter = _make_formatter()
        formatter._plain_mode = False
        snippet = CodeSnippet(
            code="something weird",
            language="unknown",
            label="mystery",
        )
        rendered = _capture_code_render(formatter, snippet)
        # Should still contain the code content
        assert "something weird" in rendered
        # Should contain the label
        assert "mystery" in rendered


# ─── Tests: Table and Panel Formatting (Req 6.3) ─────────────────────────────


class TestTableAndPanelFormatting:
    """Test structured data formatting with Rich tables and panels."""

    def test_table_headers_rendered(self):
        """Table headers appear in the rendered output."""
        formatter = _make_formatter()
        table = TableData(
            headers=["Name", "Type", "Risk"],
            rows=[["auth.py", "Module", "High"]],
            title="Dependencies",
        )
        rendered = _capture_table_render(formatter, table)
        assert "Name" in rendered
        assert "Type" in rendered
        assert "Risk" in rendered

    def test_table_rows_rendered(self):
        """Table rows appear in the rendered output."""
        formatter = _make_formatter()
        table = TableData(
            headers=["File", "Lines"],
            rows=[
                ["main.py", "150"],
                ["utils.py", "75"],
            ],
            title="File Stats",
        )
        rendered = _capture_table_render(formatter, table)
        assert "main.py" in rendered
        assert "150" in rendered
        assert "utils.py" in rendered
        assert "75" in rendered

    def test_table_title_rendered(self):
        """Table title appears in the rendered output."""
        formatter = _make_formatter()
        table = TableData(
            headers=["Column A", "Column B"],
            rows=[["val1", "val2"]],
            title="Stats",
        )
        rendered = _capture_table_render(formatter, table)
        assert "Stats" in rendered

    def test_empty_table_renders_headers_only(self):
        """A table with no rows still renders headers."""
        formatter = _make_formatter()
        table = TableData(
            headers=["Column A", "Column B"],
            rows=[],
            title="Empty",
        )
        rendered = _capture_table_render(formatter, table)
        assert "Column A" in rendered
        assert "Column B" in rendered

    def test_table_plain_mode_rendering(self):
        """Table in plain text mode uses text-based formatting."""
        formatter = _make_formatter()
        formatter._plain_mode = True
        table = TableData(
            headers=["Name", "Value"],
            rows=[["key", "data"]],
            title="Config",
        )
        rendered = _capture_table_render(formatter, table)
        assert "Name" in rendered
        assert "Value" in rendered
        assert "key" in rendered
        assert "data" in rendered
        # Plain text uses pipe separators
        assert "|" in rendered

    def test_tables_in_command_output(self):
        """Tables embedded in CommandOutput render correctly."""
        formatter = _make_formatter()
        output = CommandOutput(
            title="Report",
            content="Summary below:",
            tables=[
                TableData(
                    headers=["Module", "Deps"],
                    rows=[["agent", "3"], ["memory", "2"]],
                    title="Dependency Count",
                )
            ],
        )
        rendered = _capture_output(formatter, output)
        assert "Module" in rendered
        assert "agent" in rendered
        assert "memory" in rendered


# ─── Tests: Markdown Rendering (Req 6.4) ─────────────────────────────────────


class TestMarkdownRendering:
    """Test markdown content rendering (headings, bold, code blocks)."""

    def test_heading_rendered(self):
        """Markdown heading content appears in output."""
        formatter = _make_formatter()
        output = CommandOutput(
            title="Doc",
            content="# Main Heading\n\nSome content below.",
        )
        rendered = _capture_output(formatter, output)
        # Rich renders headings; text should be present
        assert "Main Heading" in rendered
        assert "Some content below." in rendered

    def test_bold_text_rendered(self):
        """Bold markdown text is present in the rendered output."""
        formatter = _make_formatter()
        output = CommandOutput(
            title="Doc",
            content="This is **important** information.",
        )
        rendered = _capture_output(formatter, output)
        assert "important" in rendered

    def test_code_block_rendered(self):
        """Fenced code block content appears in output."""
        formatter = _make_formatter()
        output = CommandOutput(
            title="Doc",
            content="```python\ndef foo():\n    pass\n```",
        )
        rendered = _capture_output(formatter, output)
        assert "foo" in rendered

    def test_inline_code_rendered(self):
        """Inline code is rendered in the output."""
        formatter = _make_formatter()
        output = CommandOutput(
            title="Doc",
            content="Use the `configure()` function.",
        )
        rendered = _capture_output(formatter, output)
        assert "configure()" in rendered

    def test_list_items_rendered(self):
        """Markdown list items appear in the output."""
        formatter = _make_formatter()
        output = CommandOutput(
            title="Doc",
            content="Features:\n- Fast indexing\n- Rich output\n- Pagination",
        )
        rendered = _capture_output(formatter, output)
        assert "Fast indexing" in rendered
        assert "Rich output" in rendered
        assert "Pagination" in rendered

    def test_empty_content_no_crash(self):
        """Empty content string does not cause errors."""
        formatter = _make_formatter()
        output = CommandOutput(title="Empty", content="")
        rendered = _capture_output(formatter, output)
        assert "Empty" in rendered


# ─── Tests: Pagination Logic (Req 6.5) ───────────────────────────────────────


class TestPaginationLogic:
    """Test pagination when output exceeds terminal height."""

    def test_short_output_no_pagination(self):
        """Short output prints directly without pagination."""
        formatter = _make_formatter()
        output = CommandOutput(
            title="Short",
            content="Brief content.",
        )
        # Should not raise, just prints
        rendered = _capture_output(formatter, output)
        assert "Brief content." in rendered

    def test_long_output_triggers_pagination_logic(self):
        """Output exceeding terminal height goes through pagination path.

        We test the line count estimation by checking the formatter method
        handles long content without error.
        """
        formatter = _make_formatter()
        # Create content that would exceed a typical terminal (24 lines)
        long_content = "\n".join([f"Line {i}: Some explanation text." for i in range(50)])
        output = CommandOutput(
            title="Long Report",
            content=long_content,
        )
        # When console is not a terminal, pagination is skipped (direct print)
        # This tests the non-terminal branch (StringIO is not a terminal)
        rendered = _capture_output(formatter, output)
        assert "Line 0:" in rendered
        assert "Line 49:" in rendered

    def test_pagination_method_handles_empty_renderables(self):
        """Pagination logic handles empty renderables list gracefully."""
        formatter = _make_formatter()
        # Directly test the pagination helper
        formatter._output_with_pagination([])
        # Should not raise

    def test_pagination_estimation_counts_lines(self):
        """The formatter estimates line count by rendering to a buffer."""
        formatter = _make_formatter()
        # Large output with many code snippets
        snippets = [
            CodeSnippet(code=f"line_{i} = {i}", language="python")
            for i in range(30)
        ]
        output = CommandOutput(
            title="Many Snippets",
            content="Code examples:",
            code_snippets=snippets,
        )
        # Should handle gracefully
        rendered = _capture_output(formatter, output)
        assert "line_0" in rendered
        assert "line_29" in rendered


# ─── Tests: Demo Mode Indicator (Req 8.3) ────────────────────────────────────


class TestDemoModeIndicator:
    """Test demo mode indicator display."""

    def test_demo_mode_indicator_shown(self):
        """When is_demo_mode=True, [DEMO MODE] appears in output."""
        formatter = _make_formatter()
        output = CommandOutput(
            title="Explain",
            content="Explanation here.",
            is_demo_mode=True,
        )
        rendered = _capture_output(formatter, output)
        assert "[DEMO MODE]" in rendered

    def test_no_demo_mode_indicator_when_disabled(self):
        """When is_demo_mode=False, [DEMO MODE] does not appear."""
        formatter = _make_formatter()
        output = CommandOutput(
            title="Explain",
            content="Explanation here.",
            is_demo_mode=False,
        )
        rendered = _capture_output(formatter, output)
        assert "[DEMO MODE]" not in rendered

    def test_demo_mode_in_plain_text(self):
        """Demo mode indicator shows in plain text mode too."""
        formatter = _make_formatter()
        formatter._plain_mode = True
        output = CommandOutput(
            title="Explain",
            content="Result.",
            is_demo_mode=True,
        )
        rendered = _capture_output(formatter, output)
        assert "[DEMO MODE]" in rendered


# ─── Tests: Confidence Score Color Coding ─────────────────────────────────────


class TestConfidenceScoreColorCoding:
    """Test confidence score color coding thresholds."""

    def test_high_confidence_green(self):
        """Score > 0.7 yields green color."""
        assert _get_confidence_color(0.95) == "green"
        assert _get_confidence_color(0.71) == "green"
        assert _get_confidence_color(1.0) == "green"

    def test_medium_confidence_yellow(self):
        """Score in [0.4, 0.7] yields yellow color."""
        assert _get_confidence_color(0.7) == "yellow"
        assert _get_confidence_color(0.5) == "yellow"
        assert _get_confidence_color(0.4) == "yellow"

    def test_low_confidence_red(self):
        """Score < 0.4 yields red color."""
        assert _get_confidence_color(0.39) == "red"
        assert _get_confidence_color(0.1) == "red"
        assert _get_confidence_color(0.0) == "red"

    def test_confidence_score_displayed_in_output(self):
        """Confidence score value appears in rendered output."""
        formatter = _make_formatter()
        output = CommandOutput(
            title="Analysis",
            content="Result.",
            confidence=0.85,
        )
        rendered = _capture_output(formatter, output)
        assert "Confidence: 85.0%" in rendered

    def test_no_confidence_no_display(self):
        """When confidence is None, no confidence line appears."""
        formatter = _make_formatter()
        output = CommandOutput(
            title="Analysis",
            content="Result.",
            confidence=None,
        )
        rendered = _capture_output(formatter, output)
        assert "Confidence" not in rendered

    def test_zero_confidence_displayed(self):
        """Zero confidence still displays (red color)."""
        formatter = _make_formatter()
        output = CommandOutput(
            title="Analysis",
            content="Uncertain.",
            confidence=0.0,
        )
        rendered = _capture_output(formatter, output)
        assert "Confidence: 0.0%" in rendered

    def test_boundary_confidence_0_7_is_yellow(self):
        """Exactly 0.7 is yellow (not green, since threshold is > 0.7)."""
        assert _get_confidence_color(0.7) == "yellow"

    def test_boundary_confidence_0_4_is_yellow(self):
        """Exactly 0.4 is yellow (not red, since threshold is >= 0.4)."""
        assert _get_confidence_color(0.4) == "yellow"
