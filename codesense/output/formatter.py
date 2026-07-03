"""Rich output formatter for CodeSense CLI output.

Provides terminal-formatted output with syntax highlighting, tables,
panels, markdown rendering, pagination, and accessibility fallbacks.
"""

import os
import shutil
from io import StringIO
from typing import Optional

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from codesense.models.output import CodeSnippet, CommandOutput, TableData
from codesense.models.state import Conflict


# File extension to language mapping for syntax highlighting
_EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "jsx",
    ".tsx": "tsx",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "zsh",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
    ".md": "markdown",
    ".sql": "sql",
    ".r": "r",
    ".lua": "lua",
    ".dart": "dart",
    ".ex": "elixir",
    ".exs": "elixir",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".clj": "clojure",
}


def _is_plain_text_mode() -> bool:
    """Check if terminal requires plain text output (no colors/formatting)."""
    if os.environ.get("NO_COLOR") is not None:
        return True
    term = os.environ.get("TERM", "")
    if term == "dumb":
        return True
    return False


def _detect_language(snippet: CodeSnippet) -> Optional[str]:
    """Detect programming language from code snippet metadata.

    Uses the snippet's language field directly, or infers from label
    (which may contain a file extension or code fence language).
    """
    # Direct language field
    if snippet.language and snippet.language.strip():
        lang = snippet.language.strip().lower()
        if lang != "unknown":
            return lang

    # Try to detect from label (could be a filename or extension)
    if snippet.label:
        label = snippet.label.strip().lower()
        # Check if label is a file extension
        if label.startswith("."):
            return _EXTENSION_MAP.get(label)
        # Check if label contains an extension
        for ext, lang in _EXTENSION_MAP.items():
            if label.endswith(ext):
                return lang
        # Label might be a language name directly
        if label in _EXTENSION_MAP.values():
            return label

    return None


class RichFormatter:
    """Formats CommandOutput for terminal display using Rich.

    Handles syntax highlighting, tables, panels, markdown rendering,
    pagination, and plain-text fallback for accessibility.
    """

    def __init__(self, console: Optional[Console] = None) -> None:
        """Initialize the formatter.

        Args:
            console: Optional Rich Console instance. If not provided,
                     one will be created based on terminal capabilities.
        """
        self._plain_mode = _is_plain_text_mode()

        if console is not None:
            self._console = console
        else:
            if self._plain_mode:
                self._console = Console(no_color=True, highlight=False)
            else:
                self._console = Console()

    @property
    def console(self) -> Console:
        """Access the underlying Rich Console."""
        return self._console

    def format_output(self, output: CommandOutput) -> None:
        """Render CommandOutput directly to the terminal using Rich.

        Applies syntax highlighting, formats structured data, renders markdown,
        displays demo mode indicator and confidence score, renders conflicts
        in visually distinct panels, and paginates if output exceeds terminal height.
        Falls back to plain text when TERM=dumb or NO_COLOR is set.

        Args:
            output: The CommandOutput to render.
        """
        if self._plain_mode:
            self._render_plain(output)
            return

        renderables: list = []

        # Demo mode indicator
        if output.is_demo_mode:
            renderables.append(Text("[DEMO MODE]", style="bold magenta"))
            renderables.append(Text())

        # Title panel
        renderables.append(
            Panel(Text(output.title, style="bold white"), style="blue")
        )

        # Confidence score
        if output.confidence is not None:
            color = _get_confidence_color(output.confidence)
            label = f"Confidence: {output.confidence:.1%}"
            renderables.append(Text(label, style=f"bold {color}"))
            renderables.append(Text())

        # Markdown content
        if output.content.strip():
            renderables.append(Markdown(output.content))
            renderables.append(Text())

        # Code snippets
        for snippet in output.code_snippets:
            renderables.append(self._build_code_renderable(snippet))
            renderables.append(Text())

        # Tables
        for table_data in output.tables:
            renderables.append(self._build_table_renderable(table_data))
            renderables.append(Text())

        # Conflicts
        if output.conflicts:
            renderables.append(
                self._build_conflicts_renderable(output.conflicts)
            )

        # Pagination check
        self._output_with_pagination(renderables)

    def render_code(self, snippet: CodeSnippet) -> None:
        """Render a code snippet with syntax highlighting.

        Falls back to plain monospaced text when language cannot be detected.

        Args:
            snippet: The code snippet to render.
        """
        if self._plain_mode:
            self._render_code_plain(snippet)
            return

        renderable = self._build_code_renderable(snippet)
        self._console.print(renderable)

    def render_table(self, table: TableData) -> None:
        """Render a table using Rich Table formatting.

        Args:
            table: The table data to render.
        """
        if self._plain_mode:
            self._render_table_plain(table)
            return

        renderable = self._build_table_renderable(table)
        self._console.print(renderable)

    def render_conflicts(self, conflicts: list[Conflict]) -> None:
        """Render conflict sections with numeric indices.

        Each conflict is displayed in a visually distinct panel with
        equal formatting for each side.

        Args:
            conflicts: List of conflicts to render.
        """
        if self._plain_mode:
            self._render_conflicts_plain(conflicts)
            return

        renderable = self._build_conflicts_renderable(conflicts)
        self._console.print(renderable)

    # ─── Private: Build Rich renderables ──────────────────────────────────

    def _build_code_renderable(self, snippet: CodeSnippet):
        """Build a Rich renderable for a code snippet."""
        language = _detect_language(snippet)

        if language:
            syntax = Syntax(
                snippet.code,
                language,
                theme="monokai",
                line_numbers=True,
                word_wrap=True,
            )
            if snippet.label:
                return Panel(syntax, title=snippet.label, border_style="dim")
            return syntax
        else:
            # Plain monospaced text fallback (no language detected)
            text = Text(snippet.code, style="dim")
            title = snippet.label or "Code"
            return Panel(text, title=title, border_style="dim")

    def _build_table_renderable(self, table_data: TableData) -> Table:
        """Build a Rich Table from TableData."""
        table = Table(
            title=table_data.title, show_header=True, header_style="bold cyan"
        )
        for header in table_data.headers:
            table.add_column(header)
        for row in table_data.rows:
            table.add_row(*row)
        return table

    def _build_conflicts_renderable(self, conflicts: list[Conflict]):
        """Build Rich renderables for a list of conflicts.

        Each conflict gets a distinct red-bordered panel with numeric index.
        All conflict panels are wrapped in an outer panel to visually separate
        them from the main answer content.
        """
        panels: list = []

        for idx, conflict in enumerate(conflicts, start=1):
            # Build side panels with equal formatting per source
            side_panels: list = []
            for source in conflict.sources:
                side_panel = Panel(
                    Text(source.claim),
                    title=f"Source: {source.source_id}",
                    border_style="yellow",
                    expand=True,
                )
                side_panels.append(side_panel)

            # Wrap in a container panel per conflict
            conflict_content = Group(*side_panels)
            conflict_panel = Panel(
                conflict_content,
                title=f"Conflict {idx}: {conflict.description}",
                border_style="red",
                subtitle=f"[{len(conflict.sources)} conflicting sources]",
            )
            panels.append(conflict_panel)

        # Wrap all conflict panels in a distinct outer panel to separate
        # from the main answer visually
        return Panel(
            Group(*panels),
            title="⚠ Conflicts Detected",
            border_style="bold red",
            padding=(1, 1),
        )

    # ─── Private: Plain text fallback ─────────────────────────────────────

    def _render_plain(self, output: CommandOutput) -> None:
        """Render full output as plain text without Rich formatting.

        Dynamic content (LLM answers, code, table cells) is printed with
        markup=False so that square brackets in code/text (e.g. arr[i],
        list[str], [INFO]) are never interpreted as Rich markup and dropped.
        """
        if output.is_demo_mode:
            self._console.print("[DEMO MODE]", markup=False)
            self._console.print()

        self._console.print(f"=== {output.title} ===", markup=False)
        self._console.print()

        if output.confidence is not None:
            self._console.print(f"Confidence: {output.confidence:.1%}", markup=False)
            self._console.print()

        if output.content.strip():
            self._console.print(output.content, markup=False, highlight=False)
            self._console.print()

        for snippet in output.code_snippets:
            self._render_code_plain(snippet)

        for table_data in output.tables:
            self._render_table_plain(table_data)

        if output.conflicts:
            self._render_conflicts_plain(output.conflicts)

    def _render_code_plain(self, snippet: CodeSnippet) -> None:
        """Render code snippet as plain monospaced text.

        markup=False keeps square brackets in code (indexing, type hints,
        decorators) intact instead of being parsed as Rich markup tags.
        """
        label = snippet.label or "Code"
        self._console.print(f"--- {label} ---", markup=False)
        self._console.print(snippet.code, markup=False, highlight=False)
        self._console.print()

    def _render_table_plain(self, table_data: TableData) -> None:
        """Render table as plain text."""
        if table_data.title:
            self._console.print(f"--- {table_data.title} ---", markup=False)

        # Calculate column widths
        col_widths = [len(h) for h in table_data.headers]
        for row in table_data.rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(cell))

        # Header
        header_line = " | ".join(
            h.ljust(col_widths[i]) for i, h in enumerate(table_data.headers)
        )
        self._console.print(header_line, markup=False, highlight=False)
        separator = "-+-".join("-" * w for w in col_widths)
        self._console.print(separator, markup=False)

        # Rows
        for row in table_data.rows:
            row_line = " | ".join(
                cell.ljust(col_widths[i]) if i < len(col_widths) else cell
                for i, cell in enumerate(row)
            )
            self._console.print(row_line, markup=False, highlight=False)

        self._console.print()

    def _render_conflicts_plain(self, conflicts: list[Conflict]) -> None:
        """Render conflicts as plain text with numeric indices."""
        self._console.print("=== Conflicts ===", markup=False)
        self._console.print()

        for idx, conflict in enumerate(conflicts, start=1):
            self._console.print(f"Conflict {idx}: {conflict.description}", markup=False)
            self._console.print("-" * 40, markup=False)
            for source in conflict.sources:
                self._console.print(
                    f"  Source [{source.source_id}]: {source.claim}",
                    markup=False,
                    highlight=False,
                )
            self._console.print()

    # ─── Private: Pagination ──────────────────────────────────────────────

    def _output_with_pagination(self, renderables: list) -> None:
        """Output with pagination when content exceeds terminal height.

        Uses Rich's built-in pager for long content when outputting to a
        real terminal. Falls back to direct printing for non-interactive output.
        """
        # Only paginate when writing to an actual terminal
        is_terminal = self._console.is_terminal

        if not is_terminal:
            for item in renderables:
                self._console.print(item)
            return

        terminal_height = shutil.get_terminal_size(fallback=(80, 24)).lines

        # Estimate total lines by rendering to a buffer
        buffer = StringIO()
        buffer_console = Console(
            file=buffer,
            width=self._console.width,
            no_color=True,
        )
        for item in renderables:
            buffer_console.print(item)

        rendered = buffer.getvalue()
        line_count = rendered.count("\n") + 1

        if line_count > terminal_height:
            with self._console.pager(styles=True):
                for item in renderables:
                    self._console.print(item)
        else:
            for item in renderables:
                self._console.print(item)


def _get_confidence_color(score: float) -> str:
    """Return color name based on confidence score thresholds.

    green: score > 0.7
    yellow: 0.4 <= score <= 0.7
    red: score < 0.4
    """
    if score > 0.7:
        return "green"
    elif score >= 0.4:
        return "yellow"
    else:
        return "red"
