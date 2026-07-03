"""Unit tests for conflict display rendering.

Validates Requirements 7.3, 7.4, 7.5:
- Conflicts displayed in visually distinct sections separated from main answer
- Each conflict has a numeric index label
- Each side receives equal formatting and equivalent space
- Multiple conflicts listed with sequential numeric indices
- No conflict section when no conflicts exist
"""

import os
from io import StringIO
from unittest.mock import patch

from rich.console import Console

from codesense.models.output import CommandOutput
from codesense.models.state import Conflict, ConflictSource
from codesense.output.formatter import RichFormatter


def _make_conflict(
    id: str, description: str, sources: list[tuple[str, str]]
) -> Conflict:
    """Helper to create a Conflict with ConflictSource pairs."""
    return Conflict(
        id=id,
        sources=[ConflictSource(source_id=sid, claim=claim) for sid, claim in sources],
        description=description,
    )


def _capture_rich_output(formatter: RichFormatter, output: CommandOutput) -> str:
    """Capture formatted output as a string."""
    buffer = StringIO()
    console = Console(file=buffer, width=120, no_color=True, highlight=False)
    formatter._console = console
    formatter.format_output(output)
    return buffer.getvalue()


def _capture_plain_output(output: CommandOutput) -> str:
    """Capture plain text output (NO_COLOR mode)."""
    buffer = StringIO()
    console = Console(file=buffer, width=120, no_color=True, highlight=False)
    with patch.dict(os.environ, {"NO_COLOR": "1"}):
        formatter = RichFormatter(console=console)
        formatter._plain_mode = True
        formatter.format_output(output)
    return buffer.getvalue()


class TestSingleConflictRendering:
    """Test rendering of a single conflict."""

    def test_single_conflict_has_numeric_index(self):
        """A single conflict renders with 'Conflict 1:' label."""
        conflict = _make_conflict(
            id="c1",
            description="Authorship dispute",
            sources=[
                ("git-log", "File was authored by Alice"),
                ("pr-comment", "File was authored by Bob"),
            ],
        )
        output = CommandOutput(
            title="Test",
            content="Main answer here.",
            conflicts=[conflict],
        )
        formatter = RichFormatter(
            console=Console(file=StringIO(), width=120, no_color=True)
        )
        rendered = _capture_rich_output(formatter, output)

        assert "Conflict 1:" in rendered

    def test_single_conflict_shows_all_sources(self):
        """Each source's claim and ID appear in the output."""
        conflict = _make_conflict(
            id="c1",
            description="Version conflict",
            sources=[
                ("commit-abc", "Module was added in v2.0"),
                ("issue-42", "Module was added in v1.5"),
            ],
        )
        output = CommandOutput(
            title="Test",
            content="Answer.",
            conflicts=[conflict],
        )
        formatter = RichFormatter(
            console=Console(file=StringIO(), width=120, no_color=True)
        )
        rendered = _capture_rich_output(formatter, output)

        assert "commit-abc" in rendered
        assert "issue-42" in rendered
        assert "Module was added in v2.0" in rendered
        assert "Module was added in v1.5" in rendered

    def test_single_conflict_visually_separated(self):
        """Conflict section is visually distinct from main answer (wrapped in Conflicts panel)."""
        conflict = _make_conflict(
            id="c1",
            description="Ownership",
            sources=[
                ("src-a", "Claim A"),
                ("src-b", "Claim B"),
            ],
        )
        output = CommandOutput(
            title="Test",
            content="Main answer text.",
            conflicts=[conflict],
        )
        formatter = RichFormatter(
            console=Console(file=StringIO(), width=120, no_color=True)
        )
        rendered = _capture_rich_output(formatter, output)

        # The outer panel title indicates a conflicts section
        assert "Conflicts Detected" in rendered

    def test_each_side_receives_equal_formatting(self):
        """Both sides of a conflict get their own labeled source section."""
        conflict = _make_conflict(
            id="c1",
            description="Timing",
            sources=[
                ("source-alpha", "Happened in January"),
                ("source-beta", "Happened in March"),
            ],
        )
        output = CommandOutput(
            title="Test",
            content="",
            conflicts=[conflict],
        )
        formatter = RichFormatter(
            console=Console(file=StringIO(), width=120, no_color=True)
        )
        rendered = _capture_rich_output(formatter, output)

        # Each source gets its own "Source: ..." label
        assert "Source: source-alpha" in rendered
        assert "Source: source-beta" in rendered


class TestMultipleConflictsRendering:
    """Test rendering of multiple conflicts with sequential indices."""

    def test_multiple_conflicts_sequential_indices(self):
        """Multiple conflicts get sequential numeric indices."""
        conflicts = [
            _make_conflict(
                id="c1",
                description="First issue",
                sources=[("s1", "Claim A"), ("s2", "Claim B")],
            ),
            _make_conflict(
                id="c2",
                description="Second issue",
                sources=[("s3", "Claim C"), ("s4", "Claim D")],
            ),
            _make_conflict(
                id="c3",
                description="Third issue",
                sources=[("s5", "Claim E"), ("s6", "Claim F")],
            ),
        ]
        output = CommandOutput(
            title="Test",
            content="Answer.",
            conflicts=conflicts,
        )
        formatter = RichFormatter(
            console=Console(file=StringIO(), width=120, no_color=True)
        )
        rendered = _capture_rich_output(formatter, output)

        assert "Conflict 1:" in rendered
        assert "Conflict 2:" in rendered
        assert "Conflict 3:" in rendered

    def test_multiple_conflicts_all_descriptions_present(self):
        """Each conflict's description is shown in the output."""
        conflicts = [
            _make_conflict(
                id="c1",
                description="Design rationale disagreement",
                sources=[("s1", "Claim A"), ("s2", "Claim B")],
            ),
            _make_conflict(
                id="c2",
                description="Timeline inconsistency",
                sources=[("s3", "Claim C"), ("s4", "Claim D")],
            ),
        ]
        output = CommandOutput(
            title="Test",
            content="Answer.",
            conflicts=conflicts,
        )
        formatter = RichFormatter(
            console=Console(file=StringIO(), width=120, no_color=True)
        )
        rendered = _capture_rich_output(formatter, output)

        assert "Design rationale disagreement" in rendered
        assert "Timeline inconsistency" in rendered

    def test_conflict_source_count_subtitle(self):
        """Each conflict shows the number of conflicting sources."""
        conflict = _make_conflict(
            id="c1",
            description="Multi-source",
            sources=[
                ("s1", "Claim A"),
                ("s2", "Claim B"),
                ("s3", "Claim C"),
            ],
        )
        output = CommandOutput(
            title="Test",
            content="",
            conflicts=[conflict],
        )
        formatter = RichFormatter(
            console=Console(file=StringIO(), width=120, no_color=True)
        )
        rendered = _capture_rich_output(formatter, output)

        assert "3 conflicting sources" in rendered


class TestNoConflictsRendering:
    """Test that no conflict section appears when there are no conflicts."""

    def test_no_conflicts_no_conflict_section(self):
        """When conflicts list is empty, no conflict-related text appears."""
        output = CommandOutput(
            title="Test",
            content="A clean answer with no conflicts.",
            conflicts=[],
        )
        formatter = RichFormatter(
            console=Console(file=StringIO(), width=120, no_color=True)
        )
        rendered = _capture_rich_output(formatter, output)

        assert "Conflict" not in rendered
        assert "Conflicts Detected" not in rendered

    def test_no_conflicts_main_answer_renders(self):
        """Without conflicts, the main answer content still renders normally."""
        output = CommandOutput(
            title="Explanation",
            content="This code exists because of requirement X.",
            conflicts=[],
        )
        formatter = RichFormatter(
            console=Console(file=StringIO(), width=120, no_color=True)
        )
        rendered = _capture_rich_output(formatter, output)

        assert "This code exists because of requirement X." in rendered
        assert "Explanation" in rendered


class TestPlainTextConflictRendering:
    """Test plain text fallback also renders conflicts with numeric indices."""

    def test_plain_text_has_conflicts_header(self):
        """Plain text output includes a Conflicts section header."""
        conflict = _make_conflict(
            id="c1",
            description="Disagreement",
            sources=[("s1", "A"), ("s2", "B")],
        )
        output = CommandOutput(
            title="Test",
            content="Answer.",
            conflicts=[conflict],
        )
        rendered = _capture_plain_output(output)

        assert "=== Conflicts ===" in rendered

    def test_plain_text_numeric_indices(self):
        """Plain text conflict rendering uses numeric indices."""
        conflicts = [
            _make_conflict(
                id="c1",
                description="First",
                sources=[("s1", "A"), ("s2", "B")],
            ),
            _make_conflict(
                id="c2",
                description="Second",
                sources=[("s3", "C"), ("s4", "D")],
            ),
        ]
        output = CommandOutput(
            title="Test",
            content="Answer.",
            conflicts=conflicts,
        )
        rendered = _capture_plain_output(output)

        assert "Conflict 1:" in rendered
        assert "Conflict 2:" in rendered

    def test_plain_text_shows_sources(self):
        """Plain text conflict rendering shows source IDs and claims."""
        conflict = _make_conflict(
            id="c1",
            description="Dispute",
            sources=[
                ("git-log-123", "Changed for performance"),
                ("adr-005", "Changed for readability"),
            ],
        )
        output = CommandOutput(
            title="Test",
            content="Answer.",
            conflicts=[conflict],
        )
        rendered = _capture_plain_output(output)

        assert "git-log-123" in rendered
        assert "adr-005" in rendered
        assert "Changed for performance" in rendered
        assert "Changed for readability" in rendered

    def test_plain_text_no_conflicts_no_section(self):
        """Plain text output with no conflicts has no Conflicts section."""
        output = CommandOutput(
            title="Test",
            content="Clean answer.",
            conflicts=[],
        )
        rendered = _capture_plain_output(output)

        assert "Conflict" not in rendered
        assert "=== Conflicts ===" not in rendered
