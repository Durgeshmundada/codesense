"""Property-based tests for demo mode indicator presence.

Tests Property 19 from the design document using Hypothesis.

**Validates: Requirements 8.3**
"""

from io import StringIO

from hypothesis import given, settings
from hypothesis import strategies as st
from rich.console import Console

from codesense.models.output import CommandOutput, CodeSnippet, TableData
from codesense.models.state import Conflict, ConflictSource
from codesense.output.formatter import RichFormatter


# --- Strategies ---

# Strategy for non-empty text content
text_st = st.text(min_size=1, max_size=200, alphabet=st.characters(
    whitelist_categories=("L", "N", "P", "Z"),
    blacklist_characters="\x00",
))

# Strategy for optional confidence score
confidence_st = st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0))

# Strategy for code snippets
code_snippet_st = st.builds(
    CodeSnippet,
    code=st.text(min_size=1, max_size=100, alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z"),
        blacklist_characters="\x00",
    )),
    language=st.sampled_from(["python", "javascript", "typescript", "go", "rust", "unknown", ""]),
    label=st.one_of(st.none(), st.text(min_size=1, max_size=30, alphabet=st.characters(
        whitelist_categories=("L", "N"),
    ))),
)

# Strategy for table data
table_data_st = st.builds(
    TableData,
    headers=st.lists(st.text(min_size=1, max_size=20, alphabet=st.characters(
        whitelist_categories=("L", "N"),
    )), min_size=1, max_size=5),
    rows=st.lists(
        st.lists(st.text(min_size=0, max_size=20, alphabet=st.characters(
            whitelist_categories=("L", "N"),
        )), min_size=1, max_size=5),
        min_size=0, max_size=5,
    ),
    title=st.one_of(st.none(), st.text(min_size=1, max_size=30, alphabet=st.characters(
        whitelist_categories=("L", "N"),
    ))),
)

# Strategy for conflict sources
conflict_source_st = st.builds(
    ConflictSource,
    source_id=st.text(min_size=1, max_size=30, alphabet=st.characters(
        whitelist_categories=("L", "N"),
    )),
    claim=st.text(min_size=1, max_size=100, alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z"),
        blacklist_characters="\x00",
    )),
)

# Strategy for conflicts
conflict_st = st.builds(
    Conflict,
    id=st.text(min_size=1, max_size=20, alphabet=st.characters(
        whitelist_categories=("L", "N"),
    )),
    sources=st.lists(conflict_source_st, min_size=2, max_size=4),
    description=st.text(min_size=1, max_size=100, alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z"),
        blacklist_characters="\x00",
    )),
)

# Strategy for CommandOutput with is_demo_mode=True
demo_mode_output_st = st.builds(
    CommandOutput,
    title=text_st,
    content=text_st,
    code_snippets=st.lists(code_snippet_st, min_size=0, max_size=3),
    tables=st.lists(table_data_st, min_size=0, max_size=2),
    conflicts=st.lists(conflict_st, min_size=0, max_size=2),
    confidence=confidence_st,
    is_demo_mode=st.just(True),
)


# --- Property 19 Tests ---


# Feature: codesense, Property 19: Demo mode indicator presence
@settings(max_examples=100)
@given(output=demo_mode_output_st)
def test_demo_mode_indicator_present_in_output(output: CommandOutput) -> None:
    """For any command executed with mock mode enabled (is_demo_mode=True),
    the formatted output contains the "[DEMO MODE]" indicator string.

    **Validates: Requirements 8.3**
    """
    # Capture formatter output to a StringIO buffer
    buffer = StringIO()
    console = Console(file=buffer, width=120, no_color=True)
    formatter = RichFormatter(console=console)

    formatter.format_output(output)

    rendered = buffer.getvalue()
    assert "[DEMO MODE]" in rendered, (
        f"Expected '[DEMO MODE]' indicator in output but it was not found.\n"
        f"Output title: {output.title!r}\n"
        f"Rendered output:\n{rendered[:500]}"
    )
