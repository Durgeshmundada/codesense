"""Property-based tests for DecisionUnitChunker.

Tests Properties 21, 22, 23 from the design document using Hypothesis.

Validates: Requirements 10.2, 10.3, 10.5, 10.6
"""

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from codesense.memory.chunker import DecisionUnitChunker
from codesense.models.memory import DocumentMetadata


# --- Hypothesis strategies for generating ADR documents ---

def _metadata() -> DocumentMetadata:
    """Create a fixed DocumentMetadata for testing."""
    return DocumentMetadata(
        filename="test-adr.md",
        ingestion_timestamp="2024-01-01T00:00:00Z",
        source_path="/docs/test-adr.md",
    )


# Strategy: generate paragraph text that does NOT look like a boundary
# Avoids lines starting with #, digits followed by dot, or horizontal rules
_safe_word = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "Nd", "Zs"),
        whitelist_characters=" ",
    ),
    min_size=1,
    max_size=30,
).filter(lambda s: s.strip() != "")


_paragraph_line = st.builds(
    lambda words: " ".join(words),
    st.lists(_safe_word, min_size=2, max_size=8),
).map(lambda line: line.strip()).filter(
    lambda line: (
        len(line) > 0
        and not line.startswith("#")
        and not line[0].isdigit()
        and not all(c in "-*_" for c in line.replace(" ", ""))
    )
)


_paragraph = st.builds(
    lambda lines: "\n".join(lines),
    st.lists(_paragraph_line, min_size=1, max_size=5),
)


# Strategy: generate a markdown heading (H1-H4)
_heading_level = st.integers(min_value=1, max_value=4)
_heading_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "Nd", "Zs"), whitelist_characters=" "),
    min_size=3,
    max_size=40,
).filter(lambda s: s.strip() != "")


@st.composite
def heading_strategy(draw: st.DrawFn) -> str:
    """Generate a markdown heading like '## Decision Title'."""
    level = draw(_heading_level)
    text = draw(_heading_text).strip()
    return f"{'#' * level} {text}"


@st.composite
def adr_document_with_headings(draw: st.DrawFn) -> tuple[str, int]:
    """Generate an ADR document with N headings (1 <= N <= 10).

    Returns (document_text, number_of_headings).
    """
    num_headings = draw(st.integers(min_value=1, max_value=10))

    # Optionally add a preamble before the first heading
    parts: list[str] = []
    has_preamble = draw(st.booleans())
    if has_preamble:
        preamble = draw(_paragraph)
        parts.append(preamble)
        parts.append("\n")

    for i in range(num_headings):
        h = draw(heading_strategy())
        body = draw(_paragraph)
        parts.append(h)
        parts.append("\n")
        parts.append(body)
        if i < num_headings - 1:
            parts.append("\n")

    document = "\n".join(parts)
    return document, num_headings


@st.composite
def boundaryless_document(draw: st.DrawFn) -> str:
    """Generate a document with NO recognizable decision boundaries.

    No H1-H4 headings, no numbered sections (digit+dot at line start),
    no horizontal rules.
    """
    lines = draw(st.lists(_paragraph_line, min_size=1, max_size=20))
    document = "\n".join(lines)

    # Extra safety: ensure no line starts with # or is a horizontal rule
    for line in document.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            assume(False)
        if len(stripped) >= 3 and all(c in "-*_" for c in stripped):
            assume(False)
        # Check for numbered section: starts with digit(s) + "."
        if stripped and stripped[0].isdigit():
            parts = stripped.split(".", 1)
            if len(parts) > 1 and parts[0].isdigit():
                assume(False)

    return document


@st.composite
def any_adr_document(draw: st.DrawFn) -> str:
    """Generate any ADR document (with or without boundaries) for round-trip testing."""
    has_boundaries = draw(st.booleans())
    if has_boundaries:
        doc, _ = draw(adr_document_with_headings())
        return doc
    else:
        return draw(boundaryless_document())


# --- Property Tests ---

chunker = DecisionUnitChunker()


# Feature: codesense, Property 21: Decision-unit chunking round-trip
@settings(max_examples=100)
@given(document=any_adr_document())
def test_chunking_round_trip(document: str) -> None:
    """For any ADR document (string), chunking then concatenating all DecisionUnit
    text content in document order produces a character-for-character match with
    the original document content.

    **Validates: Requirements 10.5**
    """
    metadata = _metadata()
    result = chunker.chunk(document, metadata)

    # chunk() returns a list of DecisionUnits on success
    assert isinstance(result, list), f"Expected list of DecisionUnits, got {type(result)}"
    assert len(result) >= 1, "Chunker should produce at least one DecisionUnit"

    # Sort by order_index to ensure document order
    sorted_units = sorted(result, key=lambda u: u.order_index)

    # Concatenate all chunk contents
    reconstructed = "".join(unit.content for unit in sorted_units)

    assert reconstructed == document, (
        f"Round-trip failed.\n"
        f"Original length: {len(document)}\n"
        f"Reconstructed length: {len(reconstructed)}\n"
        f"Number of chunks: {len(result)}"
    )


# Feature: codesense, Property 22: One decision per chunk
@settings(max_examples=100)
@given(data=adr_document_with_headings())
def test_one_decision_per_chunk(data: tuple[str, int]) -> None:
    """For N identifiable decisions (markdown headings), the chunker produces
    exactly N DecisionUnits (plus any preamble).

    **Validates: Requirements 10.2, 10.3**
    """
    document, num_headings = data
    metadata = _metadata()
    result = chunker.chunk(document, metadata)

    assert isinstance(result, list), f"Expected list of DecisionUnits, got {type(result)}"

    # The chunker produces one chunk per boundary found. If there's content
    # before the first heading, it becomes a preamble chunk.
    # So total chunks = num_headings + (1 if preamble exists else 0)
    # We check that exactly num_headings boundary chunks exist
    # (chunks with has_structural_boundaries=True and heading != "(preamble)")
    boundary_chunks = [
        u for u in result
        if u.has_structural_boundaries and u.section_heading != "(preamble)"
    ]

    assert len(boundary_chunks) == num_headings, (
        f"Expected {num_headings} decision chunks but got {len(boundary_chunks)}. "
        f"Total chunks: {len(result)}, "
        f"Headings found: {[u.section_heading for u in result]}"
    )


# Feature: codesense, Property 23: Boundary-less document handling
@settings(max_examples=100)
@given(document=boundaryless_document())
def test_boundaryless_document_handling(document: str) -> None:
    """For any document that contains no recognizable decision boundaries
    (no H1-H4 headings, no numbered sections, no horizontal rules), the chunker
    produces exactly one DecisionUnit with has_structural_boundaries=False.

    **Validates: Requirements 10.6**
    """
    metadata = _metadata()
    result = chunker.chunk(document, metadata)

    assert isinstance(result, list), f"Expected list of DecisionUnits, got {type(result)}"
    assert len(result) == 1, (
        f"Expected exactly 1 DecisionUnit for boundary-less document, got {len(result)}"
    )
    assert result[0].has_structural_boundaries is False, (
        "DecisionUnit from boundary-less document should have has_structural_boundaries=False"
    )
    assert result[0].content == document, (
        "Single DecisionUnit content should match entire document"
    )
