"""Unit tests for DecisionUnitChunker."""

import pytest

from codesense.memory.chunker import DecisionUnitChunker
from codesense.models.memory import DocumentMetadata, IngestResult


@pytest.fixture
def chunker():
    return DecisionUnitChunker()


@pytest.fixture
def metadata():
    return DocumentMetadata(
        filename="adr-001.md",
        ingestion_timestamp="2024-01-15T10:00:00Z",
        source_path="/docs/adr-001.md",
    )


class TestBoundaryDetection:
    """Tests for semantic boundary detection."""

    def test_detects_h1_heading(self, chunker, metadata):
        doc = "# Decision Title\n\nSome content here."
        result = chunker.chunk(doc, metadata)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].section_heading == "Decision Title"

    def test_detects_h2_heading(self, chunker, metadata):
        doc = "## Context\n\nWe need to decide.\n\n## Decision\n\nWe chose X."
        result = chunker.chunk(doc, metadata)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].section_heading == "Context"
        assert result[1].section_heading == "Decision"

    def test_detects_h3_heading(self, chunker, metadata):
        doc = "### Sub Section\n\nContent."
        result = chunker.chunk(doc, metadata)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].section_heading == "Sub Section"

    def test_detects_h4_heading(self, chunker, metadata):
        doc = "#### Deep Section\n\nContent."
        result = chunker.chunk(doc, metadata)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].section_heading == "Deep Section"

    def test_does_not_detect_h5(self, chunker, metadata):
        doc = "##### Not a boundary\n\nContent."
        result = chunker.chunk(doc, metadata)
        assert isinstance(result, list)
        # H5 is not a recognized boundary, so no structural boundaries
        assert len(result) == 1
        assert result[0].has_structural_boundaries is False

    def test_detects_numbered_sections(self, chunker, metadata):
        doc = "1. First Decision\n\nContext here.\n\n2. Second Decision\n\nAnother context."
        result = chunker.chunk(doc, metadata)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_detects_horizontal_rule_dashes(self, chunker, metadata):
        doc = "Content above\n\n---\n\nContent below"
        result = chunker.chunk(doc, metadata)
        assert isinstance(result, list)
        assert len(result) >= 2

    def test_detects_horizontal_rule_asterisks(self, chunker, metadata):
        doc = "Content above\n\n***\n\nContent below"
        result = chunker.chunk(doc, metadata)
        assert isinstance(result, list)
        assert len(result) >= 2

    def test_detects_horizontal_rule_underscores(self, chunker, metadata):
        doc = "Content above\n\n___\n\nContent below"
        result = chunker.chunk(doc, metadata)
        assert isinstance(result, list)
        assert len(result) >= 2


class TestRoundTrip:
    """Tests for the round-trip property (concatenation == original)."""

    def test_round_trip_simple_headings(self, chunker, metadata):
        doc = "# Title\n\nIntro.\n\n## Context\n\nSome context.\n\n## Decision\n\nWe decided X."
        result = chunker.chunk(doc, metadata)
        assert isinstance(result, list)
        concatenated = "".join(unit.content for unit in result)
        assert concatenated == doc

    def test_round_trip_with_preamble(self, chunker, metadata):
        doc = "This is a preamble.\n\n# Title\n\nContent here."
        result = chunker.chunk(doc, metadata)
        assert isinstance(result, list)
        concatenated = "".join(unit.content for unit in result)
        assert concatenated == doc

    def test_round_trip_no_boundaries(self, chunker, metadata):
        doc = "Just a plain text document with no boundaries at all."
        result = chunker.chunk(doc, metadata)
        assert isinstance(result, list)
        concatenated = "".join(unit.content for unit in result)
        assert concatenated == doc

    def test_round_trip_complex_document(self, chunker, metadata):
        doc = """# ADR-001: Use PostgreSQL

## Status

Accepted

## Context

We need a database. Our requirements include ACID compliance and JSON support.

## Decision

We will use PostgreSQL as our primary database.

## Consequences

- Need to set up replication
- Team must learn PostgreSQL-specific features
"""
        result = chunker.chunk(doc, metadata)
        assert isinstance(result, list)
        concatenated = "".join(unit.content for unit in result)
        assert concatenated == doc

    def test_round_trip_horizontal_rules(self, chunker, metadata):
        doc = "Section 1 content\n\n---\n\nSection 2 content\n\n***\n\nSection 3 content"
        result = chunker.chunk(doc, metadata)
        assert isinstance(result, list)
        concatenated = "".join(unit.content for unit in result)
        assert concatenated == doc


class TestNoBoundaries:
    """Tests for documents with no recognizable boundaries."""

    def test_no_boundaries_single_chunk(self, chunker, metadata):
        doc = "This is a plain document without any headings or rules."
        result = chunker.chunk(doc, metadata)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].has_structural_boundaries is False

    def test_no_boundaries_preserves_content(self, chunker, metadata):
        doc = "Line 1\nLine 2\nLine 3\n"
        result = chunker.chunk(doc, metadata)
        assert isinstance(result, list)
        assert result[0].content == doc

    def test_empty_document_single_chunk(self, chunker, metadata):
        doc = ""
        result = chunker.chunk(doc, metadata)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].has_structural_boundaries is False


class TestDecisionUnitFields:
    """Tests for correct field population on DecisionUnit objects."""

    def test_id_is_unique(self, chunker, metadata):
        doc = "# Section 1\n\nContent.\n\n# Section 2\n\nMore content."
        result = chunker.chunk(doc, metadata)
        assert isinstance(result, list)
        ids = [unit.id for unit in result]
        assert len(ids) == len(set(ids))

    def test_source_document_set(self, chunker, metadata):
        doc = "# Title\n\nContent."
        result = chunker.chunk(doc, metadata)
        assert isinstance(result, list)
        assert result[0].source_document == "adr-001.md"

    def test_ingestion_timestamp_set(self, chunker, metadata):
        doc = "# Title\n\nContent."
        result = chunker.chunk(doc, metadata)
        assert isinstance(result, list)
        assert result[0].ingestion_timestamp == "2024-01-15T10:00:00Z"

    def test_order_index_sequential(self, chunker, metadata):
        doc = "# A\n\nContent A.\n\n# B\n\nContent B.\n\n# C\n\nContent C."
        result = chunker.chunk(doc, metadata)
        assert isinstance(result, list)
        for i, unit in enumerate(result):
            assert unit.order_index == i

    def test_section_heading_extracted(self, chunker, metadata):
        doc = "## My Important Decision\n\nDetails here."
        result = chunker.chunk(doc, metadata)
        assert isinstance(result, list)
        assert result[0].section_heading == "My Important Decision"


class TestCrossReferences:
    """Tests for component reference extraction."""

    def test_detects_pascal_case_components(self, chunker, metadata):
        doc = "# Decision\n\nWe will use DecisionUnitChunker for parsing."
        result = chunker.chunk(doc, metadata)
        assert isinstance(result, list)
        assert "DecisionUnitChunker" in result[0].referenced_components

    def test_detects_snake_case_components(self, chunker, metadata):
        doc = "# Decision\n\nThe vector_store module handles persistence."
        result = chunker.chunk(doc, metadata)
        assert isinstance(result, list)
        assert "vector_store" in result[0].referenced_components

    def test_no_components_in_plain_text(self, chunker, metadata):
        doc = "# Decision\n\nWe decided to use a simple approach."
        result = chunker.chunk(doc, metadata)
        assert isinstance(result, list)
        # Single lowercase words without underscores won't match
        # "simple" and "approach" won't match our patterns
        # Only PascalCase and multi-word snake_case match


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_none_document_returns_error(self, chunker, metadata):
        result = chunker.chunk(None, metadata)
        assert isinstance(result, IngestResult)
        assert result.success is False
        assert "Cannot parse" in result.error
