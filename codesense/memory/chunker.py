"""Decision-unit chunker for semantic boundary detection in ADR documents.

Splits documents into DecisionUnit objects based on structural boundaries:
- Markdown headings (H1-H4)
- Numbered section labels (e.g., "1.", "2.")
- Horizontal rules (---, ***, ___)
"""

import re
import uuid
from datetime import datetime, timezone

from codesense.models.memory import DecisionUnit, DocumentMetadata, IngestResult


# Patterns for detecting semantic boundaries
_HEADING_PATTERN = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
_NUMBERED_SECTION_PATTERN = re.compile(r"^(\d+\.)\s+(.+)$", re.MULTILINE)
_HORIZONTAL_RULE_PATTERN = re.compile(r"^([-*_])\1{2,}\s*$", re.MULTILINE)

# Pattern for detecting component references (PascalCase or snake_case identifiers)
_COMPONENT_PATTERN = re.compile(
    r"\b([A-Z][a-zA-Z0-9]+(?:[A-Z][a-zA-Z0-9]*)+)\b"  # PascalCase
    r"|\b([a-z][a-z0-9]*(?:_[a-z][a-z0-9]*)+)\b"  # snake_case (multi-word)
)


class DecisionUnitChunker:
    """Chunks ADR documents into semantically coherent DecisionUnit objects.

    Each DecisionUnit represents one complete decision context, preserving
    the round-trip property (concatenation of all chunks == original document).
    """

    def chunk(
        self, document: str, metadata: DocumentMetadata
    ) -> list[DecisionUnit] | IngestResult:
        """Chunk a document into DecisionUnit objects.

        Args:
            document: The full text content of the document.
            metadata: Metadata about the document being chunked.

        Returns:
            A list of DecisionUnit objects if successful, or an IngestResult
            with error details if the document cannot be parsed.
        """
        if not document and document != "":
            return IngestResult(
                success=False,
                document_id="",
                chunks_created=0,
                error=f"Cannot parse document '{metadata.filename}': document is None",
            )

        # Find all boundary positions in the document
        boundaries = self._find_boundaries(document)

        # If no boundaries found, return single chunk with flag
        if not boundaries:
            unit = self._create_decision_unit(
                content=document,
                section_heading="(no heading)",
                metadata=metadata,
                order_index=0,
                has_structural_boundaries=False,
            )
            return [unit]

        # Split document at boundaries
        chunks = self._split_at_boundaries(document, boundaries)

        # Create DecisionUnit objects
        units: list[DecisionUnit] = []
        for i, (content, heading) in enumerate(chunks):
            unit = self._create_decision_unit(
                content=content,
                section_heading=heading,
                metadata=metadata,
                order_index=i,
                has_structural_boundaries=True,
            )
            units.append(unit)

        return units

    def _find_boundaries(self, document: str) -> list[tuple[int, str]]:
        """Find all semantic boundary positions in the document.

        Returns a sorted list of (position, heading_text) tuples where
        position is the character offset of the boundary line start.
        """
        boundaries: list[tuple[int, str]] = []

        # Find markdown headings (H1-H4)
        for match in _HEADING_PATTERN.finditer(document):
            boundaries.append((match.start(), match.group(2).strip()))

        # Find numbered section labels
        for match in _NUMBERED_SECTION_PATTERN.finditer(document):
            # Avoid matching numbered items within a paragraph (only match at
            # beginning of line, which the regex already ensures with ^)
            boundaries.append((match.start(), match.group(0).strip()))

        # Find horizontal rules
        for match in _HORIZONTAL_RULE_PATTERN.finditer(document):
            boundaries.append((match.start(), "(horizontal rule)"))

        # Sort by position and deduplicate positions (keep first heading found)
        boundaries.sort(key=lambda x: x[0])
        deduplicated: list[tuple[int, str]] = []
        seen_positions: set[int] = set()
        for pos, heading in boundaries:
            if pos not in seen_positions:
                deduplicated.append((pos, heading))
                seen_positions.add(pos)

        return deduplicated

    def _split_at_boundaries(
        self, document: str, boundaries: list[tuple[int, str]]
    ) -> list[tuple[str, str]]:
        """Split document at boundary positions.

        Returns a list of (content, heading) tuples preserving all characters.
        """
        chunks: list[tuple[str, str]] = []

        # If document starts before first boundary, include preamble
        if boundaries[0][0] > 0:
            preamble = document[: boundaries[0][0]]
            chunks.append((preamble, "(preamble)"))

        # Split at each boundary
        for i, (pos, heading) in enumerate(boundaries):
            if i + 1 < len(boundaries):
                next_pos = boundaries[i + 1][0]
                content = document[pos:next_pos]
            else:
                content = document[pos:]
            chunks.append((content, heading))

        return chunks

    def _create_decision_unit(
        self,
        content: str,
        section_heading: str,
        metadata: DocumentMetadata,
        order_index: int,
        has_structural_boundaries: bool,
    ) -> DecisionUnit:
        """Create a DecisionUnit with all required fields populated."""
        referenced_components = self._extract_component_references(content)

        return DecisionUnit(
            id=str(uuid.uuid4()),
            content=content,
            section_heading=section_heading,
            source_document=metadata.filename,
            ingestion_timestamp=metadata.ingestion_timestamp,
            referenced_components=referenced_components,
            has_structural_boundaries=has_structural_boundaries,
            order_index=order_index,
        )

    def _extract_component_references(self, content: str) -> list[str]:
        """Extract referenced component names from chunk content.

        Looks for PascalCase and multi-word snake_case identifiers which
        typically indicate component or class references.
        """
        components: set[str] = set()

        for match in _COMPONENT_PATTERN.finditer(content):
            # Group 1 is PascalCase, Group 2 is snake_case
            component = match.group(1) or match.group(2)
            if component:
                components.add(component)

        return sorted(components)
