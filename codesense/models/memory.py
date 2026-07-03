"""Decision Memory models for the RAG pipeline."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DocumentMetadata:
    """Metadata about an ingested document."""

    filename: str
    ingestion_timestamp: str
    source_path: str


@dataclass
class DecisionUnit:
    """A semantically coherent chunk representing a single architectural decision."""

    id: str
    content: str
    section_heading: str
    source_document: str
    ingestion_timestamp: str
    referenced_components: list[str] = field(default_factory=list)
    has_structural_boundaries: bool = True
    order_index: int = 0  # Position in original document for round-trip


@dataclass
class RetrievalResult:
    """A result from the vector store similarity search."""

    decision_unit: DecisionUnit
    similarity_score: float
    metadata: dict


@dataclass
class IngestResult:
    """Result of a document ingestion operation."""

    success: bool
    document_id: str
    chunks_created: int
    error: Optional[str] = None
