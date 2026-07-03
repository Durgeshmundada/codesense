"""IngestPipeline for the Decision Memory RAG pipeline.

Wires together the full ingestion pipeline: chunker → embedder → vector_store.
Implements atomic ingestion semantics — either all chunks are stored or none.
"""

import os
from datetime import datetime, timezone

from codesense.memory.chunker import DecisionUnitChunker
from codesense.memory.embedder import HuggingFaceEmbedder
from codesense.memory.vector_store import VectorStore
from codesense.models.memory import DocumentMetadata, IngestResult


class IngestPipeline:
    """Orchestrates document ingestion through the RAG pipeline.

    Reads documents from disk, chunks them into DecisionUnits, embeds them,
    and stores them in the vector store. Implements atomic ingestion: if any
    step fails after partial work, previously stored data is rolled back.

    Args:
        chunker: DecisionUnitChunker instance for splitting documents.
            Creates a default instance if not provided.
        embedder: HuggingFaceEmbedder instance for generating embeddings.
            Creates a default instance if not provided.
        vector_store: VectorStore instance for persisting embedded chunks.
            Creates a default instance if not provided.
    """

    def __init__(
        self,
        chunker: DecisionUnitChunker | None = None,
        embedder: HuggingFaceEmbedder | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self._chunker = chunker or DecisionUnitChunker()
        self._embedder = embedder or HuggingFaceEmbedder()
        self._vector_store = vector_store or VectorStore()

    def ingest_document(self, path: str) -> IngestResult:
        """Ingest a single document through the full pipeline.

        Reads the file, chunks it, embeds the chunks, and stores them.
        Implements atomic semantics:
        - If chunking fails → no chunks stored, error returned immediately.
        - If embedding fails after chunking → no chunks stored.
        - If vector store fails after embedding → rollback via delete_by_document.

        Args:
            path: File path to the document to ingest.

        Returns:
            IngestResult indicating success or failure with details.
        """
        # Read the document from disk
        filename = os.path.basename(path)

        try:
            with open(path, "r", encoding="utf-8") as f:
                document = f.read()
        except (OSError, IOError) as e:
            return IngestResult(
                success=False,
                document_id=filename,
                chunks_created=0,
                error=f"Failed to read file '{path}': {e}",
            )

        # Create metadata
        metadata = DocumentMetadata(
            filename=filename,
            source_path=path,
            ingestion_timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Chunk the document
        chunk_result = self._chunker.chunk(document, metadata)

        # If chunker returns IngestResult (error), return it immediately
        if isinstance(chunk_result, IngestResult):
            return chunk_result

        chunks = chunk_result

        # Embed the chunks
        try:
            texts = [chunk.content for chunk in chunks]
            embeddings = self._embedder.embed(texts)
        except Exception as e:
            # Embedding failed after chunking → no chunks stored (atomic)
            return IngestResult(
                success=False,
                document_id=filename,
                chunks_created=0,
                error=f"Embedding failed for '{filename}': {e}",
            )

        # Store in vector store
        try:
            self._vector_store.store(chunks, embeddings)
        except Exception as e:
            # Vector store failed after embedding → rollback
            try:
                self._vector_store.delete_by_document(filename)
            except Exception:
                pass  # Best-effort rollback
            return IngestResult(
                success=False,
                document_id=filename,
                chunks_created=0,
                error=f"Vector store failed for '{filename}': {e}",
            )

        return IngestResult(
            success=True,
            document_id=filename,
            chunks_created=len(chunks),
        )

    def ingest_folder(self, folder_path: str) -> list[IngestResult]:
        """Walk a directory and ingest all .md files found.

        Args:
            folder_path: Path to the folder to walk recursively.

        Returns:
            A list of IngestResult objects, one per .md file found.
        """
        results: list[IngestResult] = []

        if not os.path.isdir(folder_path):
            results.append(
                IngestResult(
                    success=False,
                    document_id=folder_path,
                    chunks_created=0,
                    error=f"Folder not found: '{folder_path}'",
                )
            )
            return results

        for root, _dirs, files in os.walk(folder_path):
            for filename in sorted(files):
                if filename.endswith(".md"):
                    file_path = os.path.join(root, filename)
                    result = self.ingest_document(file_path)
                    results.append(result)

        return results
