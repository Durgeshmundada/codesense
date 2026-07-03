"""Unit tests for IngestPipeline."""

import os
from unittest.mock import MagicMock, patch

import pytest

from codesense.memory.ingest import IngestPipeline
from codesense.models.memory import IngestResult


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1, 0.2, 0.3]]
    return embedder


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    return store


@pytest.fixture
def pipeline(mock_embedder, mock_vector_store):
    return IngestPipeline(
        embedder=mock_embedder,
        vector_store=mock_vector_store,
    )


@pytest.fixture
def valid_adr_content():
    return (
        "# ADR-001: Use PostgreSQL\n\n"
        "## Context\n\n"
        "We need a reliable database with ACID compliance.\n\n"
        "## Decision\n\n"
        "We will use PostgreSQL as our primary database.\n"
    )


class TestSuccessfulIngestion:
    """Tests for successful document ingestion."""

    def test_ingest_valid_adr_document(self, pipeline, tmp_path, valid_adr_content):
        """Test successful ingestion with a valid ADR markdown document."""
        adr_file = tmp_path / "adr-001.md"
        adr_file.write_text(valid_adr_content, encoding="utf-8")

        result = pipeline.ingest_document(str(adr_file))

        assert result.success is True
        assert result.document_id == "adr-001.md"
        assert result.chunks_created > 0
        assert result.error is None

    def test_ingest_calls_embedder_with_chunk_texts(
        self, mock_embedder, mock_vector_store, tmp_path, valid_adr_content
    ):
        """Test that the embedder receives chunk text content."""
        # Make embedder return correct number of embeddings
        mock_embedder.embed.return_value = [[0.1] * 3, [0.2] * 3, [0.3] * 3, [0.4] * 3]
        pipeline = IngestPipeline(embedder=mock_embedder, vector_store=mock_vector_store)

        adr_file = tmp_path / "adr-001.md"
        adr_file.write_text(valid_adr_content, encoding="utf-8")

        pipeline.ingest_document(str(adr_file))

        mock_embedder.embed.assert_called_once()
        texts_arg = mock_embedder.embed.call_args[0][0]
        assert isinstance(texts_arg, list)
        assert all(isinstance(t, str) for t in texts_arg)

    def test_ingest_calls_vector_store_with_chunks_and_embeddings(
        self, mock_embedder, mock_vector_store, tmp_path, valid_adr_content
    ):
        """Test that vector store receives chunks and embeddings."""
        mock_embedder.embed.return_value = [[0.1] * 3, [0.2] * 3, [0.3] * 3, [0.4] * 3]
        pipeline = IngestPipeline(embedder=mock_embedder, vector_store=mock_vector_store)

        adr_file = tmp_path / "adr-001.md"
        adr_file.write_text(valid_adr_content, encoding="utf-8")

        pipeline.ingest_document(str(adr_file))

        mock_vector_store.store.assert_called_once()


class TestAtomicRollback:
    """Tests for atomic rollback on failure."""

    def test_rollback_on_embedder_failure(
        self, mock_embedder, mock_vector_store, tmp_path, valid_adr_content
    ):
        """Test that no chunks are stored when embedder raises an exception."""
        mock_embedder.embed.side_effect = RuntimeError("Embedding model OOM")
        pipeline = IngestPipeline(embedder=mock_embedder, vector_store=mock_vector_store)

        adr_file = tmp_path / "adr-001.md"
        adr_file.write_text(valid_adr_content, encoding="utf-8")

        result = pipeline.ingest_document(str(adr_file))

        assert result.success is False
        assert result.chunks_created == 0
        assert "Embedding failed" in result.error
        # Vector store should never be called when embedder fails
        mock_vector_store.store.assert_not_called()

    def test_rollback_on_vector_store_failure(
        self, mock_embedder, mock_vector_store, tmp_path, valid_adr_content
    ):
        """Test rollback via delete_by_document when vector store raises."""
        mock_embedder.embed.return_value = [[0.1] * 3, [0.2] * 3, [0.3] * 3, [0.4] * 3]
        mock_vector_store.store.side_effect = RuntimeError("Chroma connection lost")
        pipeline = IngestPipeline(embedder=mock_embedder, vector_store=mock_vector_store)

        adr_file = tmp_path / "adr-001.md"
        adr_file.write_text(valid_adr_content, encoding="utf-8")

        result = pipeline.ingest_document(str(adr_file))

        assert result.success is False
        assert result.chunks_created == 0
        assert "Vector store failed" in result.error
        # Rollback should have been called
        mock_vector_store.delete_by_document.assert_called_once_with("adr-001.md")


class TestUnparseableDocument:
    """Tests for unparseable/None document handling."""

    def test_none_document_content_rejected(
        self, mock_embedder, mock_vector_store, tmp_path
    ):
        """Test that a file producing None from the chunker is rejected."""
        # The chunker returns IngestResult for None documents.
        # We can't easily pass None via file read, so we test by mocking the chunker.
        from codesense.memory.chunker import DecisionUnitChunker

        mock_chunker = MagicMock(spec=DecisionUnitChunker)
        mock_chunker.chunk.return_value = IngestResult(
            success=False,
            document_id="",
            chunks_created=0,
            error="Cannot parse document 'bad.md': document is None",
        )

        pipeline = IngestPipeline(
            chunker=mock_chunker,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
        )

        # Create a file that will be read, but the mocked chunker treats it as unparseable
        bad_file = tmp_path / "bad.md"
        bad_file.write_text("content", encoding="utf-8")

        result = pipeline.ingest_document(str(bad_file))

        assert result.success is False
        assert "Cannot parse" in result.error
        assert result.chunks_created == 0
        # Neither embedder nor vector store should be called
        mock_embedder.embed.assert_not_called()
        mock_vector_store.store.assert_not_called()


class TestFileNotFound:
    """Tests for file not found handling."""

    def test_file_not_found_returns_error(self, pipeline):
        """Test that a non-existent file path returns an error IngestResult."""
        result = pipeline.ingest_document("/nonexistent/path/adr-999.md")

        assert result.success is False
        assert result.document_id == "adr-999.md"
        assert result.chunks_created == 0
        assert "Failed to read file" in result.error


class TestIngestFolder:
    """Tests for ingest_folder directory walking."""

    def test_ingest_folder_processes_md_files(
        self, mock_embedder, mock_vector_store, tmp_path
    ):
        """Test that ingest_folder walks directory and processes .md files."""
        mock_embedder.embed.return_value = [[0.1] * 3]
        pipeline = IngestPipeline(embedder=mock_embedder, vector_store=mock_vector_store)

        # Create some .md files and one non-.md file
        (tmp_path / "adr-001.md").write_text("# Decision 1\n\nContent.", encoding="utf-8")
        (tmp_path / "adr-002.md").write_text("# Decision 2\n\nContent.", encoding="utf-8")
        (tmp_path / "readme.txt").write_text("Not a markdown file.", encoding="utf-8")

        results = pipeline.ingest_folder(str(tmp_path))

        # Only .md files should be processed
        assert len(results) == 2
        doc_ids = {r.document_id for r in results}
        assert "adr-001.md" in doc_ids
        assert "adr-002.md" in doc_ids

    def test_ingest_folder_empty_directory_returns_empty_list(
        self, pipeline, tmp_path
    ):
        """Test that an empty directory returns an empty result list."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        results = pipeline.ingest_folder(str(empty_dir))

        assert results == []

    def test_ingest_folder_nonexistent_path_returns_error(self, pipeline):
        """Test that a non-existent folder path returns an error IngestResult."""
        results = pipeline.ingest_folder("/nonexistent/folder/path")

        assert len(results) == 1
        assert results[0].success is False
        assert "Folder not found" in results[0].error
