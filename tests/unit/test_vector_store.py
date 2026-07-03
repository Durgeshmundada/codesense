"""Unit tests for VectorStore (Chroma backend)."""

import json
import os
import shutil
import tempfile

import pytest

from codesense.memory.vector_store import VectorStore
from codesense.models.memory import DecisionUnit, RetrievalResult


@pytest.fixture
def temp_dir():
    """Create a temporary directory for Chroma persistence."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def store(temp_dir):
    """Create a VectorStore instance with a temp directory."""
    return VectorStore(
        persist_directory=temp_dir,
        collection_name="test_collection",
    )


def _make_unit(
    id: str = "unit-1",
    content: str = "Test content",
    section_heading: str = "Test Heading",
    source_document: str = "test.md",
    ingestion_timestamp: str = "2024-01-01T00:00:00Z",
    order_index: int = 0,
    has_structural_boundaries: bool = True,
    referenced_components: list[str] | None = None,
) -> DecisionUnit:
    """Helper to create a DecisionUnit for testing."""
    return DecisionUnit(
        id=id,
        content=content,
        section_heading=section_heading,
        source_document=source_document,
        ingestion_timestamp=ingestion_timestamp,
        order_index=order_index,
        has_structural_boundaries=has_structural_boundaries,
        referenced_components=referenced_components or [],
    )


def _make_embedding(dim: int = 384, value: float = 0.1) -> list[float]:
    """Create a simple embedding vector."""
    return [value] * dim


class TestVectorStoreInit:
    """Tests for VectorStore initialization."""

    def test_creates_with_defaults(self, temp_dir):
        vs = VectorStore(persist_directory=temp_dir)
        assert vs._collection.name == "decision_memory"

    def test_creates_with_custom_collection_name(self, temp_dir):
        vs = VectorStore(persist_directory=temp_dir, collection_name="custom")
        assert vs._collection.name == "custom"

    def test_collection_uses_cosine_metric(self, store):
        metadata = store._collection.metadata
        assert metadata.get("hnsw:space") == "cosine"


class TestStore:
    """Tests for the store() method."""

    def test_store_empty_list_is_noop(self, store):
        store.store([], [])
        assert store._collection.count() == 0

    def test_store_single_unit(self, store):
        unit = _make_unit()
        embedding = _make_embedding()
        store.store([unit], [embedding])
        assert store._collection.count() == 1

    def test_store_multiple_units(self, store):
        units = [_make_unit(id=f"unit-{i}") for i in range(3)]
        embeddings = [_make_embedding(value=0.1 * (i + 1)) for i in range(3)]
        store.store(units, embeddings)
        assert store._collection.count() == 3

    def test_store_preserves_metadata(self, store):
        unit = _make_unit(
            id="meta-test",
            source_document="architecture.md",
            section_heading="Database Choice",
            ingestion_timestamp="2024-06-15T12:00:00Z",
            order_index=3,
            has_structural_boundaries=True,
            referenced_components=["DatabaseService", "UserRepo"],
        )
        embedding = _make_embedding()
        store.store([unit], [embedding])

        result = store._collection.get(ids=["meta-test"], include=["metadatas"])
        meta = result["metadatas"][0]
        assert meta["source_document"] == "architecture.md"
        assert meta["section_heading"] == "Database Choice"
        assert meta["ingestion_timestamp"] == "2024-06-15T12:00:00Z"
        assert meta["order_index"] == 3
        assert meta["has_structural_boundaries"] is True
        assert json.loads(meta["referenced_components"]) == [
            "DatabaseService",
            "UserRepo",
        ]

    def test_store_mismatched_lengths_raises(self, store):
        units = [_make_unit(id="a"), _make_unit(id="b")]
        embeddings = [_make_embedding()]
        with pytest.raises(ValueError, match="must match"):
            store.store(units, embeddings)

    def test_store_upserts_existing_ids(self, store):
        unit = _make_unit(id="upsert-test", content="original")
        embedding = _make_embedding()
        store.store([unit], [embedding])

        updated_unit = _make_unit(id="upsert-test", content="updated")
        store.store([updated_unit], [embedding])

        assert store._collection.count() == 1
        result = store._collection.get(ids=["upsert-test"], include=["documents"])
        assert result["documents"][0] == "updated"


class TestQuery:
    """Tests for the query() method."""

    def test_query_empty_collection_returns_empty(self, store):
        results = store.query(_make_embedding())
        assert results == []

    def test_query_returns_retrieval_results(self, store):
        unit = _make_unit(content="Python is great for data science")
        embedding = _make_embedding(value=0.5)
        store.store([unit], [embedding])

        # Query with exact same embedding should give high similarity
        results = store.query(embedding, top_k=5, min_similarity=0.0)
        assert len(results) >= 1
        assert isinstance(results[0], RetrievalResult)

    def test_query_filters_by_min_similarity(self, store):
        unit = _make_unit()
        embedding = _make_embedding(value=0.5)
        store.store([unit], [embedding])

        # Query with a very different embedding
        different_embedding = _make_embedding(value=-0.5)
        results = store.query(different_embedding, min_similarity=0.99)
        assert results == []

    def test_query_exact_match_high_similarity(self, store):
        unit = _make_unit(content="exact match test")
        embedding = _make_embedding(value=0.3)
        store.store([unit], [embedding])

        # Same embedding should yield similarity ~1.0
        results = store.query(embedding, min_similarity=0.9)
        assert len(results) == 1
        assert results[0].similarity_score >= 0.9

    def test_query_results_sorted_descending(self, store):
        # Store units with different embeddings
        units = [_make_unit(id=f"sort-{i}", content=f"content {i}") for i in range(3)]
        embeddings = [
            _make_embedding(value=0.1),
            _make_embedding(value=0.5),
            _make_embedding(value=0.9),
        ]
        store.store(units, embeddings)

        # Query should return results sorted by similarity descending
        query_emb = _make_embedding(value=0.5)
        results = store.query(query_emb, top_k=10, min_similarity=0.0)
        similarities = [r.similarity_score for r in results]
        assert similarities == sorted(similarities, reverse=True)

    def test_query_respects_top_k(self, store):
        units = [_make_unit(id=f"topk-{i}") for i in range(10)]
        embeddings = [_make_embedding(value=0.1 * (i + 1)) for i in range(10)]
        store.store(units, embeddings)

        results = store.query(_make_embedding(value=0.5), top_k=3, min_similarity=0.0)
        assert len(results) <= 3

    def test_query_reconstructs_decision_unit(self, store):
        unit = _make_unit(
            id="reconstruct-test",
            content="Decision content here",
            section_heading="Architecture",
            source_document="adr-001.md",
            ingestion_timestamp="2024-03-01T10:00:00Z",
            order_index=2,
            has_structural_boundaries=False,
            referenced_components=["AuthService", "TokenStore"],
        )
        embedding = _make_embedding(value=0.4)
        store.store([unit], [embedding])

        results = store.query(embedding, min_similarity=0.0)
        assert len(results) == 1
        du = results[0].decision_unit
        assert du.id == "reconstruct-test"
        assert du.content == "Decision content here"
        assert du.section_heading == "Architecture"
        assert du.source_document == "adr-001.md"
        assert du.ingestion_timestamp == "2024-03-01T10:00:00Z"
        assert du.order_index == 2
        assert du.has_structural_boundaries is False
        assert du.referenced_components == ["AuthService", "TokenStore"]

    def test_query_includes_metadata_in_result(self, store):
        unit = _make_unit(
            source_document="decisions.md",
            section_heading="Caching Strategy",
            ingestion_timestamp="2024-02-20T08:30:00Z",
        )
        embedding = _make_embedding(value=0.6)
        store.store([unit], [embedding])

        results = store.query(embedding, min_similarity=0.0)
        assert len(results) == 1
        meta = results[0].metadata
        assert meta["source_document"] == "decisions.md"
        assert meta["section_heading"] == "Caching Strategy"
        assert meta["ingestion_timestamp"] == "2024-02-20T08:30:00Z"


class TestDeleteCollection:
    """Tests for the delete_collection() method."""

    def test_delete_collection_removes_all_data(self, store):
        units = [_make_unit(id=f"del-{i}") for i in range(5)]
        embeddings = [_make_embedding() for _ in range(5)]
        store.store(units, embeddings)
        assert store._collection.count() == 5

        store.delete_collection()

        # Collection should no longer exist
        collections = store._client.list_collections()
        collection_names = [c.name for c in collections]
        assert "test_collection" not in collection_names


class TestDeleteByDocument:
    """Tests for the delete_by_document() method."""

    def test_delete_by_document_removes_matching(self, store):
        units = [
            _make_unit(id="doc1-chunk1", source_document="doc1.md"),
            _make_unit(id="doc1-chunk2", source_document="doc1.md"),
            _make_unit(id="doc2-chunk1", source_document="doc2.md"),
        ]
        embeddings = [_make_embedding() for _ in range(3)]
        store.store(units, embeddings)

        store.delete_by_document("doc1.md")
        assert store._collection.count() == 1

        remaining = store._collection.get(ids=["doc2-chunk1"])
        assert len(remaining["ids"]) == 1
