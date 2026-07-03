"""Unit tests for HuggingFaceEmbedder."""

import pytest

from codesense.memory.embedder import HuggingFaceEmbedder


@pytest.fixture(scope="module")
def embedder():
    """Create a shared embedder instance (model loading is expensive)."""
    return HuggingFaceEmbedder()


class TestHuggingFaceEmbedderInit:
    """Tests for embedder initialization."""

    def test_default_model_name(self):
        emb = HuggingFaceEmbedder()
        assert emb._model_name == "all-MiniLM-L6-v2"

    def test_custom_model_name(self):
        emb = HuggingFaceEmbedder(model_name="paraphrase-MiniLM-L3-v2")
        assert emb._model_name == "paraphrase-MiniLM-L3-v2"

    def test_lazy_loading_model_not_loaded_on_init(self):
        emb = HuggingFaceEmbedder()
        assert emb._model is None

    def test_model_loaded_on_first_access(self, embedder):
        _ = embedder.model
        assert embedder._model is not None

    def test_max_seq_length_set_to_512(self, embedder):
        assert embedder.model.max_seq_length == 512


class TestEmbed:
    """Tests for the embed() method."""

    def test_embed_single_text(self, embedder):
        results = embedder.embed(["hello world"])
        assert len(results) == 1
        assert isinstance(results[0], list)
        assert all(isinstance(v, float) for v in results[0])

    def test_embed_multiple_texts(self, embedder):
        texts = ["first text", "second text", "third text"]
        results = embedder.embed(texts)
        assert len(results) == 3
        # Each embedding should be the same dimension
        dims = {len(r) for r in results}
        assert len(dims) == 1

    def test_embed_empty_list_returns_empty(self, embedder):
        results = embedder.embed([])
        assert results == []

    def test_embed_returns_consistent_dimensions(self, embedder):
        results = embedder.embed(["short", "a much longer sentence with many words"])
        assert len(results[0]) == len(results[1])

    def test_embed_similar_texts_have_high_similarity(self, embedder):
        """Semantically similar texts should produce similar embeddings."""
        results = embedder.embed([
            "the cat sat on the mat",
            "a cat was sitting on a mat",
        ])
        # Compute cosine similarity
        import math

        vec_a, vec_b = results[0], results[1]
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        mag_a = math.sqrt(sum(a * a for a in vec_a))
        mag_b = math.sqrt(sum(b * b for b in vec_b))
        similarity = dot / (mag_a * mag_b)
        assert similarity > 0.8

    def test_embed_long_text_truncated_without_error(self, embedder):
        """Text exceeding 512 tokens should be truncated, not raise an error."""
        long_text = "word " * 1000  # Way more than 512 tokens
        results = embedder.embed([long_text])
        assert len(results) == 1
        assert len(results[0]) > 0


class TestEmbedSingle:
    """Tests for the embed_single() method."""

    def test_embed_single_returns_vector(self, embedder):
        result = embedder.embed_single("hello world")
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_embed_single_empty_string_returns_empty_list(self, embedder):
        result = embedder.embed_single("")
        assert result == []

    def test_embed_single_matches_embed_batch(self, embedder):
        """embed_single should return the same result as embed with one item."""
        text = "architectural decision record"
        single_result = embedder.embed_single(text)
        batch_result = embedder.embed([text])
        assert single_result == batch_result[0]
