"""HuggingFace sentence-transformer embedder for Decision Memory.

Embeds text into dense vectors using a configurable sentence-transformers model.
Default model: "all-MiniLM-L6-v2" (free, no API key required).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class HuggingFaceEmbedder:
    """Embeds text using HuggingFace sentence-transformers.

    The model is loaded lazily on first use to avoid startup cost when
    the embedder is instantiated but not immediately needed.

    Args:
        model_name: Name of the sentence-transformers model to use.
            Defaults to "all-MiniLM-L6-v2".
    """

    MAX_INPUT_LENGTH = 512  # Maximum tokens per text input

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> "SentenceTransformer":
        """Lazily load and return the SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            # Set max sequence length to enforce the 512 token limit
            self._model.max_seq_length = self.MAX_INPUT_LENGTH
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text strings into vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            A list of embedding vectors (list of floats), one per input text.
            Returns an empty list if the input list is empty.
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [embedding.tolist() for embedding in embeddings]

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text string into a vector.

        Convenience wrapper around embed() for single inputs.

        Args:
            text: A single text string to embed.

        Returns:
            The embedding vector as a list of floats.
            Returns an empty list if the input text is empty.
        """
        if not text:
            return []

        results = self.embed([text])
        return results[0]
