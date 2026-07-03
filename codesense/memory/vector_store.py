"""VectorStore with Chroma backend for Decision Memory.

Provides persistent storage and cosine-similarity retrieval of embedded
DecisionUnit chunks using ChromaDB as the vector database.
"""

import json

import chromadb

from codesense.models.memory import DecisionUnit, RetrievalResult


class VectorStore:
    """Chroma-backed vector store for Decision Memory.

    Stores and retrieves embedded DecisionUnit chunks with cosine similarity
    search. Uses Chroma's built-in cosine distance metric.

    Args:
        persist_directory: Path to the directory for persisting Chroma data.
            Defaults to "./chroma_db".
        collection_name: Name of the Chroma collection to use.
            Defaults to "decision_memory".
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "decision_memory",
    ) -> None:
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def store(self, units: list[DecisionUnit], embeddings: list[list[float]]) -> None:
        """Persist embedded DecisionUnits to the Chroma collection.

        Stores each unit's content, id, and metadata (source_document,
        section_heading, ingestion_timestamp, order_index,
        has_structural_boundaries, referenced_components).

        The embeddings are pre-computed (passed in from HuggingFaceEmbedder).

        Args:
            units: List of DecisionUnit objects to store.
            embeddings: Corresponding embedding vectors for each unit.

        Raises:
            ValueError: If the number of units and embeddings don't match.
        """
        if not units:
            return

        if len(units) != len(embeddings):
            raise ValueError(
                f"Number of units ({len(units)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )

        ids = [unit.id for unit in units]
        documents = [unit.content for unit in units]
        metadatas = [
            {
                "source_document": unit.source_document,
                "section_heading": unit.section_heading,
                "ingestion_timestamp": unit.ingestion_timestamp,
                "order_index": unit.order_index,
                "has_structural_boundaries": unit.has_structural_boundaries,
                "referenced_components": json.dumps(unit.referenced_components),
            }
            for unit in units
        ]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        embedding: list[float],
        top_k: int = 5,
        min_similarity: float = 0.7,
    ) -> list[RetrievalResult]:
        """Search for similar DecisionUnits by cosine similarity.

        Queries the vector store by cosine similarity. Only returns results
        with similarity >= min_similarity threshold. Results are sorted
        descending by similarity_score.

        Each result includes the DecisionUnit reconstructed from stored data
        plus metadata.

        Args:
            embedding: The query embedding vector.
            top_k: Maximum number of results to return. Defaults to 5.
            min_similarity: Minimum cosine similarity threshold (0.0 to 1.0).
                Only results with similarity >= this value are returned.
                Defaults to 0.7.

        Returns:
            A list of RetrievalResult objects sorted descending by similarity.
            Returns an empty list when no results meet the threshold.
        """
        # If collection is empty, return early
        if self._collection.count() == 0:
            return []

        # Query Chroma for top_k results (we filter by threshold afterward)
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        # Chroma returns lists-of-lists; extract the first (and only) query's results
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieval_results: list[RetrievalResult] = []

        for i, doc_id in enumerate(ids):
            # Chroma cosine distance: similarity = 1 - distance
            similarity = 1.0 - distances[i]

            if similarity < min_similarity:
                continue

            metadata = metadatas[i]

            # Deserialize referenced_components from JSON string
            referenced_components_raw = metadata.get("referenced_components", "[]")
            if isinstance(referenced_components_raw, str):
                try:
                    referenced_components = json.loads(referenced_components_raw)
                except (json.JSONDecodeError, ValueError):
                    # Malformed/hand-edited metadata should not abort the query.
                    referenced_components = []
            else:
                referenced_components = referenced_components_raw or []

            decision_unit = DecisionUnit(
                id=doc_id,
                content=documents[i],
                section_heading=metadata.get("section_heading", ""),
                source_document=metadata.get("source_document", ""),
                ingestion_timestamp=metadata.get("ingestion_timestamp", ""),
                order_index=metadata.get("order_index", 0),
                has_structural_boundaries=metadata.get(
                    "has_structural_boundaries", True
                ),
                referenced_components=referenced_components,
            )

            result = RetrievalResult(
                decision_unit=decision_unit,
                similarity_score=similarity,
                metadata={
                    "source_document": metadata.get("source_document", ""),
                    "section_heading": metadata.get("section_heading", ""),
                    "ingestion_timestamp": metadata.get("ingestion_timestamp", ""),
                },
            )
            retrieval_results.append(result)

        # Sort descending by similarity score
        retrieval_results.sort(key=lambda r: r.similarity_score, reverse=True)

        return retrieval_results

    def delete_collection(self) -> None:
        """Delete the entire collection for cleanup/testing.

        Removes the collection and all stored data. After calling this,
        the VectorStore instance should not be used further without
        re-initialization.
        """
        self._client.delete_collection(name=self._collection.name)

    def delete_by_document(self, source_document: str) -> None:
        """Delete all chunks from a specific source document.

        Used for atomic ingestion rollback — removes all DecisionUnits
        that were stored from the given document.

        Args:
            source_document: The source document filename to delete chunks for.
        """
        self._collection.delete(
            where={"source_document": source_document},
        )
