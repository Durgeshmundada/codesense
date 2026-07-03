"""Property-based tests for Decision Memory retrieval invariants.

Tests Properties 8 and 9 from the design document using Hypothesis
to verify that VectorStore retrieval satisfies ordering, threshold,
count, and metadata invariants, and that ingestion round-trips correctly.

**Validates: Requirements 3.3, 3.5, 3.6**
"""

import tempfile
import uuid

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from codesense.memory.embedder import HuggingFaceEmbedder
from codesense.memory.vector_store import VectorStore
from codesense.models.memory import DecisionUnit


# --- Strategies ---

section_heading_st = st.text(min_size=1, max_size=50)
source_document_st = st.text(min_size=1, max_size=50)
timestamp_st = st.from_regex(r"2024-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", fullmatch=True)

top_k_st = st.integers(min_value=1, max_value=20)
min_similarity_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


def make_decision_unit(
    content: str,
    section_heading: str = "Test Section",
    source_document: str = "test_adr.md",
    ingestion_timestamp: str = "2024-01-01T00:00:00",
) -> DecisionUnit:
    """Create a DecisionUnit with a unique ID."""
    return DecisionUnit(
        id=str(uuid.uuid4()),
        content=content,
        section_heading=section_heading,
        source_document=source_document,
        ingestion_timestamp=ingestion_timestamp,
        referenced_components=["component_a"],
        has_structural_boundaries=True,
        order_index=0,
    )


# Shared embedder instance (expensive to load, reuse across tests)
_embedder = None


def get_embedder() -> HuggingFaceEmbedder:
    """Get or create a shared HuggingFaceEmbedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbedder(model_name="all-MiniLM-L6-v2")
    return _embedder


# Use a shared temp directory that persists for the test session to avoid
# Windows file-lock issues with ChromaDB's PersistentClient.
_test_base_dir = tempfile.mkdtemp(prefix="codesense_pbt_")
_collection_counter = 0


def get_unique_collection_name() -> str:
    """Generate a unique collection name for each test invocation."""
    global _collection_counter
    _collection_counter += 1
    return f"test_coll_{_collection_counter}_{uuid.uuid4().hex[:8]}"


def create_test_store() -> VectorStore:
    """Create a VectorStore with a unique collection in the shared temp dir."""
    return VectorStore(
        persist_directory=_test_base_dir,
        collection_name=get_unique_collection_name(),
    )


# --- Property 8: Decision Memory retrieval invariants ---
# For any semantic search query against the Decision Memory, results satisfy:
# (a) count <= top_k
# (b) all similarity >= min_similarity threshold
# (c) sorted descending by similarity_score
# (d) each includes required metadata fields (source_document, section_heading, ingestion_timestamp)


@settings(max_examples=20, deadline=None)
@given(
    top_k=top_k_st,
    min_similarity=min_similarity_st,
)
def test_retrieval_count_never_exceeds_top_k(top_k: int, min_similarity: float) -> None:
    """Property 8a: Result count never exceeds top_k.

    **Validates: Requirements 3.3**
    """
    embedder = get_embedder()
    store = create_test_store()

    # Insert several decision units
    units = [
        make_decision_unit(f"Decision about authentication mechanism {i}")
        for i in range(10)
    ]
    embeddings = embedder.embed([u.content for u in units])
    store.store(units, embeddings)

    # Query with a related text
    query_embedding = embedder.embed_single("authentication decision")
    results = store.query(
        embedding=query_embedding, top_k=top_k, min_similarity=min_similarity
    )

    assert len(results) <= top_k, (
        f"Got {len(results)} results but top_k={top_k}"
    )


@settings(max_examples=20, deadline=None)
@given(
    min_similarity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_retrieval_all_results_above_min_similarity(min_similarity: float) -> None:
    """Property 8b: All results have similarity_score >= min_similarity.

    **Validates: Requirements 3.3**
    """
    embedder = get_embedder()
    store = create_test_store()

    units = [
        make_decision_unit("We chose PostgreSQL for data persistence"),
        make_decision_unit("The frontend uses React with TypeScript"),
        make_decision_unit("Authentication uses JWT tokens with refresh"),
        make_decision_unit("Logging is handled by structured JSON output"),
        make_decision_unit("Caching layer uses Redis for session state"),
    ]
    embeddings = embedder.embed([u.content for u in units])
    store.store(units, embeddings)

    query_embedding = embedder.embed_single("database choice")
    results = store.query(
        embedding=query_embedding, top_k=5, min_similarity=min_similarity
    )

    for result in results:
        assert result.similarity_score >= min_similarity, (
            f"Result similarity {result.similarity_score} is below "
            f"min_similarity threshold {min_similarity}"
        )


@settings(max_examples=20, deadline=None)
@given(
    top_k=top_k_st,
)
def test_retrieval_results_sorted_descending(top_k: int) -> None:
    """Property 8c: Results are sorted in descending order by similarity_score.

    **Validates: Requirements 3.3**
    """
    embedder = get_embedder()
    store = create_test_store()

    units = [
        make_decision_unit("API gateway routes requests to microservices"),
        make_decision_unit("Database migrations use Alembic with SQLAlchemy"),
        make_decision_unit("Message queue uses RabbitMQ for async tasks"),
        make_decision_unit("CI pipeline runs tests then deploys to staging"),
        make_decision_unit("Monitoring uses Prometheus with Grafana dashboards"),
        make_decision_unit("Error tracking integrates with Sentry for alerts"),
        make_decision_unit("Rate limiting uses token bucket algorithm"),
    ]
    embeddings = embedder.embed([u.content for u in units])
    store.store(units, embeddings)

    query_embedding = embedder.embed_single("message processing")
    results = store.query(
        embedding=query_embedding, top_k=top_k, min_similarity=0.0
    )

    for i in range(len(results) - 1):
        assert results[i].similarity_score >= results[i + 1].similarity_score, (
            f"Results not sorted descending: index {i} has score "
            f"{results[i].similarity_score} but index {i+1} has "
            f"{results[i+1].similarity_score}"
        )


@settings(max_examples=20, deadline=None)
@given(
    top_k=top_k_st,
)
def test_retrieval_results_include_required_metadata(top_k: int) -> None:
    """Property 8d: Each result includes source_document, section_heading, ingestion_timestamp.

    **Validates: Requirements 3.5**
    """
    embedder = get_embedder()
    store = create_test_store()

    units = [
        make_decision_unit(
            content="We use event sourcing for audit trail",
            section_heading="Architecture Decision",
            source_document="adr-001.md",
            ingestion_timestamp="2024-03-15T10:30:00",
        ),
        make_decision_unit(
            content="GraphQL chosen over REST for flexibility",
            section_heading="API Design",
            source_document="adr-002.md",
            ingestion_timestamp="2024-04-01T14:00:00",
        ),
    ]
    embeddings = embedder.embed([u.content for u in units])
    store.store(units, embeddings)

    query_embedding = embedder.embed_single("event sourcing")
    results = store.query(
        embedding=query_embedding, top_k=top_k, min_similarity=0.0
    )

    required_keys = {"source_document", "section_heading", "ingestion_timestamp"}
    for result in results:
        assert required_keys.issubset(result.metadata.keys()), (
            f"Result metadata missing required keys. "
            f"Expected {required_keys}, got {set(result.metadata.keys())}"
        )
        # Values should be non-empty strings
        for key in required_keys:
            assert isinstance(result.metadata[key], str) and result.metadata[key], (
                f"Metadata field '{key}' is empty or not a string: "
                f"{result.metadata[key]!r}"
            )


# --- Property 9: Decision Memory ingestion round-trip ---
# For any DecisionUnit that is embedded and stored, querying with the exact
# content as search text returns that unit with similarity >= 0.9.


@settings(max_examples=20, deadline=None)
@given(
    content=st.sampled_from([
        "We decided to use microservices architecture for scalability",
        "Authentication is handled via OAuth2 with PKCE flow",
        "The database schema uses soft deletes for audit compliance",
        "Deployment uses Kubernetes with Helm charts for orchestration",
        "Logging follows structured JSON format with correlation IDs",
        "Cache invalidation uses event-driven pub/sub pattern",
        "API versioning uses URL path prefix strategy",
        "Feature flags managed through LaunchDarkly integration",
        "Error handling follows the Result monad pattern",
        "Testing strategy uses the testing pyramid approach",
    ]),
)
def test_ingestion_round_trip_exact_content(content: str) -> None:
    """Property 9: Querying with exact DecisionUnit content returns it with similarity >= 0.9.

    **Validates: Requirements 3.6**
    """
    embedder = get_embedder()
    store = create_test_store()

    unit = make_decision_unit(content=content)
    embedding = embedder.embed([content])
    store.store([unit], embedding)

    # Query with the exact same content
    query_embedding = embedder.embed_single(content)
    results = store.query(
        embedding=query_embedding, top_k=5, min_similarity=0.0
    )

    # The stored unit must appear in results
    assert len(results) >= 1, (
        f"Expected at least 1 result for exact content query, got 0"
    )

    # Find the matching unit
    matching = [r for r in results if r.decision_unit.id == unit.id]
    assert len(matching) == 1, (
        f"Expected to find the stored unit in results, "
        f"but found {len(matching)} matches"
    )

    # Similarity must be >= 0.9 for exact content
    assert matching[0].similarity_score >= 0.9, (
        f"Expected similarity >= 0.9 for exact content round-trip, "
        f"got {matching[0].similarity_score}"
    )
