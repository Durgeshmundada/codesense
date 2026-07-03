"""Property-based tests for IntentClassifier.

Tests Property 24 from the design document using Hypothesis.

Validates: Requirements 11.1, 11.3
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from codesense.capabilities.ask import IntentClassifier, VALID_INTENTS


# Strategy: generate varied non-empty query strings.
# Combines printable text, words that may or may not match keywords,
# and random unicode to test robustness.
_query_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S", "Z"),
        blacklist_characters="\x00",
    ),
    min_size=1,
    max_size=200,
).filter(lambda s: s.strip() != "")


# Feature: codesense, Property 24: Query intent classification range
@settings(max_examples=100)
@given(query=_query_strategy)
def test_query_intent_classification_range(query: str) -> None:
    """For any non-empty natural language query, the intent classifier returns
    an intent from the valid set {explain, describe, tree, flow, diagram, trace,
    deps, related, risk, onboard} OR routes to the general reasoning loop
    (intent="explain" with low confidence).

    Test with GeminiService=None (keyword-only mode) to avoid LLM calls.
    Verify that classify() never returns an intent outside VALID_INTENTS
    (or None for non-empty queries).

    **Validates: Requirements 11.1, 11.3**
    """
    classifier = IntentClassifier(gemini_service=None, confidence_threshold=0.6)

    result = classifier.classify(query)

    # The result must be a dict with an "intent" key
    assert isinstance(result, dict), f"Expected dict result, got {type(result)}"
    assert "intent" in result, "Result must contain 'intent' key"

    intent = result["intent"]

    # Intent must be from the valid set — never None or an unknown value
    assert intent is not None, (
        f"Intent must not be None for non-empty query: {query!r}"
    )
    assert intent in VALID_INTENTS, (
        f"Intent '{intent}' not in VALID_INTENTS for query: {query!r}"
    )

    # Confidence must be a float in [0.0, 1.0]
    assert "confidence" in result, "Result must contain 'confidence' key"
    confidence = result["confidence"]
    assert isinstance(confidence, (int, float)), (
        f"Confidence must be numeric, got {type(confidence)}"
    )
    assert 0.0 <= confidence <= 1.0, (
        f"Confidence {confidence} out of range [0.0, 1.0] for query: {query!r}"
    )
