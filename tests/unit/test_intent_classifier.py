"""Unit tests for the IntentClassifier in codesense/capabilities/ask.py.

Tests keyword-based classification, LLM response parsing, parameter extraction,
empty query handling, and confidence threshold behavior.

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5
"""

import json
from unittest.mock import MagicMock

import pytest

from codesense.capabilities.ask import (
    AskHandler,
    IntentClassifier,
    VALID_INTENTS,
)
from codesense.models.output import CommandParams


class TestKeywordClassification:
    """Tests for keyword-based intent classification."""

    def setup_method(self):
        """Create a classifier with a mock gemini_service (won't be called for keyword matches)."""
        self.classifier = IntentClassifier(
            gemini_service=MagicMock(),
            confidence_threshold=0.6,
        )

    def test_explain_keywords(self):
        """Queries with 'why', 'explain', 'history' -> explain intent."""
        result = self.classifier.classify("why does this file exist?")
        assert result["intent"] == "explain"
        assert result["confidence"] >= 0.7
        assert result["error"] is None

    def test_describe_keywords(self):
        """Queries with 'what', 'describe', 'how does' -> describe intent."""
        result = self.classifier.classify("what does this module do?")
        assert result["intent"] == "describe"
        assert result["confidence"] >= 0.7

    def test_tree_keywords(self):
        """Queries with 'tree', 'structure', 'folders' -> tree intent."""
        result = self.classifier.classify("show me the folder structure")
        assert result["intent"] == "tree"
        assert result["confidence"] >= 0.7

    def test_flow_keywords(self):
        """Queries with 'flow', 'execution', 'sequence' -> flow intent."""
        result = self.classifier.classify("show execution flow of main")
        assert result["intent"] == "flow"
        assert result["confidence"] >= 0.7

    def test_diagram_keywords(self):
        """Queries with 'diagram', 'draw', 'visualize' -> diagram intent."""
        result = self.classifier.classify("draw a diagram of the classes")
        assert result["intent"] == "diagram"
        assert result["confidence"] >= 0.7

    def test_trace_keywords(self):
        """Queries with 'trace', 'when', 'timeline' -> trace intent."""
        result = self.classifier.classify("trace the timeline of changes")
        assert result["intent"] == "trace"
        assert result["confidence"] >= 0.7

    def test_deps_keywords(self):
        """Queries with 'depend', 'import', 'env', 'api' -> deps intent."""
        result = self.classifier.classify("what are the dependencies this imports?")
        assert result["intent"] == "deps"
        assert result["confidence"] >= 0.7

    def test_related_keywords(self):
        """Queries with 'related', 'affects', 'impact' -> related intent."""
        result = self.classifier.classify("what files are related to this?")
        assert result["intent"] == "related"
        assert result["confidence"] >= 0.7

    def test_risk_keywords(self):
        """Queries with 'risk', 'safe', 'touch', 'delete' -> risk intent."""
        result = self.classifier.classify("is it safe to delete this file?")
        assert result["intent"] == "risk"
        assert result["confidence"] >= 0.7

    def test_onboard_keywords(self):
        """Queries with 'onboard', 'guide', 'overview' -> onboard intent."""
        result = self.classifier.classify("give me an onboarding guide")
        assert result["intent"] == "onboard"
        assert result["confidence"] >= 0.7


class TestEmptyQueryHandling:
    """Tests for empty/uninterpretable query handling."""

    def setup_method(self):
        self.classifier = IntentClassifier(confidence_threshold=0.6)

    def test_empty_string(self):
        """Empty string returns error prompting rephrase."""
        result = self.classifier.classify("")
        assert result["intent"] is None
        assert result["error"] is not None
        assert "empty" in result["error"].lower() or "rephrase" in result["error"].lower()

    def test_whitespace_only(self):
        """Whitespace-only string returns error."""
        result = self.classifier.classify("   \t\n  ")
        assert result["intent"] is None
        assert result["error"] is not None

    def test_confidence_zero_for_empty(self):
        """Empty queries have confidence 0.0."""
        result = self.classifier.classify("")
        assert result["confidence"] == 0.0


class TestLLMClassification:
    """Tests for LLM-based classification and response parsing."""

    def setup_method(self):
        self.mock_gemini = MagicMock()
        self.classifier = IntentClassifier(
            gemini_service=self.mock_gemini,
            confidence_threshold=0.6,
        )

    def test_valid_llm_json_response(self):
        """Properly formatted LLM JSON response is parsed correctly."""
        llm_response = json.dumps({
            "intent": "flow",
            "confidence": 0.85,
            "params": {
                "file_path": "src/auth.py",
                "function_name": "authenticate",
                "line_number": None,
                "feature_name": None,
            }
        })
        self.mock_gemini.generate.return_value = llm_response

        # Use a query that doesn't match any keyword patterns
        result = self.classifier.classify("orchestrate the process pipeline")
        assert result["intent"] == "flow"
        assert result["confidence"] == 0.85
        assert result["error"] is None

    def test_llm_response_with_markdown_fences(self):
        """LLM response wrapped in markdown code fences is parsed."""
        llm_response = '```json\n{"intent": "deps", "confidence": 0.9, "params": {}}\n```'
        self.mock_gemini.generate.return_value = llm_response

        result = self.classifier.classify("list all external connections")
        assert result["intent"] == "deps"
        assert result["confidence"] == 0.9

    def test_llm_response_invalid_intent_falls_back(self):
        """Invalid intent in LLM response falls back to 'explain'."""
        llm_response = json.dumps({
            "intent": "invalid_intent",
            "confidence": 0.8,
            "params": {},
        })
        self.mock_gemini.generate.return_value = llm_response

        result = self.classifier.classify("something completely novel")
        assert result["intent"] == "explain"

    def test_llm_failure_falls_back_to_explain(self):
        """When LLM raises an exception, fallback to explain intent."""
        self.mock_gemini.generate.side_effect = RuntimeError("API unavailable")

        result = self.classifier.classify("completely ambiguous query here")
        assert result["intent"] == "explain"
        assert result["confidence"] <= 0.5
        assert result["error"] is None


class TestConfidenceThreshold:
    """Tests for confidence threshold and candidate presentation."""

    def setup_method(self):
        self.mock_gemini = MagicMock()
        self.classifier = IntentClassifier(
            gemini_service=self.mock_gemini,
            confidence_threshold=0.6,
        )

    def test_low_confidence_returns_candidates(self):
        """When confidence < threshold, candidates are returned."""
        llm_response = json.dumps({
            "intent": "describe",
            "confidence": 0.4,
            "params": {},
        })
        self.mock_gemini.generate.return_value = llm_response

        result = self.classifier.classify("something vague about the code")
        assert result["candidates"] is not None
        assert len(result["candidates"]) >= 1
        # Top candidate should be the classified intent
        assert result["candidates"][0]["intent"] == "describe"

    def test_high_confidence_no_candidates(self):
        """When confidence >= threshold, no candidates returned."""
        llm_response = json.dumps({
            "intent": "tree",
            "confidence": 0.9,
            "params": {},
        })
        self.mock_gemini.generate.return_value = llm_response

        result = self.classifier.classify("completely new query for proj layout")
        assert result["candidates"] is None


class TestParameterExtraction:
    """Tests for parameter extraction from natural language queries."""

    def setup_method(self):
        self.classifier = IntentClassifier(
            gemini_service=MagicMock(),
            confidence_threshold=0.6,
        )

    def test_extract_file_path(self):
        """File paths in queries are extracted."""
        result = self.classifier.classify("explain src/auth.py")
        params = result["params"]
        assert params.get("file_path") is not None
        assert "auth.py" in params["file_path"]

    def test_extract_line_number(self):
        """Line numbers in queries are extracted."""
        result = self.classifier.classify("trace changes at line 42")
        params = result["params"]
        assert params.get("line_number") == 42

    def test_extract_function_name(self):
        """Function names with 'function' prefix are extracted."""
        result = self.classifier.classify("explain function authenticate")
        params = result["params"]
        assert params.get("function_name") == "authenticate"

    def test_extract_function_with_parens(self):
        """Function names with parentheses are extracted."""
        result = self.classifier.classify("explain main()")
        params = result["params"]
        assert params.get("function_name") == "main"


class TestIntentAlwaysValid:
    """Tests that classified intent is always from the valid set or routes to general reasoning."""

    def setup_method(self):
        self.mock_gemini = MagicMock()
        self.classifier = IntentClassifier(
            gemini_service=self.mock_gemini,
            confidence_threshold=0.6,
        )

    def test_result_intent_in_valid_set_or_none(self):
        """Non-empty queries always produce an intent from VALID_INTENTS."""
        queries = [
            "why does auth exist?",
            "what is this?",
            "show tree",
            "random gibberish xyzzy",
            "do something weird",
        ]
        # For queries that don't match keywords, LLM fallback returns explain
        self.mock_gemini.generate.side_effect = RuntimeError("no key")

        for query in queries:
            result = self.classifier.classify(query)
            if result["intent"] is not None:  # None only for empty queries
                assert result["intent"] in VALID_INTENTS, (
                    f"Query '{query}' produced invalid intent: {result['intent']}"
                )


class TestAskHandler:
    """Tests for the AskHandler capability."""

    def test_ask_handler_empty_query(self):
        """AskHandler with empty query returns error output."""
        handler = AskHandler(gemini_service=MagicMock())
        params = CommandParams(query="", mock=False)
        output = handler.run(params)
        assert "Error" in output.content
        assert output.confidence == 0.0

    def test_ask_handler_valid_query(self):
        """AskHandler with valid keyword query returns classification."""
        handler = AskHandler(gemini_service=MagicMock())
        params = CommandParams(query="why does auth.py exist?", mock=False)
        output = handler.run(params)
        assert "explain" in output.content
        assert output.confidence >= 0.7

    def test_ask_handler_demo_mode(self):
        """AskHandler respects demo mode flag."""
        handler = AskHandler(gemini_service=MagicMock())
        params = CommandParams(query="show tree structure", mock=True)
        output = handler.run(params)
        assert output.is_demo_mode is True
