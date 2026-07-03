"""Unit tests for parameter extraction from natural language queries.

Tests the IntentClassifier.extract_params method which extracts file paths,
function names, line numbers, and feature names from natural language queries.

Requirements: 11.2
"""

import pytest

from codesense.capabilities.ask import IntentClassifier, VALID_INTENTS
from codesense.models.output import CommandParams


@pytest.fixture
def classifier():
    """Create an IntentClassifier with no LLM (keyword/regex-only mode)."""
    return IntentClassifier(gemini_service=None)


class TestFilePathExtraction:
    """Tests for extracting file paths from natural language queries."""

    def test_unix_style_path(self, classifier):
        params = classifier.extract_params("explain src/auth.py", "explain")
        assert params.path == "src/auth.py"

    def test_nested_unix_path(self, classifier):
        params = classifier.extract_params(
            "why does codesense/capabilities/ask.py exist?", "explain"
        )
        assert params.path == "codesense/capabilities/ask.py"

    def test_relative_path_with_dot(self, classifier):
        params = classifier.extract_params(
            "describe ./utils/helpers.ts", "describe"
        )
        assert params.path == "./utils/helpers.ts"

    def test_quoted_path_single(self, classifier):
        params = classifier.extract_params(
            "what does 'src/models/user.py' do?", "describe"
        )
        assert params.path == "src/models/user.py"

    def test_quoted_path_double(self, classifier):
        params = classifier.extract_params(
            'explain "lib/auth/token.js"', "explain"
        )
        assert params.path == "lib/auth/token.js"

    def test_backtick_path(self, classifier):
        params = classifier.extract_params(
            "what is `app/controllers/home.rb`?", "describe"
        )
        assert params.path == "app/controllers/home.rb"

    def test_path_with_dashes(self, classifier):
        params = classifier.extract_params(
            "show flow for my-app/src/main.ts", "flow"
        )
        assert params.path == "my-app/src/main.ts"

    def test_no_path_in_query(self, classifier):
        params = classifier.extract_params(
            "what is the overall project layout?", "tree"
        )
        assert params.path is None

    def test_path_with_multiple_extensions_picks_first(self, classifier):
        params = classifier.extract_params(
            "compare config.yaml and settings.json", "describe"
        )
        # Should pick a valid path
        assert params.path in ("config.yaml", "settings.json")


class TestFunctionNameExtraction:
    """Tests for extracting function names from natural language queries."""

    def test_after_def_keyword(self, classifier):
        params = classifier.extract_params(
            "explain def process_payment", "explain"
        )
        assert params.function_name == "process_payment"

    def test_after_function_keyword(self, classifier):
        params = classifier.extract_params(
            "describe function calculateTotal", "describe"
        )
        assert params.function_name == "calculateTotal"

    def test_after_method_keyword(self, classifier):
        params = classifier.extract_params(
            "what does method handle_request do?", "describe"
        )
        assert params.function_name == "handle_request"

    def test_the_x_function_pattern(self, classifier):
        params = classifier.extract_params(
            "explain the authenticate function", "explain"
        )
        assert params.function_name == "authenticate"

    def test_backtick_function_name(self, classifier):
        params = classifier.extract_params(
            "how does `get_user_by_id` work?", "describe"
        )
        assert params.function_name == "get_user_by_id"

    def test_function_with_parens(self, classifier):
        params = classifier.extract_params(
            "what does process_data() do?", "describe"
        )
        assert params.function_name == "process_data"

    def test_no_function_in_query(self, classifier):
        params = classifier.extract_params(
            "show me the project tree", "tree"
        )
        assert params.function_name is None


class TestLineNumberExtraction:
    """Tests for extracting line numbers from natural language queries."""

    def test_line_number_pattern(self, classifier):
        params = classifier.extract_params(
            "what happens at line 44?", "explain"
        )
        assert params.line_number == 44

    def test_line_with_capital_L(self, classifier):
        params = classifier.extract_params(
            "explain L128", "explain"
        )
        assert params.line_number == 128

    def test_colon_line_number(self, classifier):
        params = classifier.extract_params(
            "trace src/main.py:25", "trace"
        )
        assert params.line_number == 25

    def test_line_number_word(self, classifier):
        params = classifier.extract_params(
            "show me line number 99", "trace"
        )
        assert params.line_number == 99

    def test_no_line_number(self, classifier):
        params = classifier.extract_params(
            "describe the auth module", "describe"
        )
        assert params.line_number is None

    def test_unreasonable_line_number_rejected(self, classifier):
        """Line numbers above 100000 should be rejected."""
        params = classifier.extract_params(
            "go to line 999999999", "trace"
        )
        assert params.line_number is None


class TestFeatureNameExtraction:
    """Tests for extracting feature names from natural language queries."""

    def test_feature_keyword_with_quotes(self, classifier):
        params = classifier.extract_params(
            'diagram the feature "user authentication"', "diagram"
        )
        assert params.output == "user authentication"

    def test_feature_keyword_without_quotes(self, classifier):
        params = classifier.extract_params(
            "diagram feature login flow", "diagram"
        )
        assert params.output == "login flow"

    def test_no_feature_in_query(self, classifier):
        params = classifier.extract_params(
            "explain src/auth.py", "explain"
        )
        # path was extracted, so feature should not be in output
        assert params.output is None


class TestCombinedExtraction:
    """Tests for extracting multiple parameters from a single query."""

    def test_path_and_line_number(self, classifier):
        params = classifier.extract_params(
            "trace src/auth.py at line 42", "trace"
        )
        assert params.path == "src/auth.py"
        assert params.line_number == 42

    def test_path_and_function(self, classifier):
        params = classifier.extract_params(
            "explain function handle_login in src/auth.py", "explain"
        )
        assert params.path == "src/auth.py"
        assert params.function_name == "handle_login"

    def test_query_always_preserved(self, classifier):
        query = "why does src/auth.py exist?"
        params = classifier.extract_params(query, "explain")
        assert params.query == query

    def test_returns_command_params(self, classifier):
        """extract_params always returns a CommandParams instance."""
        params = classifier.extract_params("explain src/main.py", "explain")
        assert isinstance(params, CommandParams)


class TestIntentClassification:
    """Tests for the full classify method with keyword fallback."""

    def test_explain_intent(self, classifier):
        result = classifier.classify("why does src/auth.py exist?")
        assert result["intent"] == "explain"
        assert result["confidence"] > 0

    def test_describe_intent(self, classifier):
        result = classifier.classify("what does the auth module do?")
        assert result["intent"] == "describe"
        assert result["confidence"] > 0

    def test_tree_intent(self, classifier):
        result = classifier.classify("show me the project structure")
        assert result["intent"] == "tree"

    def test_deps_intent(self, classifier):
        result = classifier.classify("what dependencies does this import?")
        assert result["intent"] == "deps"

    def test_risk_intent(self, classifier):
        result = classifier.classify("how risky is this file to change?")
        assert result["intent"] == "risk"

    def test_general_routing_for_ambiguous(self, classifier):
        """Very generic query should fallback to explain with lower confidence."""
        result = classifier.classify("tell me about it")
        # Should fall back to explain with low confidence
        assert result["intent"] == "explain"
        assert result["confidence"] <= 0.5

    def test_empty_query_raises(self, classifier):
        with pytest.raises(ValueError, match="empty"):
            classifier.classify("")

    def test_whitespace_only_raises(self, classifier):
        with pytest.raises(ValueError, match="empty"):
            classifier.classify("   ")

    def test_valid_intent_set(self, classifier):
        """All returned intents must be from the valid set."""
        queries = [
            "why does this exist?",
            "describe the auth module",
            "show project structure",
            "trace the history of main.py",
            "show dependencies of this module",
        ]
        for q in queries:
            result = classifier.classify(q)
            assert result["intent"] in VALID_INTENTS

    def test_params_included_in_result(self, classifier):
        """classify returns extracted params alongside the intent."""
        result = classifier.classify("why does src/auth.py exist?")
        assert "params" in result
        params = result["params"]
        assert isinstance(params, CommandParams)
        assert params.path == "src/auth.py"
