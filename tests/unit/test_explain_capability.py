"""Unit tests for the ExplainHandler capability handler."""

from unittest.mock import MagicMock, patch

import pytest

from codesense.capabilities.explain import ExplainHandler, REPORT_TITLE
from codesense.models.output import CommandOutput, CommandParams
from codesense.models.state import (
    AgentState,
    Conflict,
    ConflictSource,
    NodeType,
    SynthesisResult,
)

# Patch target: the module where ReasoningGraph is imported inside ExplainHandler.run()
GRAPH_PATCH_TARGET = "codesense.agent.graph.ReasoningGraph"


@pytest.fixture
def mock_gemini_service():
    """Create a mock GeminiService."""
    return MagicMock()


@pytest.fixture
def mock_vector_store():
    """Create a mock VectorStore."""
    return MagicMock()


@pytest.fixture
def mock_embedder():
    """Create a mock HuggingFaceEmbedder."""
    return MagicMock()


def _make_final_state(
    answer: str = "This code exists for authentication.",
    confidence: float = 0.85,
    supporting_evidence: list[str] | None = None,
    conflicts: list[Conflict] | None = None,
    is_incomplete: bool = False,
    synthesis: SynthesisResult | None = "auto",
) -> AgentState:
    """Helper to construct a final AgentState with synthesis."""
    if supporting_evidence is None:
        supporting_evidence = ["commit_abc123", "issue_42"]
    if conflicts is None:
        conflicts = []

    if synthesis == "auto":
        synthesis = SynthesisResult(
            answer=answer,
            confidence=confidence,
            supporting_evidence=supporting_evidence,
            conflicts=conflicts,
            reasoning_path=[NodeType.EXPLORE, NodeType.HYPOTHESIZE, NodeType.VERIFY, NodeType.SYNTHESIZE],
            is_incomplete=is_incomplete,
        )

    return AgentState(
        query="Why does this code exist: src/auth.py",
        code_path="src/auth.py",
        loop_counter=1,
        remaining_iterations=2,
        evidence=[],
        hypotheses=[],
        confidence_score=confidence,
        conflicts=conflicts,
        synthesis=synthesis,
        current_node=NodeType.SYNTHESIZE,
        is_incomplete=is_incomplete,
    )


class TestExplainHandler:
    """Tests for ExplainHandler.run()."""

    @patch(GRAPH_PATCH_TARGET)
    def test_successful_explanation(self, mock_graph_cls, mock_gemini_service):
        """Test that a successful reasoning run produces correct CommandOutput."""
        final_state = _make_final_state()
        mock_graph_instance = MagicMock()
        mock_graph_instance.run.return_value = final_state
        mock_graph_cls.return_value = mock_graph_instance

        handler = ExplainHandler(gemini_service=mock_gemini_service)
        params = CommandParams(path="src/auth.py", mock=False)
        result = handler.run(params)

        assert isinstance(result, CommandOutput)
        assert result.title == REPORT_TITLE
        assert "This code exists for authentication." in result.content
        assert result.confidence == 0.85
        assert result.conflicts == []
        assert result.is_demo_mode is False
        assert result.code_snippets == []
        assert result.tables == []

    @patch(GRAPH_PATCH_TARGET)
    def test_output_includes_source_citations(self, mock_graph_cls, mock_gemini_service):
        """Test that supporting evidence appears as source citations in content."""
        final_state = _make_final_state(
            supporting_evidence=["commit_abc123", "issue_42", "pr_comment_7"]
        )
        mock_graph_instance = MagicMock()
        mock_graph_instance.run.return_value = final_state
        mock_graph_cls.return_value = mock_graph_instance

        handler = ExplainHandler(gemini_service=mock_gemini_service)
        params = CommandParams(path="src/auth.py", mock=False)
        result = handler.run(params)

        assert "Sources:" in result.content
        assert "commit_abc123" in result.content
        assert "issue_42" in result.content
        assert "pr_comment_7" in result.content

    @patch(GRAPH_PATCH_TARGET)
    def test_output_includes_conflicts(self, mock_graph_cls, mock_gemini_service):
        """Test that conflicts from synthesis are passed through to CommandOutput."""
        conflicts = [
            Conflict(
                id="c1",
                sources=[
                    ConflictSource(source_id="commit_1", claim="Added for security"),
                    ConflictSource(source_id="issue_5", claim="Added for performance"),
                ],
                description="Conflicting reasons for code introduction",
            )
        ]
        final_state = _make_final_state(conflicts=conflicts)
        mock_graph_instance = MagicMock()
        mock_graph_instance.run.return_value = final_state
        mock_graph_cls.return_value = mock_graph_instance

        handler = ExplainHandler(gemini_service=mock_gemini_service)
        params = CommandParams(path="src/auth.py", mock=False)
        result = handler.run(params)

        assert len(result.conflicts) == 1
        assert result.conflicts[0].id == "c1"
        assert len(result.conflicts[0].sources) == 2

    @patch(GRAPH_PATCH_TARGET)
    def test_demo_mode_via_params(self, mock_graph_cls, mock_gemini_service):
        """Test that is_demo_mode is True when params.mock is set."""
        final_state = _make_final_state()
        mock_graph_instance = MagicMock()
        mock_graph_instance.run.return_value = final_state
        mock_graph_cls.return_value = mock_graph_instance

        handler = ExplainHandler(gemini_service=mock_gemini_service)
        params = CommandParams(path="src/auth.py", mock=True)
        result = handler.run(params)

        assert result.is_demo_mode is True

    @patch(GRAPH_PATCH_TARGET)
    def test_no_synthesis_produces_fallback(self, mock_graph_cls, mock_gemini_service):
        """Test that when synthesis is None, a fallback output is returned."""
        final_state = _make_final_state(synthesis=None)
        final_state.synthesis = None
        mock_graph_instance = MagicMock()
        mock_graph_instance.run.return_value = final_state
        mock_graph_cls.return_value = mock_graph_instance

        handler = ExplainHandler(gemini_service=mock_gemini_service)
        params = CommandParams(path="src/auth.py", mock=False)
        result = handler.run(params)

        assert result.title == REPORT_TITLE
        assert "Unable to produce an explanation" in result.content
        assert result.confidence == 0.0
        assert result.conflicts == []
        assert result.code_snippets == []
        assert result.tables == []

    @patch(GRAPH_PATCH_TARGET)
    def test_incomplete_reasoning_note(self, mock_graph_cls, mock_gemini_service):
        """Test that incomplete reasoning adds a warning note to content."""
        final_state = _make_final_state(is_incomplete=True)
        mock_graph_instance = MagicMock()
        mock_graph_instance.run.return_value = final_state
        mock_graph_cls.return_value = mock_graph_instance

        handler = ExplainHandler(gemini_service=mock_gemini_service)
        params = CommandParams(path="src/auth.py", mock=False)
        result = handler.run(params)

        assert "Reasoning was incomplete" in result.content

    @patch(GRAPH_PATCH_TARGET)
    def test_custom_query_from_params(self, mock_graph_cls, mock_gemini_service):
        """Test that params.query overrides the default query construction."""
        final_state = _make_final_state()
        mock_graph_instance = MagicMock()
        mock_graph_instance.run.return_value = final_state
        mock_graph_cls.return_value = mock_graph_instance

        handler = ExplainHandler(gemini_service=mock_gemini_service)
        params = CommandParams(path="src/auth.py", query="Why was the auth module refactored?", mock=False)
        result = handler.run(params)

        # Verify the graph was called with the custom query
        mock_graph_instance.run.assert_called_once_with(
            query="Why was the auth module refactored?",
            code_path="src/auth.py",
            mock=False,
        )

    @patch(GRAPH_PATCH_TARGET)
    def test_confidence_score_passthrough(self, mock_graph_cls, mock_gemini_service):
        """Test that confidence score from synthesis is passed to output."""
        final_state = _make_final_state(confidence=0.42)
        mock_graph_instance = MagicMock()
        mock_graph_instance.run.return_value = final_state
        mock_graph_cls.return_value = mock_graph_instance

        handler = ExplainHandler(gemini_service=mock_gemini_service)
        params = CommandParams(path="src/auth.py", mock=False)
        result = handler.run(params)

        assert result.confidence == 0.42

    @patch(GRAPH_PATCH_TARGET)
    def test_query_includes_function_name(self, mock_graph_cls, mock_gemini_service):
        """Test that function_name is included in the query to the reasoning loop."""
        final_state = _make_final_state()
        mock_graph_instance = MagicMock()
        mock_graph_instance.run.return_value = final_state
        mock_graph_cls.return_value = mock_graph_instance

        handler = ExplainHandler(gemini_service=mock_gemini_service)
        params = CommandParams(path="src/auth.py", function_name="authenticate", mock=False)
        result = handler.run(params)

        call_args = mock_graph_instance.run.call_args
        query = call_args.kwargs.get("query", "")
        assert "authenticate" in query

    @patch(GRAPH_PATCH_TARGET)
    def test_query_includes_line_number(self, mock_graph_cls, mock_gemini_service):
        """Test that line_number is included in the query to the reasoning loop."""
        final_state = _make_final_state()
        mock_graph_instance = MagicMock()
        mock_graph_instance.run.return_value = final_state
        mock_graph_cls.return_value = mock_graph_instance

        handler = ExplainHandler(gemini_service=mock_gemini_service)
        params = CommandParams(path="src/auth.py", line_number=42, mock=False)
        result = handler.run(params)

        call_args = mock_graph_instance.run.call_args
        query = call_args.kwargs.get("query", "")
        assert "42" in query

    @patch(GRAPH_PATCH_TARGET)
    def test_query_includes_function_and_line(self, mock_graph_cls, mock_gemini_service):
        """Test that both function_name and line_number are included in the query."""
        final_state = _make_final_state()
        mock_graph_instance = MagicMock()
        mock_graph_instance.run.return_value = final_state
        mock_graph_cls.return_value = mock_graph_instance

        handler = ExplainHandler(gemini_service=mock_gemini_service)
        params = CommandParams(
            path="src/auth.py", function_name="login", line_number=15, mock=False
        )
        result = handler.run(params)

        call_args = mock_graph_instance.run.call_args
        query = call_args.kwargs.get("query", "")
        assert "login" in query
        assert "15" in query

    @patch(GRAPH_PATCH_TARGET)
    def test_error_handling_when_reasoning_fails(self, mock_graph_cls, mock_gemini_service):
        """Test that errors from the reasoning loop produce error CommandOutput."""
        mock_graph_cls.side_effect = RuntimeError("Graph construction failed")

        handler = ExplainHandler(gemini_service=mock_gemini_service)
        params = CommandParams(path="src/auth.py", mock=False)
        result = handler.run(params)

        assert result.title == REPORT_TITLE
        assert "Error" in result.content
        assert result.confidence == 0.0
        assert result.is_demo_mode is False
        assert result.code_snippets == []
        assert result.tables == []

    @patch(GRAPH_PATCH_TARGET)
    def test_error_handling_when_graph_run_fails(self, mock_graph_cls, mock_gemini_service):
        """Test that runtime errors from graph.run() are handled gracefully."""
        mock_graph_instance = MagicMock()
        mock_graph_instance.run.side_effect = RuntimeError("LLM timeout")
        mock_graph_cls.return_value = mock_graph_instance

        handler = ExplainHandler(gemini_service=mock_gemini_service)
        params = CommandParams(path="src/auth.py", mock=False)
        result = handler.run(params)

        assert result.title == REPORT_TITLE
        assert "Error" in result.content
        assert "LLM timeout" in result.content
        assert result.confidence == 0.0
        assert result.code_snippets == []
        assert result.tables == []

    def test_default_dependencies_no_keys_produces_error(self):
        """Test that missing API keys produce an error output gracefully."""
        handler = ExplainHandler()  # No dependencies provided

        with patch.dict("os.environ", {"GEMINI_API_KEYS": "", "GEMINI_API_KEY": ""}, clear=False):
            params = CommandParams(path="src/auth.py", mock=False)
            result = handler.run(params)

        assert result.title == REPORT_TITLE
        assert "Error" in result.content
        assert result.confidence == 0.0
