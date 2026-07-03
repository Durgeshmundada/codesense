"""Unit tests for SynthesizeNode."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock out the heavy imports before importing nodes module
sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("chromadb", MagicMock())
sys.modules.setdefault("chromadb.api", MagicMock())
sys.modules.setdefault("chromadb.api.types", MagicMock())

from codesense.agent.nodes import SynthesizeNode
from codesense.llm.gemini_service import GeminiService
from codesense.models.state import (
    AgentState,
    Conflict,
    ConflictSource,
    Evidence,
    Hypothesis,
    NodeType,
    SynthesisResult,
)


def _make_gemini_service_mock(response: str = "WHY: test\nCONFIDENCE: 0.8\nSOURCES: [src1]\nCONFLICTS: [none]") -> GeminiService:
    """Create a mock GeminiService that returns the given response."""
    service = MagicMock(spec=GeminiService)
    service.generate.return_value = response
    return service


def _make_state(**kwargs) -> AgentState:
    """Create a basic AgentState with defaults."""
    defaults = {
        "query": "Why does auth.py exist?",
        "code_path": "src/auth.py",
        "loop_counter": 1,
        "remaining_iterations": 2,
        "evidence": [
            Evidence(
                source_type="git_commit",
                source_id="commit-abc123",
                content="Added authentication module for OAuth2 support",
                timestamp="2024-01-15",
            ),
            Evidence(
                source_type="github_issue",
                source_id="issue-42",
                content="Need OAuth2 integration for SSO",
                timestamp="2024-01-10",
            ),
        ],
        "hypotheses": [
            Hypothesis(
                id="h1",
                explanation="Auth module implements OAuth2 for SSO integration",
                supporting_evidence=["commit-abc123", "issue-42"],
                confidence=0.85,
            ),
        ],
        "confidence_score": 0.85,
        "conflicts": [],
    }
    defaults.update(kwargs)
    return AgentState(**defaults)


class TestSynthesizeNodeInit:
    """Test SynthesizeNode initialization."""

    def test_init_stores_gemini_service(self):
        service = _make_gemini_service_mock()
        node = SynthesizeNode(gemini_service=service)
        assert node._gemini_service is service


class TestSynthesizeNodeExecute:
    """Test the execute method."""

    def test_execute_sets_current_node_to_synthesize(self):
        service = _make_gemini_service_mock()
        node = SynthesizeNode(gemini_service=service)
        state = _make_state()

        result = node.execute(state)

        assert result.current_node == NodeType.SYNTHESIZE

    def test_execute_stores_synthesis_result(self):
        service = _make_gemini_service_mock(
            "WHY: Auth module provides OAuth2 SSO\nCONFIDENCE: 0.85\nSOURCES: [commit-abc123, issue-42]\nCONFLICTS: [none]"
        )
        node = SynthesizeNode(gemini_service=service)
        state = _make_state()

        result = node.execute(state)

        assert result.synthesis is not None
        assert isinstance(result.synthesis, SynthesisResult)
        assert result.synthesis.answer == "Auth module provides OAuth2 SSO"
        assert result.synthesis.confidence == 0.85

    def test_execute_calls_gemini_generate(self):
        service = _make_gemini_service_mock()
        node = SynthesizeNode(gemini_service=service)
        state = _make_state()

        node.execute(state)

        service.generate.assert_called_once()
        prompt = service.generate.call_args[0][0]
        # Prompt should contain the query and code path
        assert "Why does auth.py exist?" in prompt
        assert "src/auth.py" in prompt

    def test_execute_prompt_includes_hypotheses(self):
        service = _make_gemini_service_mock()
        node = SynthesizeNode(gemini_service=service)
        state = _make_state()

        node.execute(state)

        prompt = service.generate.call_args[0][0]
        assert "h1" in prompt
        assert "OAuth2" in prompt
        assert "0.85" in prompt

    def test_execute_prompt_includes_evidence(self):
        service = _make_gemini_service_mock()
        node = SynthesizeNode(gemini_service=service)
        state = _make_state()

        node.execute(state)

        prompt = service.generate.call_args[0][0]
        assert "commit-abc123" in prompt
        assert "issue-42" in prompt

    def test_execute_prompt_includes_conflicts(self):
        conflicts = [
            Conflict(
                id="c1",
                sources=[
                    ConflictSource(source_id="src1", claim="Added for OAuth2"),
                    ConflictSource(source_id="src2", claim="Added for API key auth"),
                ],
                description="Conflicting claims about auth purpose",
            )
        ]
        service = _make_gemini_service_mock()
        node = SynthesizeNode(gemini_service=service)
        state = _make_state(conflicts=conflicts)

        node.execute(state)

        prompt = service.generate.call_args[0][0]
        assert "Conflicting claims about auth purpose" in prompt
        assert "Added for OAuth2" in prompt
        assert "Added for API key auth" in prompt

    def test_execute_includes_conflicts_in_synthesis(self):
        conflicts = [
            Conflict(
                id="c1",
                sources=[
                    ConflictSource(source_id="src1", claim="Claim A"),
                    ConflictSource(source_id="src2", claim="Claim B"),
                ],
                description="Test conflict",
            )
        ]
        response = "WHY: There is a conflict\nCONFIDENCE: 0.6\nSOURCES: [src1, src2]\nCONFLICTS: [Test conflict]"
        service = _make_gemini_service_mock(response)
        node = SynthesizeNode(gemini_service=service)
        state = _make_state(conflicts=conflicts)

        result = node.execute(state)

        assert result.synthesis is not None
        assert len(result.synthesis.conflicts) == 1
        assert result.synthesis.conflicts[0].id == "c1"

    def test_execute_returns_updated_state(self):
        service = _make_gemini_service_mock()
        node = SynthesizeNode(gemini_service=service)
        state = _make_state()

        result = node.execute(state)

        # Should be the same state object, updated
        assert result is state
        assert result.synthesis is not None
        assert result.current_node == NodeType.SYNTHESIZE


class TestSynthesizeNodeIncomplete:
    """Test is_incomplete detection."""

    def test_incomplete_when_max_loops_and_low_confidence(self):
        service = _make_gemini_service_mock()
        node = SynthesizeNode(gemini_service=service)
        state = _make_state(loop_counter=3, confidence_score=0.4)

        result = node.execute(state)

        assert result.synthesis is not None
        assert result.synthesis.is_incomplete is True

    def test_not_incomplete_when_max_loops_and_high_confidence(self):
        service = _make_gemini_service_mock()
        node = SynthesizeNode(gemini_service=service)
        state = _make_state(loop_counter=3, confidence_score=0.8)

        result = node.execute(state)

        assert result.synthesis is not None
        assert result.synthesis.is_incomplete is False

    def test_incomplete_when_state_marked_incomplete(self):
        service = _make_gemini_service_mock()
        node = SynthesizeNode(gemini_service=service)
        state = _make_state(is_incomplete=True)

        result = node.execute(state)

        assert result.synthesis is not None
        assert result.synthesis.is_incomplete is True

    def test_not_incomplete_normal_case(self):
        service = _make_gemini_service_mock()
        node = SynthesizeNode(gemini_service=service)
        state = _make_state(loop_counter=1, confidence_score=0.85)

        result = node.execute(state)

        assert result.synthesis is not None
        assert result.synthesis.is_incomplete is False


class TestSynthesizeNodeFallback:
    """Test fallback behavior when LLM fails."""

    def test_fallback_on_llm_failure(self):
        service = MagicMock(spec=GeminiService)
        service.generate.side_effect = RuntimeError("All API keys exhausted")
        node = SynthesizeNode(gemini_service=service)
        state = _make_state()

        result = node.execute(state)

        # Should still produce a synthesis result
        assert result.synthesis is not None
        assert result.synthesis.is_incomplete is True
        # Should use the best hypothesis as the answer
        assert "OAuth2" in result.synthesis.answer

    def test_fallback_with_no_hypotheses(self):
        service = MagicMock(spec=GeminiService)
        service.generate.side_effect = RuntimeError("LLM failure")
        node = SynthesizeNode(gemini_service=service)
        state = _make_state(hypotheses=[])

        result = node.execute(state)

        assert result.synthesis is not None
        assert result.synthesis.is_incomplete is True
        assert "insufficient" in result.synthesis.answer.lower() or "unable" in result.synthesis.answer.lower()
        assert result.synthesis.confidence == 0.0


class TestSynthesizeNodeReasoningPath:
    """Test reasoning path construction."""

    def test_reasoning_path_single_loop(self):
        service = _make_gemini_service_mock()
        node = SynthesizeNode(gemini_service=service)
        state = _make_state(loop_counter=1)

        result = node.execute(state)

        assert result.synthesis is not None
        expected_path = [
            NodeType.EXPLORE,
            NodeType.HYPOTHESIZE,
            NodeType.VERIFY,
            NodeType.CHECK_CONTRADICTIONS,
            NodeType.SYNTHESIZE,
        ]
        assert result.synthesis.reasoning_path == expected_path

    def test_reasoning_path_multiple_loops(self):
        service = _make_gemini_service_mock()
        node = SynthesizeNode(gemini_service=service)
        state = _make_state(loop_counter=3)

        result = node.execute(state)

        assert result.synthesis is not None
        # 3 cycles + synthesize
        assert len(result.synthesis.reasoning_path) == 3 * 4 + 1
        assert result.synthesis.reasoning_path[-1] == NodeType.SYNTHESIZE

    def test_reasoning_path_zero_loops(self):
        service = _make_gemini_service_mock()
        node = SynthesizeNode(gemini_service=service)
        state = _make_state(loop_counter=0)

        result = node.execute(state)

        assert result.synthesis is not None
        # Just synthesize
        assert result.synthesis.reasoning_path == [NodeType.SYNTHESIZE]


class TestSynthesizeNodeResponseParsing:
    """Test LLM response parsing."""

    def test_parse_well_formatted_response(self):
        response = "WHY: The code implements caching\nCONFIDENCE: 0.9\nSOURCES: [commit-1, issue-5]\nCONFLICTS: [none]"
        service = _make_gemini_service_mock(response)
        node = SynthesizeNode(gemini_service=service)
        state = _make_state()

        result = node.execute(state)

        assert result.synthesis.answer == "The code implements caching"
        assert result.synthesis.confidence == 0.9
        assert "commit-1" in result.synthesis.supporting_evidence
        assert "issue-5" in result.synthesis.supporting_evidence

    def test_parse_unstructured_response(self):
        response = "This code exists because it implements OAuth2 authentication for the application."
        service = _make_gemini_service_mock(response)
        node = SynthesizeNode(gemini_service=service)
        state = _make_state()

        result = node.execute(state)

        # Should use full response as answer and fall back to state confidence
        assert result.synthesis is not None
        assert "OAuth2" in result.synthesis.answer
        assert result.synthesis.confidence == 0.85  # from state

    def test_confidence_clamped_to_valid_range(self):
        response = "WHY: test\nCONFIDENCE: 1.5\nSOURCES: [s1]\nCONFLICTS: [none]"
        service = _make_gemini_service_mock(response)
        node = SynthesizeNode(gemini_service=service)
        state = _make_state()

        result = node.execute(state)

        assert result.synthesis.confidence <= 1.0

    def test_negative_confidence_clamped(self):
        response = "WHY: test\nCONFIDENCE: -0.5\nSOURCES: [s1]\nCONFLICTS: [none]"
        service = _make_gemini_service_mock(response)
        node = SynthesizeNode(gemini_service=service)
        state = _make_state()

        result = node.execute(state)

        assert result.synthesis.confidence >= 0.0

    def test_sources_fallback_to_evidence_when_not_parsed(self):
        response = "WHY: test\nCONFIDENCE: 0.7\nSOURCES: []\nCONFLICTS: [none]"
        service = _make_gemini_service_mock(response)
        node = SynthesizeNode(gemini_service=service)
        state = _make_state()

        result = node.execute(state)

        # Should fall back to evidence source_ids
        assert len(result.synthesis.supporting_evidence) == 2
        assert "commit-abc123" in result.synthesis.supporting_evidence
        assert "issue-42" in result.synthesis.supporting_evidence
