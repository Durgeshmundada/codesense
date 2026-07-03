"""Unit tests for VerifyNode."""

import json
from unittest.mock import MagicMock, patch

import pytest

from codesense.agent.nodes import VerifyNode
from codesense.models.state import (
    AgentState,
    Evidence,
    Hypothesis,
    NodeType,
)


def _make_state(
    evidence: list[Evidence] | None = None,
    hypotheses: list[Hypothesis] | None = None,
) -> AgentState:
    """Create a test AgentState with some defaults."""
    return AgentState(
        query="Why does auth.py exist?",
        code_path="src/auth.py",
        loop_counter=1,
        remaining_iterations=2,
        evidence=evidence or [
            Evidence(
                source_type="git_commit",
                source_id="commit-abc123",
                content="Added authentication middleware for API endpoints",
                timestamp="2024-01-15",
            ),
            Evidence(
                source_type="github_issue",
                source_id="issue-42",
                content="Need to secure API endpoints with token auth",
            ),
        ],
        hypotheses=hypotheses or [
            Hypothesis(
                id="hyp-1",
                explanation="auth.py was created to implement token-based authentication for API security",
                supporting_evidence=["commit-abc123", "issue-42"],
                confidence=0.0,
            ),
            Hypothesis(
                id="hyp-2",
                explanation="auth.py provides rate limiting functionality",
                supporting_evidence=["commit-abc123"],
                confidence=0.0,
            ),
        ],
        confidence_score=0.0,
        current_node=NodeType.HYPOTHESIZE,
    )


def _mock_gemini_service(response: str) -> MagicMock:
    """Create a mock GeminiService that returns the given response."""
    mock_service = MagicMock()
    mock_service.generate.return_value = response
    return mock_service


class TestVerifyNodeExecute:
    """Tests for VerifyNode.execute method."""

    def test_sets_current_node_to_verify(self):
        """VerifyNode sets current_node to NodeType.VERIFY."""
        response = json.dumps({
            "hypothesis_scores": [
                {"hypothesis_id": "hyp-1", "confidence": 0.8, "reasoning": "Strong evidence"},
            ],
            "overall_confidence": 0.75,
            "verification_summary": "Good support",
        })
        gemini = _mock_gemini_service(response)
        node = VerifyNode(gemini_service=gemini)
        state = _make_state()

        result = node.execute(state)

        assert result.current_node == NodeType.VERIFY

    def test_produces_confidence_score_in_valid_range(self):
        """VerifyNode always produces confidence in [0.0, 1.0]."""
        response = json.dumps({
            "hypothesis_scores": [],
            "overall_confidence": 0.65,
            "verification_summary": "Moderate support",
        })
        gemini = _mock_gemini_service(response)
        node = VerifyNode(gemini_service=gemini)
        state = _make_state()

        result = node.execute(state)

        assert 0.0 <= result.confidence_score <= 1.0
        assert result.confidence_score == 0.65

    def test_clamps_high_confidence_to_1(self):
        """Confidence scores above 1.0 are clamped."""
        response = json.dumps({
            "hypothesis_scores": [],
            "overall_confidence": 1.5,
            "verification_summary": "Over-confident",
        })
        gemini = _mock_gemini_service(response)
        node = VerifyNode(gemini_service=gemini)
        state = _make_state()

        result = node.execute(state)

        assert result.confidence_score == 1.0

    def test_clamps_negative_confidence_to_0(self):
        """Confidence scores below 0.0 are clamped."""
        response = json.dumps({
            "hypothesis_scores": [],
            "overall_confidence": -0.3,
            "verification_summary": "Negative?",
        })
        gemini = _mock_gemini_service(response)
        node = VerifyNode(gemini_service=gemini)
        state = _make_state()

        result = node.execute(state)

        assert result.confidence_score == 0.0

    def test_updates_hypothesis_confidence_scores(self):
        """VerifyNode updates individual hypothesis confidence scores."""
        response = json.dumps({
            "hypothesis_scores": [
                {"hypothesis_id": "hyp-1", "confidence": 0.85, "reasoning": "Strong"},
                {"hypothesis_id": "hyp-2", "confidence": 0.3, "reasoning": "Weak"},
            ],
            "overall_confidence": 0.8,
            "verification_summary": "First hypothesis well-supported",
        })
        gemini = _mock_gemini_service(response)
        node = VerifyNode(gemini_service=gemini)
        state = _make_state()

        result = node.execute(state)

        assert result.hypotheses[0].confidence == 0.85
        assert result.hypotheses[1].confidence == 0.3

    def test_handles_json_in_code_fence(self):
        """VerifyNode handles LLM response wrapped in markdown code fences."""
        inner_json = json.dumps({
            "hypothesis_scores": [
                {"hypothesis_id": "hyp-1", "confidence": 0.7, "reasoning": "OK"},
            ],
            "overall_confidence": 0.72,
            "verification_summary": "Decent support",
        })
        response = f"Here is the result:\n```json\n{inner_json}\n```\n"
        gemini = _mock_gemini_service(response)
        node = VerifyNode(gemini_service=gemini)
        state = _make_state()

        result = node.execute(state)

        assert result.confidence_score == 0.72

    def test_fallback_parsing_on_malformed_json(self):
        """VerifyNode uses regex fallback when JSON parsing fails."""
        response = 'Based on analysis, "overall_confidence": 0.55 seems right.'
        gemini = _mock_gemini_service(response)
        node = VerifyNode(gemini_service=gemini)
        state = _make_state()

        result = node.execute(state)

        assert result.confidence_score == 0.55

    def test_default_confidence_on_unparseable_response(self):
        """VerifyNode returns conservative 0.3 when response is unparseable."""
        response = "I cannot determine confidence from the available data."
        gemini = _mock_gemini_service(response)
        node = VerifyNode(gemini_service=gemini)
        state = _make_state()

        result = node.execute(state)

        assert result.confidence_score == 0.3

    def test_preserves_existing_evidence(self):
        """VerifyNode preserves existing evidence in state."""
        response = json.dumps({
            "hypothesis_scores": [],
            "overall_confidence": 0.6,
            "verification_summary": "OK",
        })
        gemini = _mock_gemini_service(response)
        node = VerifyNode(gemini_service=gemini)
        state = _make_state()

        result = node.execute(state)

        assert len(result.evidence) >= len(state.evidence)
        # Original evidence should still be present
        original_ids = {e.source_id for e in state.evidence}
        result_ids = {e.source_id for e in result.evidence}
        assert original_ids.issubset(result_ids)

    def test_preserves_state_fields(self):
        """VerifyNode preserves query, code_path, loop_counter, etc."""
        response = json.dumps({
            "hypothesis_scores": [],
            "overall_confidence": 0.5,
            "verification_summary": "OK",
        })
        gemini = _mock_gemini_service(response)
        node = VerifyNode(gemini_service=gemini)
        state = _make_state()

        result = node.execute(state)

        assert result.query == state.query
        assert result.code_path == state.code_path
        assert result.loop_counter == state.loop_counter
        assert result.remaining_iterations == state.remaining_iterations


class TestVerifyNodeWithDecisionMemory:
    """Tests for VerifyNode with optional VectorStore integration."""

    def test_queries_vector_store_when_available(self):
        """VerifyNode queries Decision Memory when vector_store and embedder are provided."""
        from codesense.models.memory import DecisionUnit, RetrievalResult

        response = json.dumps({
            "hypothesis_scores": [],
            "overall_confidence": 0.8,
            "verification_summary": "Memory-supported",
        })
        gemini = _mock_gemini_service(response)

        mock_embedder = MagicMock()
        mock_embedder.embed_single.return_value = [0.1] * 384

        mock_unit = DecisionUnit(
            id="du-1",
            content="We chose token auth for API security",
            section_heading="Decision: Auth Mechanism",
            source_document="adr-001.md",
            ingestion_timestamp="2024-01-01",
        )
        mock_result = RetrievalResult(
            decision_unit=mock_unit,
            similarity_score=0.85,
            metadata={
                "source_document": "adr-001.md",
                "section_heading": "Decision: Auth Mechanism",
                "ingestion_timestamp": "2024-01-01",
            },
        )

        mock_vector_store = MagicMock()
        mock_vector_store.query.return_value = [mock_result]

        node = VerifyNode(
            gemini_service=gemini,
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )
        state = _make_state()

        result = node.execute(state)

        # Should have added decision_unit evidence
        decision_evidence = [
            e for e in result.evidence if e.source_type == "decision_unit"
        ]
        assert len(decision_evidence) >= 1
        assert decision_evidence[0].source_id == "du-1"

    def test_no_error_without_vector_store(self):
        """VerifyNode works fine without vector_store (optional)."""
        response = json.dumps({
            "hypothesis_scores": [],
            "overall_confidence": 0.5,
            "verification_summary": "OK",
        })
        gemini = _mock_gemini_service(response)
        node = VerifyNode(gemini_service=gemini)
        state = _make_state()

        result = node.execute(state)

        assert result.confidence_score == 0.5
        # No additional evidence should be added
        assert len(result.evidence) == len(state.evidence)


class TestVerifyNodeConfidenceParsing:
    """Tests for the confidence score parsing logic."""

    def test_parses_json_with_overall_confidence_key(self):
        """Correctly parses overall_confidence from JSON."""
        node = VerifyNode(gemini_service=MagicMock())
        response = json.dumps({"overall_confidence": 0.82})
        score = node._parse_confidence_score(response)
        assert score == 0.82

    def test_parses_from_code_fence(self):
        """Extracts confidence from JSON inside code fence."""
        node = VerifyNode(gemini_service=MagicMock())
        inner = json.dumps({"overall_confidence": 0.91})
        response = f"```json\n{inner}\n```"
        score = node._parse_confidence_score(response)
        assert score == 0.91

    def test_clamps_above_1(self):
        """Clamps values above 1.0."""
        node = VerifyNode(gemini_service=MagicMock())
        response = json.dumps({"overall_confidence": 2.5})
        score = node._parse_confidence_score(response)
        assert score == 1.0

    def test_clamps_below_0(self):
        """Clamps values below 0.0."""
        node = VerifyNode(gemini_service=MagicMock())
        response = json.dumps({"overall_confidence": -0.1})
        score = node._parse_confidence_score(response)
        assert score == 0.0

    def test_regex_fallback_for_text_response(self):
        """Falls back to regex when JSON parsing fails."""
        node = VerifyNode(gemini_service=MagicMock())
        response = 'Based on my analysis, "overall_confidence": 0.67 seems appropriate.'
        score = node._parse_confidence_score(response)
        assert score == 0.67

    def test_returns_default_on_garbage(self):
        """Returns 0.3 when nothing can be parsed."""
        node = VerifyNode(gemini_service=MagicMock())
        response = "This is completely unparseable text with no numbers."
        score = node._parse_confidence_score(response)
        assert score == 0.3
