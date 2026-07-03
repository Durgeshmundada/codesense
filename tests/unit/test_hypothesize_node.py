"""Unit tests for HypothesizeNode."""

import json
from unittest.mock import MagicMock

import pytest

from codesense.agent.nodes import HypothesizeNode
from codesense.llm.gemini_service import GeminiService
from codesense.models.state import (
    AgentState,
    Conflict,
    ConflictSource,
    Evidence,
    Hypothesis,
    NodeType,
)


@pytest.fixture
def mock_gemini_service():
    """Create a mocked GeminiService."""
    service = MagicMock(spec=GeminiService)
    return service


@pytest.fixture
def sample_evidence():
    """Create sample evidence for testing."""
    return [
        Evidence(
            source_type="git_commit",
            source_id="commit_abc123",
            content="Added authentication middleware to handle JWT tokens",
            timestamp="2024-01-15T10:00:00Z",
        ),
        Evidence(
            source_type="github_issue",
            source_id="issue_42",
            content="Security audit required token validation on all endpoints",
            timestamp="2024-01-10T08:00:00Z",
        ),
        Evidence(
            source_type="pr_comment",
            source_id="pr_comment_99",
            content="Reviewer suggested using middleware pattern for auth",
            timestamp="2024-01-14T14:30:00Z",
        ),
    ]


@pytest.fixture
def sample_state(sample_evidence):
    """Create a sample AgentState with evidence."""
    return AgentState(
        query="Why does the auth middleware exist?",
        code_path="src/middleware/auth.py",
        evidence=sample_evidence,
    )


class TestHypothesizeNodeExecute:
    """Tests for HypothesizeNode.execute method."""

    def test_execute_sets_current_node(self, mock_gemini_service, sample_state):
        """execute() should set current_node to NodeType.HYPOTHESIZE."""
        mock_gemini_service.generate.return_value = json.dumps([
            {
                "explanation": "Code exists for authentication",
                "confidence": 0.8,
                "supporting_evidence": ["commit_abc123"],
            }
        ])
        node = HypothesizeNode(gemini_service=mock_gemini_service)
        result = node.execute(sample_state)
        assert result.current_node == NodeType.HYPOTHESIZE

    def test_execute_stores_hypotheses_in_state(self, mock_gemini_service, sample_state):
        """execute() should store parsed hypotheses in state.hypotheses."""
        mock_gemini_service.generate.return_value = json.dumps([
            {
                "explanation": "Auth middleware for JWT validation",
                "confidence": 0.85,
                "supporting_evidence": ["commit_abc123", "issue_42"],
            },
            {
                "explanation": "Compliance requirement for token handling",
                "confidence": 0.7,
                "supporting_evidence": ["issue_42"],
            },
        ])
        node = HypothesizeNode(gemini_service=mock_gemini_service)
        result = node.execute(sample_state)

        assert len(result.hypotheses) == 2
        assert result.hypotheses[0].explanation == "Auth middleware for JWT validation"
        assert result.hypotheses[0].confidence == 0.85
        assert "commit_abc123" in result.hypotheses[0].supporting_evidence
        assert "issue_42" in result.hypotheses[0].supporting_evidence

    def test_execute_calls_gemini_service(self, mock_gemini_service, sample_state):
        """execute() should call GeminiService.generate with a prompt."""
        mock_gemini_service.generate.return_value = json.dumps([
            {
                "explanation": "Test hypothesis",
                "confidence": 0.5,
                "supporting_evidence": [],
            }
        ])
        node = HypothesizeNode(gemini_service=mock_gemini_service)
        node.execute(sample_state)

        mock_gemini_service.generate.assert_called_once()
        prompt = mock_gemini_service.generate.call_args[0][0]
        assert "Why does the auth middleware exist?" in prompt
        assert "src/middleware/auth.py" in prompt

    def test_execute_includes_evidence_in_prompt(self, mock_gemini_service, sample_state):
        """execute() should include evidence details in the prompt."""
        mock_gemini_service.generate.return_value = json.dumps([
            {
                "explanation": "Test",
                "confidence": 0.5,
                "supporting_evidence": [],
            }
        ])
        node = HypothesizeNode(gemini_service=mock_gemini_service)
        node.execute(sample_state)

        prompt = mock_gemini_service.generate.call_args[0][0]
        assert "commit_abc123" in prompt
        assert "issue_42" in prompt

    def test_execute_includes_conflicts_in_prompt(self, mock_gemini_service, sample_state):
        """execute() should include known conflicts in the prompt."""
        sample_state.conflicts = [
            Conflict(
                id="conflict_1",
                sources=[
                    ConflictSource(source_id="commit_abc123", claim="Added for security"),
                    ConflictSource(source_id="issue_42", claim="Added for compliance"),
                ],
                description="Disagreement on motivation for auth middleware",
            )
        ]
        mock_gemini_service.generate.return_value = json.dumps([
            {
                "explanation": "Test",
                "confidence": 0.5,
                "supporting_evidence": [],
            }
        ])
        node = HypothesizeNode(gemini_service=mock_gemini_service)
        node.execute(sample_state)

        prompt = mock_gemini_service.generate.call_args[0][0]
        assert "Disagreement on motivation" in prompt
        assert "Known Conflicts" in prompt


class TestHypothesizeNodeBounds:
    """Tests for hypothesis count bounds enforcement."""

    def test_min_one_hypothesis_on_empty_response(self, mock_gemini_service, sample_state):
        """Should produce at least 1 hypothesis even if LLM returns empty list."""
        mock_gemini_service.generate.return_value = "[]"
        node = HypothesizeNode(gemini_service=mock_gemini_service)
        result = node.execute(sample_state)
        assert len(result.hypotheses) >= 1

    def test_min_one_hypothesis_on_invalid_json(self, mock_gemini_service, sample_state):
        """Should produce at least 1 fallback hypothesis if LLM returns garbage."""
        mock_gemini_service.generate.return_value = "not valid json at all"
        node = HypothesizeNode(gemini_service=mock_gemini_service)
        result = node.execute(sample_state)
        assert len(result.hypotheses) >= 1
        # Fallback hypothesis has low confidence
        assert result.hypotheses[0].confidence == 0.1

    def test_max_five_hypotheses(self, mock_gemini_service, sample_state):
        """Should cap hypotheses at 5 even if LLM returns more."""
        many_hypotheses = [
            {"explanation": f"Hypothesis {i}", "confidence": 0.5 + i * 0.05, "supporting_evidence": []}
            for i in range(8)
        ]
        mock_gemini_service.generate.return_value = json.dumps(many_hypotheses)
        node = HypothesizeNode(gemini_service=mock_gemini_service)
        result = node.execute(sample_state)
        assert len(result.hypotheses) <= 5

    def test_exactly_five_hypotheses_allowed(self, mock_gemini_service, sample_state):
        """Should allow exactly 5 hypotheses without trimming."""
        five_hypotheses = [
            {"explanation": f"Hypothesis {i}", "confidence": 0.6, "supporting_evidence": []}
            for i in range(5)
        ]
        mock_gemini_service.generate.return_value = json.dumps(five_hypotheses)
        node = HypothesizeNode(gemini_service=mock_gemini_service)
        result = node.execute(sample_state)
        assert len(result.hypotheses) == 5


class TestHypothesizeNodeParsing:
    """Tests for LLM response parsing."""

    def test_parse_json_in_code_fence(self, mock_gemini_service, sample_state):
        """Should extract JSON from markdown code fences."""
        response = """Here are my hypotheses:

```json
[
  {
    "explanation": "Auth middleware for security",
    "confidence": 0.9,
    "supporting_evidence": ["commit_abc123"]
  }
]
```
"""
        mock_gemini_service.generate.return_value = response
        node = HypothesizeNode(gemini_service=mock_gemini_service)
        result = node.execute(sample_state)

        assert len(result.hypotheses) == 1
        assert result.hypotheses[0].explanation == "Auth middleware for security"
        assert result.hypotheses[0].confidence == 0.9

    def test_confidence_clamped_to_range(self, mock_gemini_service, sample_state):
        """Should clamp confidence values to [0.0, 1.0]."""
        mock_gemini_service.generate.return_value = json.dumps([
            {"explanation": "High confidence", "confidence": 1.5, "supporting_evidence": []},
            {"explanation": "Negative confidence", "confidence": -0.3, "supporting_evidence": []},
        ])
        node = HypothesizeNode(gemini_service=mock_gemini_service)
        result = node.execute(sample_state)

        assert result.hypotheses[0].confidence == 1.0
        assert result.hypotheses[1].confidence == 0.0

    def test_invalid_source_ids_filtered(self, mock_gemini_service, sample_state):
        """Should filter out source_ids that don't match any evidence."""
        mock_gemini_service.generate.return_value = json.dumps([
            {
                "explanation": "Test hypothesis",
                "confidence": 0.7,
                "supporting_evidence": ["commit_abc123", "nonexistent_id", "issue_42"],
            }
        ])
        node = HypothesizeNode(gemini_service=mock_gemini_service)
        result = node.execute(sample_state)

        assert "commit_abc123" in result.hypotheses[0].supporting_evidence
        assert "issue_42" in result.hypotheses[0].supporting_evidence
        assert "nonexistent_id" not in result.hypotheses[0].supporting_evidence

    def test_hypothesis_gets_uuid(self, mock_gemini_service, sample_state):
        """Each hypothesis should have a unique UUID id."""
        mock_gemini_service.generate.return_value = json.dumps([
            {"explanation": "Hyp 1", "confidence": 0.8, "supporting_evidence": []},
            {"explanation": "Hyp 2", "confidence": 0.6, "supporting_evidence": []},
        ])
        node = HypothesizeNode(gemini_service=mock_gemini_service)
        result = node.execute(sample_state)

        assert result.hypotheses[0].id != result.hypotheses[1].id
        # Verify they look like UUIDs (36 chars with hyphens)
        assert len(result.hypotheses[0].id) == 36

    def test_empty_evidence_still_produces_hypothesis(self, mock_gemini_service):
        """Should still work when no evidence is gathered."""
        state = AgentState(
            query="Why does this code exist?",
            code_path="src/foo.py",
            evidence=[],
        )
        mock_gemini_service.generate.return_value = json.dumps([
            {"explanation": "Unknown reason", "confidence": 0.3, "supporting_evidence": []}
        ])
        node = HypothesizeNode(gemini_service=mock_gemini_service)
        result = node.execute(state)

        assert len(result.hypotheses) >= 1

    def test_returns_updated_state_preserves_fields(self, mock_gemini_service, sample_state):
        """execute() should return an AgentState preserving other fields."""
        mock_gemini_service.generate.return_value = json.dumps([
            {"explanation": "Test", "confidence": 0.5, "supporting_evidence": []}
        ])
        node = HypothesizeNode(gemini_service=mock_gemini_service)
        result = node.execute(sample_state)

        assert isinstance(result, AgentState)
        assert result.query == "Why does the auth middleware exist?"
        assert result.code_path == "src/middleware/auth.py"
        # Evidence should be preserved
        assert len(result.evidence) == 3

    def test_non_string_supporting_evidence_filtered(self, mock_gemini_service, sample_state):
        """Should filter out non-string items in supporting_evidence."""
        mock_gemini_service.generate.return_value = json.dumps([
            {
                "explanation": "Test",
                "confidence": 0.7,
                "supporting_evidence": ["commit_abc123", 123, None, "issue_42"],
            }
        ])
        node = HypothesizeNode(gemini_service=mock_gemini_service)
        result = node.execute(sample_state)

        assert "commit_abc123" in result.hypotheses[0].supporting_evidence
        assert "issue_42" in result.hypotheses[0].supporting_evidence
        assert len(result.hypotheses[0].supporting_evidence) == 2
