"""Unit tests for ExploreNode."""

from unittest.mock import MagicMock, patch

import pytest

from codesense.agent.nodes import ExploreNode
from codesense.models.state import AgentState, Evidence, NodeType


class TestExploreNodeBasic:
    """Tests for ExploreNode basic behavior."""

    def test_execute_sets_current_node_to_explore(self):
        """ExploreNode.execute should set current_node to NodeType.EXPLORE."""
        node = ExploreNode(mock=True)
        state = AgentState(query="why does auth exist", code_path="src/auth.py")

        result = node.execute(state)

        assert result.current_node == NodeType.EXPLORE

    def test_execute_appends_evidence_from_git_history(self):
        """ExploreNode should convert git history results to Evidence objects."""
        node = ExploreNode(mock=True)
        state = AgentState(query="why does auth exist", code_path="src/auth.py")

        result = node.execute(state)

        git_evidence = [e for e in result.evidence if e.source_type == "git_commit"]
        assert len(git_evidence) > 0
        for ev in git_evidence:
            assert ev.source_id != ""
            assert ev.content != ""

    def test_execute_appends_evidence_from_github_issues(self):
        """ExploreNode should convert GitHub issues to Evidence objects."""
        node = ExploreNode(mock=True)
        state = AgentState(query="why does auth exist", code_path="src/auth.py")

        result = node.execute(state)

        issue_evidence = [e for e in result.evidence if e.source_type == "github_issue"]
        assert len(issue_evidence) > 0
        for ev in issue_evidence:
            assert ev.source_id.startswith("issue-")

    def test_execute_appends_evidence_from_pr_comments(self):
        """ExploreNode should convert PR comments to Evidence objects."""
        node = ExploreNode(mock=True)
        state = AgentState(query="why does auth exist", code_path="src/auth.py")

        result = node.execute(state)

        pr_evidence = [e for e in result.evidence if e.source_type == "pr_comment"]
        assert len(pr_evidence) > 0
        for ev in pr_evidence:
            assert ev.source_id.startswith("pr-")

    def test_execute_appends_evidence_from_related_changes(self):
        """ExploreNode should convert related changes to Evidence objects."""
        node = ExploreNode(mock=True)
        state = AgentState(query="why does auth exist", code_path="src/auth.py")

        result = node.execute(state)

        related_evidence = [e for e in result.evidence if e.source_type == "related_change"]
        assert len(related_evidence) > 0
        for ev in related_evidence:
            assert ev.source_id.startswith("related-")

    def test_execute_preserves_existing_evidence(self):
        """ExploreNode should not overwrite existing evidence in state."""
        node = ExploreNode(mock=True)
        existing = Evidence(
            source_type="git_commit",
            source_id="existing-123",
            content="pre-existing evidence",
        )
        state = AgentState(
            query="why does auth exist",
            code_path="src/auth.py",
            evidence=[existing],
        )

        result = node.execute(state)

        assert existing in result.evidence
        assert len(result.evidence) > 1


class TestExploreNodeGracefulDegradation:
    """Tests for graceful failure handling."""

    def test_continues_when_git_history_fails(self):
        """If git history tool fails, other tools should still produce evidence."""
        node = ExploreNode(mock=True)
        state = AgentState(query="why does auth exist", code_path="src/auth.py")

        with patch(
            "codesense.mcp_server.server.get_git_history",
            side_effect=Exception("git unavailable"),
        ):
            result = node.execute(state)

        # Node should still succeed and collect evidence from other sources
        assert result.current_node == NodeType.EXPLORE
        # Should still have evidence from issues, PR comments, and related changes
        non_git_evidence = [e for e in result.evidence if e.source_type != "git_commit"]
        assert len(non_git_evidence) > 0

    def test_continues_when_tool_returns_error(self):
        """If a tool returns an error dict, node should log and continue."""
        node = ExploreNode(mock=False)
        state = AgentState(query="test", code_path="nonexistent.py")

        # With non-mock mode and no git repo configured, tools return errors
        # The node should still complete gracefully
        result = node.execute(state)
        assert result.current_node == NodeType.EXPLORE

    def test_skips_decision_memory_when_not_configured(self):
        """When no vector_store or embedder provided, skip Decision Memory."""
        node = ExploreNode(mock=True, vector_store=None, embedder=None)
        state = AgentState(query="test query", code_path="src/auth.py")

        result = node.execute(state)

        # Should complete without error and have no decision_unit evidence
        decision_evidence = [e for e in result.evidence if e.source_type == "decision_unit"]
        assert len(decision_evidence) == 0


class TestExploreNodeDecisionMemory:
    """Tests for Decision Memory integration."""

    def test_queries_vector_store_when_configured(self):
        """When vector_store and embedder are provided, query Decision Memory."""
        mock_embedder = MagicMock()
        mock_embedder.embed_single.return_value = [0.1] * 384

        mock_vector_store = MagicMock()
        mock_retrieval_result = MagicMock()
        mock_retrieval_result.decision_unit.id = "du-001"
        mock_retrieval_result.decision_unit.content = "We chose JWT for auth."
        mock_retrieval_result.decision_unit.ingestion_timestamp = "2024-01-01T00:00:00"
        mock_retrieval_result.decision_unit.source_document = "adr-001.md"
        mock_retrieval_result.decision_unit.section_heading = "Decision"
        mock_retrieval_result.decision_unit.referenced_components = ["auth"]
        mock_retrieval_result.similarity_score = 0.85
        mock_vector_store.query.return_value = [mock_retrieval_result]

        node = ExploreNode(
            mock=True,
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )
        state = AgentState(query="why JWT for auth", code_path="src/auth.py")

        result = node.execute(state)

        # Should have decision_unit evidence
        decision_evidence = [e for e in result.evidence if e.source_type == "decision_unit"]
        assert len(decision_evidence) == 1
        assert decision_evidence[0].source_id == "du-001"
        assert decision_evidence[0].content == "We chose JWT for auth."
        assert decision_evidence[0].metadata["similarity_score"] == 0.85

    def test_handles_empty_embedding(self):
        """When embedder returns empty embedding, skip Decision Memory."""
        mock_embedder = MagicMock()
        mock_embedder.embed_single.return_value = []

        mock_vector_store = MagicMock()

        node = ExploreNode(
            mock=True,
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )
        state = AgentState(query="", code_path="src/auth.py")

        result = node.execute(state)

        mock_vector_store.query.assert_not_called()

    def test_handles_vector_store_exception(self):
        """When vector store raises, node should log and continue."""
        mock_embedder = MagicMock()
        mock_embedder.embed_single.return_value = [0.1] * 384

        mock_vector_store = MagicMock()
        mock_vector_store.query.side_effect = Exception("DB unavailable")

        node = ExploreNode(
            mock=True,
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )
        state = AgentState(query="test query", code_path="src/auth.py")

        result = node.execute(state)

        # Should complete without crashing
        assert result.current_node == NodeType.EXPLORE
        decision_evidence = [e for e in result.evidence if e.source_type == "decision_unit"]
        assert len(decision_evidence) == 0
