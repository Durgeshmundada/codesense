"""Explain capability handler — dispatches to the reasoning loop.

Answers "WHY does this code exist?" by running the full reasoning graph
and formatting the synthesis result as a CommandOutput.

Requirements: 5.1, 7.3, 7.4
"""

import logging
import os
import re
from typing import Optional

from codesense.models.output import CommandOutput, CommandParams

logger = logging.getLogger(__name__)

REPORT_TITLE = "CodeSense — Archaeology Report"


class ExplainHandler:
    """Capability handler for the 'explain' command.

    Implements the CapabilityHandler protocol. Dispatches to the ReasoningGraph
    to gather evidence, generate hypotheses, verify them, check for contradictions,
    and produce a synthesized explanation of WHY the specified code exists.

    Constructor takes optional dependencies (gemini_service, vector_store, embedder).
    If not provided, defaults are created from environment configuration.

    Args:
        gemini_service: Optional GeminiService instance for LLM calls.
            If None, a default is created using GEMINI_API_KEYS env var.
        vector_store: Optional VectorStore for Decision Memory retrieval.
            If None, a default is created with default persist directory.
        embedder: Optional HuggingFaceEmbedder for query embedding.
            If None, a default is created with the default model.
    """

    def __init__(
        self,
        gemini_service=None,
        vector_store=None,
        embedder=None,
    ) -> None:
        self._gemini_service = gemini_service
        self._vector_store = vector_store
        self._embedder = embedder

    def _get_gemini_service(self):
        """Get or create the GeminiService instance."""
        if self._gemini_service is None:
            from codesense.llm.gemini_service import GeminiService
            from codesense.llm.key_manager import KeyRotator

            api_keys_str = os.environ.get("GEMINI_API_KEYS", "")
            if not api_keys_str:
                # Try single key fallback
                single_key = os.environ.get("GEMINI_API_KEY", "")
                if single_key:
                    api_keys_str = single_key

            keys = [k.strip() for k in api_keys_str.split(",") if k.strip()]
            if not keys:
                raise RuntimeError(
                    "No Gemini API keys configured. "
                    "Set GEMINI_API_KEYS or GEMINI_API_KEY environment variable."
                )

            rotator = KeyRotator(api_keys=keys)
            self._gemini_service = GeminiService(key_rotator=rotator)

        return self._gemini_service

    def _get_vector_store(self):
        """Get or create the VectorStore instance."""
        if self._vector_store is None:
            from codesense.memory.vector_store import VectorStore

            self._vector_store = VectorStore()
        return self._vector_store

    def _get_embedder(self):
        """Get or create the HuggingFaceEmbedder instance."""
        if self._embedder is None:
            from codesense.memory.embedder import HuggingFaceEmbedder

            self._embedder = HuggingFaceEmbedder()
        return self._embedder

    def _build_query(self, params: CommandParams) -> str:
        """Build the reasoning query from command parameters.

        Includes function name and line number if provided in params
        to help the reasoning loop focus on specific code locations.

        Args:
            params: Parsed CLI arguments.

        Returns:
            A descriptive query string for the reasoning loop.
        """
        code_path = params.path or ""

        # Use explicit query if provided
        if params.query:
            return params.query

        parts = [f"Why does this code exist: {code_path}"]

        # Include function name if provided
        if params.function_name:
            parts.append(f"function: {params.function_name}")

        # Include line number if provided
        if params.line_number is not None:
            parts.append(f"line: {params.line_number}")

        return ", ".join(parts)

    def _describe_source(self, ev) -> str:
        """Build a human-readable descriptor for an evidence source.

        Turns opaque source IDs into meaningful citations, e.g.
        "git commit `9d6df84` — Fix retry logic (by alice)".
        """
        st = ev.source_type
        md = ev.metadata or {}
        first_line = ""
        if ev.content:
            stripped = ev.content.strip().splitlines()
            if stripped:
                first_line = stripped[0][:80]

        if st == "git_commit":
            desc = f"git commit `{ev.source_id[:8]}`"
            if first_line:
                desc += f" — {first_line}"
            author = md.get("author", "")
            if author:
                desc += f" (by {author})"
            return desc
        if st == "github_issue":
            num = str(ev.source_id).replace("issue-", "#")
            desc = f"GitHub issue {num}"
            # The content is stored as "#<n>: <title>"; strip the leading
            # "#<n>:" so we don't render "issue #42 — #42: ...".
            title = re.sub(r"^#\d+:\s*", "", first_line) if first_line else ""
            if title:
                desc += f" — {title}"
            return desc
        if st == "pr_comment":
            pr = md.get("pr_number", "")
            desc = f"PR #{pr} review comment" if pr else "PR review comment"
            author = md.get("author", "")
            if author:
                desc += f" by {author}"
            return desc
        if st == "decision_unit":
            doc = md.get("source_document", "decision doc")
            sec = md.get("section_heading", "")
            desc = f"decision doc `{doc}`"
            if sec:
                desc += f" § {sec}"
            return desc
        # Fallback: still avoid a bare opaque UUID
        return f"{st}: {first_line}" if first_line else st

    def _format_citations(self, supporting_evidence: list[str], evidence_lookup: dict) -> str:
        """Format source citations into readable markdown.

        Only evidence-backed IDs are shown; opaque internal IDs (e.g. hypothesis
        UUIDs not tied to a real source) are skipped so citations stay meaningful.

        Args:
            supporting_evidence: List of source IDs from synthesis.
            evidence_lookup: Map of source_id -> Evidence for descriptor lookup.

        Returns:
            Formatted citation string in markdown, or "" if no real sources.
        """
        lines: list[str] = []
        seen: set[str] = set()
        for source_id in supporting_evidence:
            ev = evidence_lookup.get(source_id)
            if ev is None:
                continue  # skip opaque internal references
            descriptor = self._describe_source(ev)
            if descriptor in seen:
                continue
            seen.add(descriptor)
            lines.append(descriptor)

        if not lines:
            return ""

        citation_text = "\n\n---\n**Sources:**\n"
        for i, line in enumerate(lines, 1):
            citation_text += f"  {i}. {line}\n"
        return citation_text

    def run(self, params: CommandParams) -> CommandOutput:
        """Execute the explain capability with the given parameters.

        Dispatches to the ReasoningGraph.run() with the code path as query.
        The query includes function name and line number if provided in params.

        On success, returns a CommandOutput with:
          - title: "CodeSense — Archaeology Report"
          - content: the synthesis answer in markdown with formatted citations
          - confidence: from SynthesisResult
          - conflicts: from SynthesisResult
          - is_demo_mode: from params.mock
          - code_snippets: empty (archaeology doesn't produce code)
          - tables: empty

        On failure, returns a CommandOutput with error information.

        Args:
            params: Parsed CLI arguments. Must include `path`.

        Returns:
            CommandOutput formatted for Rich display.
        """
        code_path = params.path or ""
        query = self._build_query(params)
        is_demo = params.mock

        try:
            # Obtain dependencies (create defaults if needed)
            gemini_service = self._get_gemini_service()
            vector_store = self._get_vector_store()
            embedder = self._get_embedder()

            # Build the reasoning graph
            from codesense.agent.graph import ReasoningGraph

            graph = ReasoningGraph(
                gemini_service=gemini_service,
                mock=is_demo,
                vector_store=vector_store,
                embedder=embedder,
            )

            # Run the reasoning graph
            final_state = graph.run(query=query, code_path=code_path, mock=is_demo)

        except Exception as e:
            logger.error("Reasoning loop failed: %s", e, exc_info=True)
            return CommandOutput(
                title=REPORT_TITLE,
                content=(
                    f"**Error:** Unable to complete analysis.\n\n"
                    f"The reasoning loop encountered an error: {e}\n\n"
                    f"Please check your configuration and try again."
                ),
                code_snippets=[],
                tables=[],
                conflicts=[],
                confidence=0.0,
                is_demo_mode=is_demo,
            )

        # Extract synthesis result
        synthesis = final_state.synthesis

        if synthesis is None:
            return CommandOutput(
                title=REPORT_TITLE,
                content=(
                    "Unable to produce an explanation. "
                    "The reasoning loop did not reach synthesis."
                ),
                code_snippets=[],
                tables=[],
                conflicts=[],
                confidence=0.0,
                is_demo_mode=is_demo,
            )

        # Compose content: main answer + readable, evidence-backed citations.
        content = synthesis.answer

        # Evidence-grounded honesty: cap confidence and add disclaimers when the
        # answer isn't backed by real historical evidence or reasoning didn't
        # complete — so the report never sounds authoritative while guessing.
        substantive_types = {"git_commit", "github_issue", "pr_comment", "decision_unit"}
        evidence = final_state.evidence or []
        evidence_lookup = {e.source_id: e for e in evidence}
        has_evidence = any(e.source_type in substantive_types for e in evidence)

        confidence = synthesis.confidence
        incomplete = bool(synthesis.is_incomplete or final_state.is_incomplete)
        notes: list[str] = []

        if incomplete:
            # A high confidence score alongside an "incomplete" flag is
            # contradictory — cap it so the two signals agree.
            confidence = min(confidence, 0.5)
            notes.append(
                "Reasoning did not fully complete (an LLM or data step failed), "
                "so this explanation is based on partial analysis."
            )
        if not has_evidence:
            confidence = min(confidence, 0.4)
            notes.append(
                "No historical evidence (git history, GitHub issues/PRs, or decision "
                "docs) was found for this code. This explanation is inferred from the "
                "code itself and general patterns, not from recorded rationale."
            )

        content += self._format_citations(synthesis.supporting_evidence, evidence_lookup)

        if notes:
            content += "\n\n---\n"
            for note in notes:
                content += f"\n⚠️ *{note}*\n"

        return CommandOutput(
            title=REPORT_TITLE,
            content=content,
            confidence=confidence,
            conflicts=synthesis.conflicts,
            is_demo_mode=is_demo,
            code_snippets=[],
            tables=[],
        )
