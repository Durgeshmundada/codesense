"""Reasoning agent nodes for the LangGraph reasoning loop.

Each node implements the Node protocol: execute(state: AgentState) -> AgentState.
Nodes form the 5-step reasoning cycle: explore → hypothesize → verify →
check_contradictions → synthesize.
"""

import json
import logging
import re
import uuid
from typing import Optional

from codesense.llm.gemini_service import GeminiService
from codesense.memory.embedder import HuggingFaceEmbedder
from codesense.memory.vector_store import VectorStore
from codesense.models.state import (
    AgentState,
    Conflict,
    Evidence,
    Hypothesis,
    NodeType,
    SynthesisResult,
)

logger = logging.getLogger(__name__)


class ExploreNode:
    """Gathers evidence from MCP tools and Decision Memory.

    Calls MCP server module functions directly (get_git_history, get_github_issues,
    get_pr_comments, get_related_changes) and optionally queries the Decision Memory
    vector store for relevant decision units.

    All gathered evidence is appended to state.evidence. Individual tool failures
    are logged but do not halt exploration — the node continues with whatever
    data it can gather (graceful degradation per Requirement 1.9).

    Args:
        mock: Whether to use mock data sources for MCP tool calls.
        vector_store: Optional VectorStore instance for Decision Memory queries.
        embedder: Optional HuggingFaceEmbedder for embedding the query text.
    """

    def __init__(
        self,
        mock: bool = False,
        vector_store: Optional[VectorStore] = None,
        embedder: Optional[HuggingFaceEmbedder] = None,
    ) -> None:
        self._mock = mock
        self._vector_store = vector_store
        self._embedder = embedder

    def execute(self, state: AgentState) -> AgentState:
        """Execute exploration: gather evidence from all available sources.

        Calls MCP tools (via direct function calls to the server module) and
        Decision Memory, converts results to Evidence objects, and appends
        them to state.evidence. Sets current_node to NodeType.EXPLORE.

        Args:
            state: The current agent state with query and code_path.

        Returns:
            Updated AgentState with new evidence appended and
            current_node set to NodeType.EXPLORE.
        """
        from codesense.mcp_server.server import (
            get_git_history,
            get_github_issues,
            get_pr_comments,
            get_related_changes,
        )

        new_evidence: list[Evidence] = []

        # 1. Get git history → Evidence(source_type="git_commit")
        new_evidence.extend(self._gather_git_history(get_git_history, state.code_path))

        # 2. Get GitHub issues → Evidence(source_type="github_issue")
        new_evidence.extend(self._gather_github_issues(get_github_issues, state.query))

        # 3. Get PR comments → Evidence(source_type="pr_comment")
        new_evidence.extend(self._gather_pr_comments(get_pr_comments, state.code_path))

        # 4. Get related changes → Evidence(source_type="related_change")
        new_evidence.extend(
            self._gather_related_changes(get_related_changes, state.code_path)
        )

        # 5. Query Decision Memory → Evidence(source_type="decision_unit")
        new_evidence.extend(self._gather_decision_memory(state.query))

        # Append all new evidence to existing evidence
        state.evidence = state.evidence + new_evidence
        state.current_node = NodeType.EXPLORE

        return state

    def _gather_git_history(self, tool_fn, code_path: str) -> list[Evidence]:
        """Call get_git_history and convert results to Evidence objects."""
        evidence: list[Evidence] = []
        try:
            result = tool_fn(code_path=code_path, mock=self._mock)
            if "error" in result:
                # Silently skip git errors (common for non-git folders)
                logger.debug("git_history skipped: %s", result["error"])
                return evidence

            for commit in result.get("commits", []):
                ev = Evidence(
                    source_type="git_commit",
                    source_id=commit.get("sha", ""),
                    content=commit.get("message", ""),
                    timestamp=commit.get("timestamp", None),
                    metadata={
                        "author": commit.get("author", ""),
                        "diff": commit.get("diff", ""),
                        "files_changed": commit.get("files_changed", []),
                    },
                )
                evidence.append(ev)
        except Exception as e:
            logger.error("Failed to gather git history: %s", e)

        return evidence

    def _gather_github_issues(self, tool_fn, search_term: str) -> list[Evidence]:
        """Call get_github_issues and convert results to Evidence objects."""
        evidence: list[Evidence] = []
        try:
            result = tool_fn(search_term=search_term, mock=self._mock)
            if "error" in result:
                logger.debug("github_issues skipped: %s", result["error"])
                return evidence

            for issue in result.get("issues", []):
                content_parts = [
                    f"#{issue.get('number', '')}: {issue.get('title', '')}",
                    issue.get("body", ""),
                ]
                # Include comments in the evidence content
                for comment in issue.get("comments", []):
                    content_parts.append(
                        f"[{comment.get('author', '')}]: {comment.get('body', '')}"
                    )

                ev = Evidence(
                    source_type="github_issue",
                    source_id=f"issue-{issue.get('number', '')}",
                    content="\n".join(content_parts),
                    timestamp=None,
                    metadata={
                        "author": issue.get("author", ""),
                        "state": issue.get("state", ""),
                        "labels": issue.get("labels", []),
                    },
                )
                evidence.append(ev)
        except Exception as e:
            logger.error("Failed to gather GitHub issues: %s", e)

        return evidence

    def _gather_pr_comments(self, tool_fn, code_path: str) -> list[Evidence]:
        """Call get_pr_comments and convert results to Evidence objects."""
        evidence: list[Evidence] = []
        try:
            result = tool_fn(code_path=code_path, mock=self._mock)
            if "error" in result:
                logger.debug("pr_comments skipped: %s", result["error"])
                return evidence

            for comment in result.get("pr_comments", []):
                ev = Evidence(
                    source_type="pr_comment",
                    source_id=f"pr-{comment.get('pr_number', '')}-{comment.get('timestamp', '')}",
                    content=comment.get("body", ""),
                    timestamp=comment.get("timestamp", None),
                    metadata={
                        "pr_number": comment.get("pr_number", 0),
                        "file_path": comment.get("file_path", ""),
                        "line_number": comment.get("line_number"),
                        "author": comment.get("author", ""),
                    },
                )
                evidence.append(ev)
        except Exception as e:
            logger.error("Failed to gather PR comments: %s", e)

        return evidence

    def _gather_related_changes(self, tool_fn, code_path: str) -> list[Evidence]:
        """Call get_related_changes and convert results to Evidence objects."""
        evidence: list[Evidence] = []
        try:
            result = tool_fn(code_path=code_path, mock=self._mock)
            if "error" in result:
                logger.debug("related_changes skipped: %s", result["error"])
                return evidence

            for related in result.get("related_files", []):
                ev = Evidence(
                    source_type="related_change",
                    source_id=f"related-{related.get('path', '')}",
                    content=f"Co-modified file: {related.get('path', '')} "
                    f"({related.get('co_commit_count', 0)} co-commits)",
                    timestamp=related.get("last_co_modified", None),
                    metadata={
                        "path": related.get("path", ""),
                        "co_commit_count": related.get("co_commit_count", 0),
                    },
                )
                evidence.append(ev)
        except Exception as e:
            logger.error("Failed to gather related changes: %s", e)

        return evidence

    def _gather_decision_memory(self, query: str) -> list[Evidence]:
        """Query Decision Memory vector store for relevant decision units."""
        evidence: list[Evidence] = []

        if self._vector_store is None or self._embedder is None:
            logger.info(
                "Decision Memory not configured (no vector_store or embedder); skipping."
            )
            return evidence

        try:
            query_embedding = self._embedder.embed_single(query)
            if not query_embedding:
                logger.warning("Empty embedding for query; skipping Decision Memory.")
                return evidence

            results = self._vector_store.query(embedding=query_embedding)

            for retrieval_result in results:
                unit = retrieval_result.decision_unit
                ev = Evidence(
                    source_type="decision_unit",
                    source_id=unit.id,
                    content=unit.content,
                    timestamp=unit.ingestion_timestamp,
                    metadata={
                        "source_document": unit.source_document,
                        "section_heading": unit.section_heading,
                        "similarity_score": retrieval_result.similarity_score,
                        "referenced_components": unit.referenced_components,
                    },
                )
                evidence.append(ev)
        except Exception as e:
            logger.error("Failed to query Decision Memory: %s", e)

        return evidence


class HypothesizeNode:
    """Generates candidate hypotheses from gathered evidence using the LLM.

    Uses GeminiService to produce 1-5 candidate explanations based on
    the evidence gathered during exploration.

    Args:
        gemini_service: GeminiService instance for LLM calls.
    """

    def __init__(self, gemini_service: GeminiService) -> None:
        self._gemini_service = gemini_service

    def execute(self, state: AgentState) -> AgentState:
        """Generate candidate hypotheses from evidence.

        Uses the LLM to analyze evidence and produce 1-5 candidate
        explanations. Each hypothesis includes references to supporting
        evidence.

        Args:
            state: The current reasoning loop state with gathered evidence.

        Returns:
            Updated AgentState with hypotheses list (1-5 items) and
            current_node set to NodeType.HYPOTHESIZE.
        """
        prompt = self._build_prompt(state)
        response = self._gemini_service.generate(prompt)
        hypotheses = self._parse_hypotheses(response, state)

        # Enforce bounds: at least 1, at most 5
        if not hypotheses:
            hypotheses = [
                Hypothesis(
                    id=str(uuid.uuid4()),
                    explanation="Unable to generate specific hypothesis from available evidence.",
                    supporting_evidence=[e.source_id for e in state.evidence[:3]],
                    confidence=0.1,
                )
            ]
        elif len(hypotheses) > 5:
            hypotheses = hypotheses[:5]

        return AgentState(
            query=state.query,
            code_path=state.code_path,
            loop_counter=state.loop_counter,
            remaining_iterations=state.remaining_iterations,
            evidence=state.evidence,
            hypotheses=hypotheses,
            confidence_score=state.confidence_score,
            conflicts=state.conflicts,
            synthesis=state.synthesis,
            current_node=NodeType.HYPOTHESIZE,
            is_incomplete=state.is_incomplete,
        )

    def _build_prompt(self, state: AgentState) -> str:
        """Build the hypothesis generation prompt.

        Includes the query, all gathered evidence, and any known conflicts
        to instruct the LLM to generate plausible explanations for WHY
        the code exists.
        """
        evidence_text = "\n".join(
            f"- [{e.source_type}] ({e.source_id}): {e.content[:400]}"
            for e in state.evidence
        )

        conflicts_text = ""
        if state.conflicts:
            conflict_lines = []
            for conflict in state.conflicts:
                sources_desc = "; ".join(
                    f"{s.source_id}: {s.claim}" for s in conflict.sources
                )
                conflict_lines.append(
                    f"- {conflict.description} (Sources: {sources_desc})"
                )
            conflicts_text = (
                "\n\n## Known Conflicts\n"
                "The following contradictions have been detected. "
                "Consider these when generating hypotheses:\n"
                + "\n".join(conflict_lines)
            )

        return f"""You are a code reasoning agent generating hypotheses about why code exists.

## Query
{state.query}

## Code Path
{state.code_path}

## Available Evidence
{evidence_text if evidence_text else "No evidence gathered yet."}
{conflicts_text}

## Instructions
Analyze the evidence above and generate 1-5 candidate hypotheses explaining WHY this code exists.
For each hypothesis:
1. Provide a clear explanation of the rationale
2. Rate your confidence in this hypothesis (0.0 to 1.0)
3. Reference specific evidence source_ids that support this hypothesis

Respond in JSON format:
```json
[
  {{
    "explanation": "Clear explanation of why the code exists",
    "confidence": 0.8,
    "supporting_evidence": ["source_id_1", "source_id_2"]
  }}
]
```

Generate at least 1 and at most 5 hypotheses. Rank them from most to least likely.
Return ONLY the JSON array."""

    def _parse_hypotheses(
        self, response: str, state: AgentState
    ) -> list[Hypothesis]:
        """Parse hypotheses from LLM response.

        Extracts 1-5 Hypothesis objects from the JSON response. Each
        hypothesis gets a uuid id, explanation, supporting_evidence
        (filtered to valid source_ids), and confidence clamped to [0, 1].
        """
        try:
            # Extract JSON from response (handle markdown code fences)
            json_match = re.search(
                r"```(?:json)?\s*(\[.*?\])\s*```", response, re.DOTALL
            )
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                # Try finding a JSON array directly
                bracket_start = response.find("[")
                bracket_end = response.rfind("]")
                if bracket_start != -1 and bracket_end > bracket_start:
                    data = json.loads(response[bracket_start : bracket_end + 1])
                else:
                    data = json.loads(response)

            if not isinstance(data, list):
                return []

            evidence_ids = {e.source_id for e in state.evidence}
            hypotheses: list[Hypothesis] = []

            for item in data:
                if not isinstance(item, dict):
                    continue
                explanation = item.get("explanation", "")
                if not explanation:
                    continue

                # Parse confidence, clamp to [0.0, 1.0]
                raw_confidence = item.get("confidence", 0.5)
                try:
                    confidence = max(0.0, min(1.0, float(raw_confidence)))
                except (TypeError, ValueError):
                    confidence = 0.5

                supporting = item.get("supporting_evidence", [])
                if not isinstance(supporting, list):
                    supporting = []
                # Filter to only valid evidence source_ids
                valid_supporting = [
                    s for s in supporting
                    if isinstance(s, str) and s in evidence_ids
                ]

                hypotheses.append(
                    Hypothesis(
                        id=str(uuid.uuid4()),
                        explanation=explanation,
                        supporting_evidence=valid_supporting,
                        confidence=confidence,
                    )
                )

            return hypotheses

        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning("Failed to parse hypotheses from LLM response")
            return []


class VerifyNode:
    """Validates hypotheses against available evidence using the LLM.

    Uses GeminiService to assess how well the gathered evidence supports
    each hypothesis, producing an overall confidence score in [0.0, 1.0].
    Optionally queries Decision Memory (VectorStore) for additional
    supporting or contradicting evidence.

    Args:
        gemini_service: GeminiService instance for LLM calls.
        vector_store: Optional VectorStore for additional RAG queries.
        embedder: Optional HuggingFaceEmbedder to embed hypothesis text
            for similarity search.
    """

    def __init__(
        self,
        gemini_service: GeminiService,
        vector_store: Optional[VectorStore] = None,
        embedder: Optional[HuggingFaceEmbedder] = None,
    ) -> None:
        self._gemini_service = gemini_service
        self._vector_store = vector_store
        self._embedder = embedder

    def _query_decision_memory(self, state: AgentState) -> list[Evidence]:
        """Query Decision Memory for additional evidence related to hypotheses.

        Embeds each hypothesis explanation and searches the vector store for
        relevant decision units that may support or contradict the hypotheses.

        Args:
            state: Current agent state containing hypotheses.

        Returns:
            List of new Evidence objects found from Decision Memory.
        """
        if not self._vector_store or not self._embedder:
            return []

        if not state.hypotheses:
            return []

        new_evidence: list[Evidence] = []
        existing_ids = {e.source_id for e in state.evidence}

        for hypothesis in state.hypotheses:
            # Embed the hypothesis explanation for semantic search
            embedding = self._embedder.embed_single(hypothesis.explanation)
            if not embedding:
                continue

            # Query vector store for relevant decision units
            results = self._vector_store.query(
                embedding=embedding,
                top_k=3,
                min_similarity=0.7,
            )

            for result in results:
                unit = result.decision_unit
                # Avoid duplicate evidence
                if unit.id in existing_ids:
                    continue
                existing_ids.add(unit.id)

                new_evidence.append(
                    Evidence(
                        source_type="decision_unit",
                        source_id=unit.id,
                        content=unit.content,
                        timestamp=unit.ingestion_timestamp,
                        metadata={
                            "source_document": unit.source_document,
                            "section_heading": unit.section_heading,
                            "similarity_score": result.similarity_score,
                            "related_hypothesis": hypothesis.id,
                        },
                    )
                )

        return new_evidence

    def _build_verification_prompt(self, state: AgentState) -> str:
        """Build the prompt for the LLM to verify hypotheses against evidence.

        Args:
            state: Current agent state with hypotheses and evidence.

        Returns:
            The formatted prompt string.
        """
        evidence_text = "\n".join(
            f"- [{e.source_type}] ({e.source_id}): {e.content[:500]}"
            for e in state.evidence
        )

        hypotheses_text = "\n".join(
            f"- Hypothesis {h.id}: {h.explanation} "
            f"(Supporting evidence: {', '.join(h.supporting_evidence)})"
            for h in state.hypotheses
        )

        prompt = f"""You are a code reasoning verification agent. Your task is to validate hypotheses against available evidence.

## Query
{state.query}

## Code Path
{state.code_path}

## Evidence Gathered
{evidence_text if evidence_text else "No evidence gathered."}

## Hypotheses to Verify
{hypotheses_text if hypotheses_text else "No hypotheses generated."}

## Instructions
For each hypothesis, assess how well the evidence supports it. Consider:
1. Does the evidence directly support or contradict the hypothesis?
2. Is there sufficient evidence to be confident in the hypothesis?
3. Are there gaps in the evidence that reduce confidence?

Respond in the following JSON format:
{{
    "hypothesis_scores": [
        {{"hypothesis_id": "<id>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}},
        ...
    ],
    "overall_confidence": <0.0-1.0>,
    "verification_summary": "<brief summary of overall assessment>"
}}

The overall_confidence should reflect how well the evidence as a whole supports the best hypothesis.
Confidence of 0.0 means no evidence support, 1.0 means overwhelming evidence support.
Be calibrated: only give high confidence (>0.8) when evidence directly and clearly supports a hypothesis."""

        return prompt

    def _parse_confidence_score(self, response: str) -> float:
        """Parse the overall confidence score from the LLM response.

        Attempts to parse JSON from the response. Falls back to regex
        extraction if JSON parsing fails. Always clamps to [0.0, 1.0].

        Args:
            response: Raw LLM response text.

        Returns:
            Confidence score clamped to [0.0, 1.0].
        """
        # Try parsing as JSON first
        try:
            # Find JSON block in response (may be wrapped in markdown code fences)
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                # Try parsing the whole response as JSON
                data = json.loads(response)

            score = float(data.get("overall_confidence", 0.0))
            return max(0.0, min(1.0, score))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Fallback: extract confidence from text using regex patterns
        patterns = [
            r'"overall_confidence"\s*:\s*([0-9]*\.?[0-9]+)',
            r"overall.?confidence[:\s]+([0-9]*\.?[0-9]+)",
            r"confidence[:\s]+([0-9]*\.?[0-9]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    return max(0.0, min(1.0, score))
                except ValueError:
                    continue

        # If nothing could be parsed, return a conservative low-confidence score
        return 0.3

    def _update_hypothesis_confidence(
        self, state: AgentState, response: str
    ) -> list[Hypothesis]:
        """Update individual hypothesis confidence scores from the LLM response.

        Args:
            state: Current agent state with hypotheses.
            response: Raw LLM response text.

        Returns:
            Updated list of hypotheses with confidence scores.
        """
        updated_hypotheses = list(state.hypotheses)

        try:
            json_match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL
            )
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                data = json.loads(response)

            scores = data.get("hypothesis_scores", [])
            score_map = {
                s["hypothesis_id"]: max(0.0, min(1.0, float(s["confidence"])))
                for s in scores
                if "hypothesis_id" in s and "confidence" in s
            }

            for i, h in enumerate(updated_hypotheses):
                if h.id in score_map:
                    updated_hypotheses[i] = Hypothesis(
                        id=h.id,
                        explanation=h.explanation,
                        supporting_evidence=h.supporting_evidence,
                        confidence=score_map[h.id],
                    )
        except (json.JSONDecodeError, ValueError, TypeError, KeyError):
            # If parsing fails, leave hypothesis confidence unchanged
            pass

        return updated_hypotheses

    def execute(self, state: AgentState) -> AgentState:
        """Validate hypotheses against evidence and produce a confidence score.

        Steps:
        1. Optionally query Decision Memory for additional supporting/
           contradicting evidence.
        2. Build a verification prompt with all hypotheses and evidence.
        3. Use GeminiService to assess evidence support for each hypothesis.
        4. Parse overall confidence score (clamped to [0.0, 1.0]).
        5. Update state with confidence score and node marker.

        Args:
            state: The current reasoning loop state with hypotheses and evidence.

        Returns:
            Updated AgentState with:
            - confidence_score set to the overall score in [0.0, 1.0]
            - current_node set to NodeType.VERIFY
            - hypotheses updated with individual confidence scores
            - evidence potentially augmented with Decision Memory results
        """
        # Step 1: Query Decision Memory for additional evidence
        additional_evidence = self._query_decision_memory(state)
        all_evidence = list(state.evidence) + additional_evidence

        # Create state with augmented evidence for prompting
        augmented_state = AgentState(
            query=state.query,
            code_path=state.code_path,
            loop_counter=state.loop_counter,
            remaining_iterations=state.remaining_iterations,
            evidence=all_evidence,
            hypotheses=state.hypotheses,
            confidence_score=state.confidence_score,
            conflicts=state.conflicts,
            synthesis=state.synthesis,
            current_node=state.current_node,
            is_incomplete=state.is_incomplete,
        )

        # Step 2: Build the verification prompt
        prompt = self._build_verification_prompt(augmented_state)

        # Step 3: Call the LLM for verification
        response = self._gemini_service.generate(prompt)

        # Step 4: Parse confidence score — always clamp to [0.0, 1.0]
        confidence_score = self._parse_confidence_score(response)

        # Step 5: Update individual hypothesis confidence scores
        updated_hypotheses = self._update_hypothesis_confidence(state, response)

        # Return updated state
        return AgentState(
            query=state.query,
            code_path=state.code_path,
            loop_counter=state.loop_counter,
            remaining_iterations=state.remaining_iterations,
            evidence=all_evidence,
            hypotheses=updated_hypotheses,
            confidence_score=confidence_score,
            conflicts=state.conflicts,
            synthesis=state.synthesis,
            current_node=NodeType.VERIFY,
            is_incomplete=state.is_incomplete,
        )


class SynthesizeNode:
    """Produces the final plain-English explanation from accumulated state.

    Uses GeminiService to synthesize all hypotheses, evidence, and conflicts
    into a coherent answer explaining WHY the code exists. Unresolved conflicts
    are surfaced explicitly in the output — never hidden.

    The output format follows:
        WHY: <explanation>
        CONFIDENCE: <0.0-1.0>
        SOURCES: [citations]
        CONFLICTS: [if any]

    Args:
        gemini_service: GeminiService instance for LLM calls.
    """

    def __init__(self, gemini_service: GeminiService) -> None:
        self._gemini_service = gemini_service

    def _build_prompt(self, state: AgentState) -> str:
        """Build the synthesis prompt including query, hypotheses, evidence, and conflicts.

        Args:
            state: The current agent state with all accumulated data.

        Returns:
            A formatted prompt string for the LLM.
        """
        sections: list[str] = []

        # Original query
        sections.append(f"## Original Query\n{state.query}")
        sections.append(f"Code path: {state.code_path}")

        # Hypotheses with confidence scores
        sections.append("\n## Hypotheses")
        if state.hypotheses:
            for h in state.hypotheses:
                sections.append(
                    f"- [{h.id}] (confidence: {h.confidence:.2f}): {h.explanation}"
                )
                if h.supporting_evidence:
                    sections.append(
                        f"  Supporting evidence: {', '.join(h.supporting_evidence)}"
                    )
        else:
            sections.append("No hypotheses were generated.")

        # Evidence gathered
        sections.append("\n## Evidence")
        if state.evidence:
            for e in state.evidence:
                sections.append(
                    f"- [{e.source_id}] ({e.source_type}): {e.content}"
                )
        else:
            sections.append("No evidence was gathered.")

        # Unresolved conflicts — surfaced explicitly, not hidden
        sections.append("\n## Unresolved Conflicts")
        if state.conflicts:
            for c in state.conflicts:
                sections.append(f"- Conflict [{c.id}]: {c.description}")
                for src in c.sources:
                    sections.append(f"    Source {src.source_id}: {src.claim}")
        else:
            sections.append("No conflicts detected.")

        # Synthesis instructions
        sections.append("\n## Instructions")
        sections.append(
            "Based on the above hypotheses, evidence, and conflicts, produce a final "
            "plain-English explanation of WHY this code exists. Include:\n"
            "1. A clear explanation of the code's purpose and rationale\n"
            "2. An overall confidence score (0.0-1.0) reflecting how well-supported "
            "the answer is\n"
            "3. Citations to the evidence sources that support your explanation\n"
            "4. Any unresolved conflicts that the developer should be aware of\n\n"
            "Format your response EXACTLY as:\n"
            "WHY: <your explanation>\n"
            "CONFIDENCE: <score between 0.0 and 1.0>\n"
            "SOURCES: [list of source_ids used]\n"
            "CONFLICTS: [list any unresolved conflicts, or 'none']"
        )

        return "\n".join(sections)

    def _parse_response(
        self, response: str, state: AgentState
    ) -> SynthesisResult:
        """Parse the LLM response into a SynthesisResult.

        Falls back to reasonable defaults if the LLM response doesn't strictly
        follow the expected format.

        Args:
            response: Raw LLM response text.
            state: The current agent state for extracting context.

        Returns:
            A populated SynthesisResult.
        """
        answer = response
        confidence = state.confidence_score
        supporting_evidence: list[str] = []

        # Attempt to parse structured format
        lines = response.strip().split("\n")
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.upper().startswith("WHY:"):
                answer = line_stripped[4:].strip()
            elif line_stripped.upper().startswith("CONFIDENCE:"):
                try:
                    raw_conf = line_stripped[len("CONFIDENCE:"):].strip()
                    parsed_conf = float(raw_conf)
                    confidence = max(0.0, min(1.0, parsed_conf))
                except (ValueError, IndexError):
                    pass
            elif line_stripped.upper().startswith("SOURCES:"):
                raw_sources = line_stripped[len("SOURCES:"):].strip()
                # Parse source list: "[src1, src2]" or "src1, src2"
                raw_sources = raw_sources.strip("[]")
                if raw_sources and raw_sources.lower() != "none":
                    supporting_evidence = [
                        s.strip().strip("'\"")
                        for s in raw_sources.split(",")
                        if s.strip()
                    ]

        # If no sources were parsed from the response, use evidence source_ids
        if not supporting_evidence and state.evidence:
            supporting_evidence = [e.source_id for e in state.evidence]

        # If answer wasn't parsed cleanly (no WHY: prefix found), use full response
        if answer == response and "WHY:" not in response.upper():
            answer = response.strip()

        # Determine reasoning path from state traversal
        reasoning_path = self._build_reasoning_path(state)

        # Determine if reasoning was incomplete
        is_incomplete = state.is_incomplete or self._is_reasoning_incomplete(state)

        return SynthesisResult(
            answer=answer,
            confidence=confidence,
            supporting_evidence=supporting_evidence,
            conflicts=state.conflicts,
            reasoning_path=reasoning_path,
            is_incomplete=is_incomplete,
        )

    def _build_reasoning_path(self, state: AgentState) -> list[NodeType]:
        """Build the reasoning path based on state indicators.

        Reconstructs the sequence of nodes visited during reasoning based on
        the loop counter and available state data.

        Args:
            state: The current agent state.

        Returns:
            List of NodeType values representing the reasoning path.
        """
        path: list[NodeType] = []

        # Each full cycle visits: explore → hypothesize → verify → check_contradictions
        cycles_completed = state.loop_counter
        for _ in range(cycles_completed):
            path.extend([
                NodeType.EXPLORE,
                NodeType.HYPOTHESIZE,
                NodeType.VERIFY,
                NodeType.CHECK_CONTRADICTIONS,
            ])

        # Always end with synthesize since we're in the synthesize node
        path.append(NodeType.SYNTHESIZE)

        return path

    def _is_reasoning_incomplete(self, state: AgentState) -> bool:
        """Determine if reasoning was cut short.

        Reasoning is considered incomplete if:
        - Max loops reached with low confidence (< 0.7)
        - An error occurred during processing (state.is_incomplete already set)

        Args:
            state: The current agent state.

        Returns:
            True if reasoning was cut short.
        """
        max_loops = 3
        low_confidence_threshold = 0.7

        # Max loops reached with low confidence
        if (
            state.loop_counter >= max_loops
            and state.confidence_score < low_confidence_threshold
        ):
            return True

        return False

    def execute(self, state: AgentState) -> AgentState:
        """Produce the final synthesis using GeminiService.

        Builds a comprehensive prompt including the original query, all hypotheses
        with confidence scores, all evidence gathered, and all unresolved conflicts
        (surfaced explicitly, not hidden). Creates a SynthesisResult and stores
        it in state.

        Args:
            state: The current agent state with accumulated reasoning data.

        Returns:
            Updated AgentState with synthesis populated and current_node set
            to NodeType.SYNTHESIZE.
        """
        prompt = self._build_prompt(state)

        try:
            response = self._gemini_service.generate(prompt)
        except Exception:
            # On LLM failure, produce a synthesis from available data
            response = self._fallback_synthesis(state)
            state.is_incomplete = True

        synthesis = self._parse_response(response, state)
        state.synthesis = synthesis
        state.current_node = NodeType.SYNTHESIZE

        return state

    def _fallback_synthesis(self, state: AgentState) -> str:
        """Generate a fallback answer when the LLM call fails.

        Uses the highest-confidence hypothesis as the answer, or indicates
        that insufficient data was gathered.

        Args:
            state: The current agent state.

        Returns:
            A formatted fallback response string.
        """
        if state.hypotheses:
            best = max(state.hypotheses, key=lambda h: h.confidence)
            answer = best.explanation
            confidence = best.confidence
            sources = ", ".join(best.supporting_evidence) if best.supporting_evidence else "none"
        else:
            answer = (
                "Unable to determine why this code exists due to insufficient "
                "evidence and LLM service failure."
            )
            confidence = 0.0
            sources = "none"

        conflicts = "none"
        if state.conflicts:
            conflict_strs = [c.description for c in state.conflicts]
            conflicts = "; ".join(conflict_strs)

        return (
            f"WHY: {answer}\n"
            f"CONFIDENCE: {confidence}\n"
            f"SOURCES: [{sources}]\n"
            f"CONFLICTS: [{conflicts}]"
        )
