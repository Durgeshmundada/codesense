"""CheckContradictionsNode — detects conflicting claims between sources/hypotheses.

This node analyzes gathered evidence and hypotheses for contradictions.
When contradictions are found, Conflict objects are created with 2+ sources,
each having a source_id and claim. By design, conflicts have NO winner field,
NO resolution indicator, and NO ranking of claims — they surface ambiguity
honestly so developers can decide which source to trust.

Requirements: 1.4, 7.1, 7.2, 7.5
"""

import json
import logging
import uuid
from typing import Any

from codesense.llm.gemini_service import GeminiService
from codesense.models.state import (
    AgentState,
    Conflict,
    ConflictSource,
    Evidence,
    Hypothesis,
    NodeType,
)

logger = logging.getLogger(__name__)

_CONTRADICTION_PROMPT_TEMPLATE = """You are analyzing evidence sources and hypotheses for contradictions.

A contradiction exists when:
- Two or more evidence sources make conflicting claims about the same subject
- Evidence directly contradicts one or more of the top hypotheses

Your task:
1. Compare all evidence sources against each other — identify any pairs (or groups) that make contradicting claims about the same topic.
2. Compare the evidence against the hypotheses — identify any evidence that directly contradicts the top hypothesis.

IMPORTANT CONSTRAINTS:
- Do NOT resolve contradictions or pick a winner.
- Do NOT rank the claims or suggest which is more likely correct.
- Simply identify and describe the contradictions neutrally.

If NO contradictions are found, return an empty JSON array: []

If contradictions ARE found, return a JSON array of contradiction objects:
```json
[
  {{
    "description": "Brief description of what the contradiction is about",
    "sources": [
      {{
        "source_id": "the identifier of the first source",
        "claim": "what this source claims"
      }},
      {{
        "source_id": "the identifier of the second source",
        "claim": "what this source claims (contradicting the first)"
      }}
    ]
  }}
]
```

Each contradiction must have at least 2 sources. You may include more if multiple sources are involved in the same contradiction.

--- EVIDENCE ---
{evidence_text}

--- HYPOTHESES ---
{hypotheses_text}

Return ONLY the JSON array. No other text.
"""


class CheckContradictionsNode:
    """Detects conflicting claims between evidence sources and hypotheses.

    Uses GeminiService to analyze gathered evidence and hypotheses for
    contradictions. Produces Conflict objects containing 2+ ConflictSource
    entries, each with a source_id and claim.

    KEY CONSTRAINT: No winner field, no resolution indicator, no ranking
    of claims. Conflicts surface ambiguity honestly (Requirement 7.2).

    Args:
        gemini_service: GeminiService instance for LLM calls.
    """

    def __init__(self, gemini_service: GeminiService) -> None:
        self._gemini_service = gemini_service

    def execute(self, state: AgentState) -> AgentState:
        """Analyze evidence and hypotheses for contradictions.

        Sends evidence and hypotheses to the LLM to identify contradictions.
        When contradictions are found, Conflict objects are created and
        appended to state.conflicts. When no contradictions are found,
        state.conflicts remains unchanged.

        Args:
            state: Current agent state with evidence and hypotheses.

        Returns:
            Updated AgentState with any new Conflict objects appended
            and current_node set to NodeType.CHECK_CONTRADICTIONS.
        """
        state.current_node = NodeType.CHECK_CONTRADICTIONS

        # If there's no evidence or hypotheses, nothing to check
        if not state.evidence and not state.hypotheses:
            return state

        try:
            prompt = self._build_prompt(state)
            llm_response = self._gemini_service.generate(prompt)
            new_conflicts = self._parse_conflicts(llm_response)

            if new_conflicts:
                state.conflicts = state.conflicts + new_conflicts

        except Exception as e:
            logger.error("Failed to check contradictions: %s", e)
            # On failure, proceed without adding conflicts (graceful degradation)

        return state

    def _build_prompt(self, state: AgentState) -> str:
        """Build the contradiction-detection prompt from state.

        Formats evidence and hypotheses into a structured prompt
        that asks the LLM to identify contradictions without resolving them.

        Args:
            state: Current agent state.

        Returns:
            Formatted prompt string.
        """
        evidence_text = self._format_evidence(state.evidence)
        hypotheses_text = self._format_hypotheses(state.hypotheses)

        return _CONTRADICTION_PROMPT_TEMPLATE.format(
            evidence_text=evidence_text,
            hypotheses_text=hypotheses_text,
        )

    def _format_evidence(self, evidence: list[Evidence]) -> str:
        """Format evidence list into a readable string for the LLM.

        Args:
            evidence: List of Evidence objects.

        Returns:
            Formatted string with each evidence item labeled.
        """
        if not evidence:
            return "(No evidence gathered)"

        parts: list[str] = []
        for ev in evidence:
            parts.append(
                f"[{ev.source_type}] ID: {ev.source_id}\n"
                f"Content: {ev.content[:500]}"
            )
        return "\n\n".join(parts)

    def _format_hypotheses(self, hypotheses: list[Hypothesis]) -> str:
        """Format hypotheses list into a readable string for the LLM.

        Args:
            hypotheses: List of Hypothesis objects.

        Returns:
            Formatted string with each hypothesis labeled.
        """
        if not hypotheses:
            return "(No hypotheses generated)"

        parts: list[str] = []
        for i, hyp in enumerate(hypotheses, 1):
            parts.append(
                f"Hypothesis {i} (confidence: {hyp.confidence:.2f}):\n"
                f"{hyp.explanation}\n"
                f"Supporting evidence: {', '.join(hyp.supporting_evidence)}"
            )
        return "\n\n".join(parts)

    def _parse_conflicts(self, llm_response: str) -> list[Conflict]:
        """Parse the LLM response into Conflict objects.

        Extracts JSON from the response, validates the structure,
        and creates Conflict objects with proper ConflictSource entries.

        Args:
            llm_response: Raw text response from the LLM.

        Returns:
            List of Conflict objects. Empty if no contradictions found
            or if parsing fails.
        """
        json_text = self._extract_json(llm_response)

        try:
            parsed = json.loads(json_text)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse contradiction JSON: %s", e)
            return []

        if not isinstance(parsed, list):
            logger.warning("LLM response is not a JSON array, treating as no conflicts")
            return []

        conflicts: list[Conflict] = []
        for item in parsed:
            conflict = self._create_conflict_from_dict(item)
            if conflict is not None:
                conflicts.append(conflict)

        return conflicts

    def _create_conflict_from_dict(self, item: Any) -> Conflict | None:
        """Create a Conflict object from a parsed dictionary.

        Validates that the item has the required structure: a description
        and at least 2 sources, each with source_id and claim.

        Args:
            item: Parsed dictionary from LLM JSON output.

        Returns:
            Conflict object if valid, None if the item is invalid.
        """
        if not isinstance(item, dict):
            return None

        description = item.get("description", "")
        if not description:
            return None

        raw_sources = item.get("sources", [])
        if not isinstance(raw_sources, list) or len(raw_sources) < 2:
            return None

        conflict_sources: list[ConflictSource] = []
        for src in raw_sources:
            if not isinstance(src, dict):
                continue
            source_id = src.get("source_id", "")
            claim = src.get("claim", "")
            if source_id and claim:
                conflict_sources.append(
                    ConflictSource(source_id=source_id, claim=claim)
                )

        # Must have at least 2 valid sources for a valid conflict
        if len(conflict_sources) < 2:
            return None

        return Conflict(
            id=str(uuid.uuid4()),
            sources=conflict_sources,
            description=description,
        )

    def _extract_json(self, text: str) -> str:
        """Extract JSON content from LLM response text.

        Handles responses wrapped in code fences or raw JSON.

        Args:
            text: Raw LLM response text.

        Returns:
            Extracted JSON string.
        """
        # Try to find JSON within code fences
        if "```json" in text:
            start = text.index("```json") + len("```json")
            end = text.index("```", start)
            return text[start:end].strip()

        if "```" in text:
            start = text.index("```") + len("```")
            end = text.index("```", start)
            return text[start:end].strip()

        # Try to find a JSON array directly
        bracket_start = text.find("[")
        bracket_end = text.rfind("]")
        if bracket_start != -1 and bracket_end != -1 and bracket_end > bracket_start:
            return text[bracket_start : bracket_end + 1]

        return text.strip()
