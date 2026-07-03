"""Natural language query intent classification and parameter extraction.

Classifies user queries into one of 10 intent categories and extracts
relevant parameters for routing to the appropriate capability handler.
Uses GeminiService for LLM-based classification with a keyword-based
fallback when the LLM is unavailable or for high-confidence matches.

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from codesense.models.output import CommandOutput, CommandParams

logger = logging.getLogger(__name__)

# Valid intent set for classification
VALID_INTENTS = frozenset({
    "explain", "describe", "tree", "flow", "diagram",
    "trace", "deps", "related", "risk", "onboard",
})

# Default confidence threshold for auto-routing
DEFAULT_CONFIDENCE_THRESHOLD = 0.6

# Keyword patterns mapping to intents for fallback classification.
KEYWORD_PATTERNS: list[tuple[list[str], str]] = [
    (["why", "explain", "history"], "explain"),
    (["what", "describe", "how does"], "describe"),
    (["tree", "structure", "folders"], "tree"),
    (["flow", "execution", "sequence"], "flow"),
    (["diagram", "draw", "visualize"], "diagram"),
    (["trace", "when", "timeline"], "trace"),
    (["depend", "import", "env", "api"], "deps"),
    (["related", "affects", "impact"], "related"),
    (["risk", "safe", "touch", "delete"], "risk"),
    (["onboard", "guide", "overview"], "onboard"),
]

# --- Regex patterns for parameter extraction (Requirements 11.2) ---

# File paths: quoted, backtick, or unquoted with directory separators/extensions
_FILE_PATH_PATTERNS = [
    # Quoted paths (single or double quotes)
    re.compile(r"""['"]([^'"]*?[\w]\.[\w]{1,10})['"]"""),
    # Backtick paths
    re.compile(r"""`([^`]*?[\w]\.[\w]{1,10})`"""),
    # Unquoted relative paths with dot prefix: ./path/to/file.ext
    re.compile(r"""(?<!\w)(\.{1,2}/(?:[\w.-]+/)*[\w.-]+\.[\w]{1,10})\b"""),
    # Unquoted paths with at least one directory separator: dir/file.ext
    re.compile(r"""(?<!\w)([\w][\w.-]*/(?:[\w.-]+/)*[\w.-]+\.[\w]{1,10})\b"""),
    # Simple filename with extension: file.ext (must not be preceded by a slash to avoid partial matches)
    re.compile(r"""(?<![/\\\w])([\w][\w-]*\.[\w]{1,10})\b"""),
]

# Function/method names after keywords
_FUNCTION_NAME_PATTERNS = [
    # Natural-language "the X function/method/class" — checked FIRST so that
    # phrasings like "how does the get_next_key function work" extract
    # 'get_next_key' rather than the trailing verb ('work').
    re.compile(r"""\bthe\s+(\w+)\s+(?:function|method|class)\b""", re.IGNORECASE),
    # "function X", "def X", "method X", "func X"
    re.compile(r"""\b(?:function|def|method|func)\s+(\w+)""", re.IGNORECASE),
    # Backtick-quoted identifiers (e.g. `my_func`)
    re.compile(r"""`(\w+)`"""),
    # Identifier followed by parentheses: some_func()
    re.compile(r"""\b(\w+)\(\)"""),
]

# Line numbers: "line 44", "L44", ":44", "at line 44"
_LINE_NUMBER_PATTERNS = [
    re.compile(r"""\blines?\s*(?:number\s*)?#?\s*(\d+)""", re.IGNORECASE),
    re.compile(r"""\bL(\d+)\b"""),
    re.compile(r""":(\d+)\b"""),
    re.compile(r"""\bat\s+(?:line\s+)?(\d+)\b""", re.IGNORECASE),
]

# Feature names: after "feature" keyword or in quotes
_FEATURE_NAME_PATTERNS = [
    re.compile(r"""\bfeature\s+['"]([^'"]+)['"]""", re.IGNORECASE),
    re.compile(r"""\bfeature\s+([\w][\w\s-]*)""", re.IGNORECASE),
]

# Known file extensions for validation
_KNOWN_EXTENSIONS = frozenset({
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".rb", ".go", ".rs",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".php", ".swift", ".kt", ".scala",
    ".sh", ".bash", ".zsh", ".yaml", ".yml", ".json", ".toml", ".md",
    ".sql", ".html", ".css", ".xml", ".txt", ".cfg", ".ini", ".env",
})


def _is_valid_file_path(candidate: str) -> bool:
    """Check if a candidate string looks like a valid file path.

    Args:
        candidate: String to validate.

    Returns:
        True if the string appears to be a file path with a known extension.
    """
    if not candidate or len(candidate) < 3:
        return False
    if "." not in candidate:
        return False

    # Get extension
    last_dot = candidate.rfind(".")
    ext = candidate[last_dot:].lower()
    # Strip trailing punctuation
    ext = ext.rstrip(".,;:!?)")

    return ext in _KNOWN_EXTENSIONS


# LLM classification prompt template
CLASSIFICATION_PROMPT = """You are an intent classifier for a codebase understanding tool.
Given a user's natural language query, classify it into exactly ONE of these intents:

- explain: User wants to know WHY code exists, its history, or rationale
- describe: User wants to know WHAT code does (high-level description)
- tree: User wants to see project/directory structure
- flow: User wants to see execution flow, call sequences, or how code runs
- diagram: User wants a visual diagram of relationships (class, module, dependency)
- trace: User wants a timeline of changes (commits, issues, PRs) for specific code
- deps: User wants dependency information (packages, env vars, APIs, imports)
- related: User wants to find related/dependent files or impact analysis
- risk: User wants risk assessment or safety evaluation of code changes
- onboard: User wants an overview/guide to understand a project or module

Also extract any parameters from the query:
- file_path: Any file or directory path mentioned
- function_name: Any function or method name mentioned
- line_number: Any specific line number mentioned
- feature_name: Any feature or module name mentioned

Respond ONLY with valid JSON in this exact format:
{{"intent": "<one of the 10 intents above>", "confidence": <float between 0.0 and 1.0>, "params": {{"file_path": "<extracted path or null>", "function_name": "<extracted function or null>", "line_number": <extracted line number or null>, "feature_name": "<extracted feature or null>"}}}}

User query: {query}"""


class IntentClassifier:
    """Classifies natural language queries into capability intents.

    Uses a two-tier classification approach:
    1. Keyword-based fast-path for queries with clear intent signals
    2. LLM-based classification (via GeminiService) for ambiguous queries

    Also provides extract_params() for extracting file paths, function names,
    line numbers, and feature names from natural language queries.

    Args:
        gemini_service: Optional GeminiService instance. If None, only keyword
            classification is available.
        confidence_threshold: Minimum confidence to auto-route. Below this,
            top candidates are returned for user selection. Defaults to 0.6.
    """

    def __init__(
        self,
        gemini_service=None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._gemini_service = gemini_service
        self._confidence_threshold = confidence_threshold

    def classify(self, query: str) -> dict[str, Any]:
        """Classify a natural language query into an intent with parameters.

        Args:
            query: The natural language query from the user.

        Returns:
            A dict with keys:
                - "intent": str | None -- the classified intent from VALID_INTENTS
                - "confidence": float -- confidence score [0.0, 1.0]
                - "params": CommandParams -- extracted parameters
                - "candidates": list[dict] | None -- top candidates if confidence < threshold
                - "error": str | None -- error message if query is invalid

        Raises:
            ValueError: If query is empty or whitespace-only.
        """
        # Handle empty/uninterpretable queries
        if not query or not query.strip():
            raise ValueError(
                "Query is empty or cannot be interpreted. "
                "Please rephrase or provide more detail."
            )

        cleaned_query = query.strip()

        # Try keyword-based classification first
        keyword_result = self._classify_by_keywords(cleaned_query)
        if keyword_result is not None:
            return keyword_result

        # Fall back to LLM-based classification
        try:
            llm_result = self._classify_by_llm(cleaned_query)
            return llm_result
        except Exception as e:
            logger.warning("LLM classification failed: %s. Using fallback.", e)
            # Ultimate fallback: route to explain (general reasoning loop)
            return {
                "intent": "explain",
                "confidence": 0.3,
                "params": self.extract_params(cleaned_query, "explain"),
                "candidates": None,
                "error": None,
            }

    def extract_params(self, query: str, intent: str) -> CommandParams:
        """Extract parameters from a natural language query.

        Uses regex patterns to find:
        - File paths: patterns like `src/auth.py`, `path/to/file.py`, etc.
        - Function names: words after "function", "def", "method" keywords
        - Line numbers: patterns like "line 44", "L44", ":44"
        - Feature names: words in quotes or after "feature"

        Also uses the LLM response (from classify) which should include
        extracted params in its JSON when available.

        Args:
            query: The natural language query string.
            intent: The classified intent (used to prioritize extraction).

        Returns:
            CommandParams with extracted parameters populated.
        """
        file_path = self._extract_file_path(query)
        function_name = self._extract_function_name(query)
        line_number = self._extract_line_number(query)
        feature_name = self._extract_feature_name(query)

        params = CommandParams(
            path=file_path,
            query=query,
            function_name=function_name,
            line_number=line_number,
        )

        # Store feature name in output field when no file path found
        if feature_name and not file_path:
            params.output = feature_name

        return params

    def _extract_file_path(self, query: str) -> Optional[str]:
        """Extract a file path from the query using regex patterns.

        Tries quoted paths first, then backtick paths, then unquoted paths.
        Validates that the extracted path has a known file extension.

        Args:
            query: The natural language query string.

        Returns:
            Extracted file path, or None if no valid path found.
        """
        for pattern in _FILE_PATH_PATTERNS:
            matches = pattern.findall(query)
            for match in matches:
                if _is_valid_file_path(match):
                    return match
        return None

    def _extract_function_name(self, query: str) -> Optional[str]:
        """Extract a function/method name from the query.

        Looks for names after keywords like "function", "def", "method",
        or identifiers in backticks or followed by parentheses.

        Args:
            query: The natural language query string.

        Returns:
            Extracted function name, or None if not found.
        """
        # Common words to filter out as false positives
        stop_words = frozenset({
            "the", "a", "an", "this", "that", "it", "in", "on", "at",
            "to", "for", "of", "is", "are", "was", "be", "do", "does",
            # Common verbs that follow "function"/"method" in questions
            "work", "works", "working", "run", "runs", "exist", "exists",
            "behave", "behaves", "operate", "operates",
        })

        for pattern in _FUNCTION_NAME_PATTERNS:
            match = pattern.search(query)
            if match:
                name = match.group(1).strip()
                if name.lower() not in stop_words and len(name) > 1:
                    # Don't return names that look like file paths
                    if "/" not in name and "\\" not in name:
                        # If it has a dot, verify it's not a file path
                        if "." in name and _is_valid_file_path(name):
                            continue
                        return name
        return None

    def _extract_line_number(self, query: str) -> Optional[int]:
        """Extract a line number from the query.

        Matches patterns like "line 44", "L44", ":44", "at 44".

        Args:
            query: The natural language query string.

        Returns:
            Extracted line number as integer, or None if not found.
        """
        for pattern in _LINE_NUMBER_PATTERNS:
            match = pattern.search(query)
            if match:
                try:
                    num = int(match.group(1))
                    # Sanity check: line numbers should be reasonable
                    if 1 <= num <= 100000:
                        return num
                except (ValueError, IndexError):
                    continue
        return None

    def _extract_feature_name(self, query: str) -> Optional[str]:
        """Extract a feature name from the query.

        Looks for names after the word "feature" or in quotes.

        Args:
            query: The natural language query string.

        Returns:
            Extracted feature name, or None if not found.
        """
        for pattern in _FEATURE_NAME_PATTERNS:
            match = pattern.search(query)
            if match:
                name = match.group(1).strip()
                if len(name) >= 2:
                    return name

        # Also check for double-quoted strings as feature names
        quote_match = re.search(r'"([^"]{2,})"', query)
        if quote_match:
            name = quote_match.group(1)
            # Don't return if it's already identified as a file path
            if not _is_valid_file_path(name):
                return name

        return None

    def _classify_by_keywords(self, query: str) -> Optional[dict[str, Any]]:
        """Attempt keyword-based intent classification.

        Args:
            query: Cleaned query string.

        Returns:
            Classification result dict if a clear match is found, else None.
        """
        query_lower = query.lower()

        scores: dict[str, int] = {}
        for keywords, intent in KEYWORD_PATTERNS:
            match_count = sum(1 for kw in keywords if kw in query_lower)
            if match_count > 0:
                scores[intent] = match_count

        if not scores:
            return None

        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_intent, top_score = sorted_intents[0]

        # Only return keyword result if there's a clear winner
        if len(sorted_intents) == 1 or top_score > sorted_intents[1][1]:
            confidence = min(0.7 + (top_score - 1) * 0.1, 0.95)
            return {
                "intent": top_intent,
                "confidence": confidence,
                "params": self.extract_params(query, top_intent),
                "candidates": None,
                "error": None,
            }

        return None

    def _classify_by_llm(self, query: str) -> dict[str, Any]:
        """Classify a query using the GeminiService LLM.

        Args:
            query: Cleaned query string.

        Returns:
            Classification result dict.

        Raises:
            RuntimeError: If GeminiService is unavailable.
        """
        if self._gemini_service is None:
            raise RuntimeError("No GeminiService configured for LLM classification.")

        prompt = CLASSIFICATION_PROMPT.format(query=query)
        raw_response = self._gemini_service.generate(prompt)

        return self._parse_llm_response(raw_response, query)

    def _parse_llm_response(self, response: str, original_query: str) -> dict[str, Any]:
        """Parse the LLM classification response into a structured result.

        Merges LLM-extracted params with regex-extracted params (regex fills gaps).

        Args:
            response: Raw response string from the LLM.
            original_query: The original query for parameter extraction fallback.

        Returns:
            Classification result dict.
        """
        # Strip markdown code fences if present
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    return self._fallback_result(original_query)
            else:
                return self._fallback_result(original_query)

        # Validate intent
        intent = data.get("intent", "explain")
        if intent not in VALID_INTENTS:
            intent = "explain"

        # Validate confidence
        confidence = data.get("confidence", 0.5)
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5

        # Extract LLM params
        params_data = data.get("params", {})
        if not isinstance(params_data, dict):
            params_data = {}

        # Build CommandParams from LLM response
        llm_file_path = params_data.get("file_path")
        llm_function_name = params_data.get("function_name")
        llm_line_number = params_data.get("line_number")
        llm_feature_name = params_data.get("feature_name")

        # Normalize nulls
        if llm_file_path in (None, "null", ""):
            llm_file_path = None
        if llm_function_name in (None, "null", ""):
            llm_function_name = None
        if llm_feature_name in (None, "null", ""):
            llm_feature_name = None
        if llm_line_number in (None, "null", ""):
            llm_line_number = None
        elif llm_line_number is not None:
            try:
                llm_line_number = int(llm_line_number)
            except (TypeError, ValueError):
                llm_line_number = None

        # Get regex-extracted params to fill gaps
        regex_params = self.extract_params(original_query, intent)

        # Merge: LLM takes priority, regex fills gaps
        merged_params = CommandParams(
            path=llm_file_path or regex_params.path,
            query=original_query,
            function_name=llm_function_name or regex_params.function_name,
            line_number=llm_line_number or regex_params.line_number,
            output=(llm_feature_name if llm_feature_name and not llm_file_path
                    else regex_params.output),
        )

        # Build candidates if low confidence
        candidates = None
        if confidence < self._confidence_threshold:
            candidates = self._get_top_candidates(original_query, intent, confidence)

        return {
            "intent": intent,
            "confidence": confidence,
            "params": merged_params,
            "candidates": candidates,
            "error": None,
        }

    def _get_top_candidates(
        self, query: str, top_intent: str, top_confidence: float
    ) -> list[dict[str, Any]]:
        """Generate top candidate intents for low-confidence classifications.

        Args:
            query: The original query.
            top_intent: The top-classified intent.
            top_confidence: The confidence of the top intent.

        Returns:
            List of candidate dicts with 'intent' and 'confidence' keys.
        """
        candidates = [{"intent": top_intent, "confidence": top_confidence}]

        query_lower = query.lower()
        scored: list[tuple[str, float]] = []

        for keywords, intent in KEYWORD_PATTERNS:
            if intent == top_intent:
                continue
            match_count = sum(1 for kw in keywords if kw in query_lower)
            if match_count > 0:
                score = min(0.3 + match_count * 0.1, top_confidence - 0.05)
                scored.append((intent, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        for intent, score in scored[:2]:
            candidates.append({"intent": intent, "confidence": round(score, 2)})

        if not any(c["intent"] == "explain" for c in candidates):
            candidates.append({"intent": "explain", "confidence": 0.2})

        return candidates[:4]

    def _fallback_result(self, query: str) -> dict[str, Any]:
        """Generate a fallback classification result.

        Args:
            query: The original query for parameter extraction.

        Returns:
            Classification result dict with explain intent and low confidence.
        """
        return {
            "intent": "explain",
            "confidence": 0.3,
            "params": self.extract_params(query, "explain"),
            "candidates": None,
            "error": None,
        }


class AskHandler:
    """Capability handler for the 'ask' command.

    Routes natural language queries through the IntentClassifier and
    dispatches to the appropriate capability handler based on the
    classified intent.

    Args:
        gemini_service: Optional GeminiService instance.
        confidence_threshold: Minimum confidence for auto-routing. Defaults to 0.6.
    """

    def __init__(
        self,
        gemini_service=None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._classifier = IntentClassifier(
            gemini_service=gemini_service,
            confidence_threshold=confidence_threshold,
        )

    def run(self, params: CommandParams) -> CommandOutput:
        """Execute the ask capability -- classify intent and route.

        Args:
            params: Parsed CLI arguments. Must include `query`.

        Returns:
            CommandOutput with classification result and routing info.
        """
        query = params.query or ""
        is_demo = params.mock

        # Handle empty query
        if not query.strip():
            return CommandOutput(
                title="CodeSense — Ask",
                content=(
                    "**Error:** Empty query received. Please provide a question "
                    "about your codebase, e.g., 'why does auth.py exist?' or "
                    "'show me the project structure'."
                ),
                code_snippets=[],
                tables=[],
                conflicts=[],
                confidence=0.0,
                is_demo_mode=is_demo,
            )

        # Classify the query
        try:
            result = self._classifier.classify(query)
        except ValueError as e:
            return CommandOutput(
                title="CodeSense — Ask",
                content=f"**Error:** {e}",
                code_snippets=[],
                tables=[],
                conflicts=[],
                confidence=0.0,
                is_demo_mode=is_demo,
            )

        intent = result["intent"]
        confidence = result["confidence"]
        extracted_params = result.get("params")
        candidates = result.get("candidates")

        # Build response content
        content_parts = []
        content_parts.append(f"**Classified intent:** `{intent}` (confidence: {confidence:.2f})")

        if isinstance(extracted_params, CommandParams):
            param_strs = []
            if extracted_params.path:
                param_strs.append(f"  - file_path: `{extracted_params.path}`")
            if extracted_params.function_name:
                param_strs.append(f"  - function_name: `{extracted_params.function_name}`")
            if extracted_params.line_number is not None:
                param_strs.append(f"  - line_number: `{extracted_params.line_number}`")
            if extracted_params.output:
                param_strs.append(f"  - feature_name: `{extracted_params.output}`")
            if param_strs:
                content_parts.append("\n**Extracted parameters:**")
                content_parts.extend(param_strs)

        if candidates:
            content_parts.append(
                "\n**Low confidence.** Please select the intended action:"
            )
            for i, candidate in enumerate(candidates, 1):
                content_parts.append(
                    f"  {i}. `{candidate['intent']}` (confidence: {candidate['confidence']:.2f})"
                )

        content_parts.append(f"\n→ Routing to **{intent}** handler.")

        return CommandOutput(
            title="CodeSense — Ask",
            content="\n".join(content_parts),
            code_snippets=[],
            tables=[],
            conflicts=[],
            confidence=confidence,
            is_demo_mode=is_demo,
        )
