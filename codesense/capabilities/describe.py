"""Describe capability handler — the FAST PATH for code description.

Reads source code, optionally filters to a specific function (via ASTWalker)
or line range, then sends to GeminiService for a plain-English description.

This is the fast path: no MCP tools, no RAG, no reasoning loop.
Just: read code → LLM → description.

Works even in offline mode (only needs the LLM).
Does not require git history or credentials.

Requirements: 5.2
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Optional

from codesense.models.output import CodeSnippet, CommandOutput, CommandParams


class DescribeHandler:
    """Capability handler for the 'describe' command.

    Reads source code from a file path, optionally filtering to a specific
    function (using ASTWalker for Python files) or line range, then uses the
    Gemini LLM to generate a plain-English description of what the code does.

    This is the FAST PATH — no MCP tools, no RAG, no reasoning loop.
    Just: read code → LLM → description.

    Args:
        gemini_service: Optional GeminiService instance for LLM-based description.
            If None, creates a default GeminiService using environment config.
    """

    def __init__(self, gemini_service: Optional[object] = None) -> None:
        if gemini_service is None:
            gemini_service = self._create_default_service()
        self._gemini_service = gemini_service

    def run(self, params: CommandParams) -> CommandOutput:
        """Generate a high-level code description for the given path.

        Args:
            params: Parsed CLI arguments. Uses:
                - params.path: File path to describe (required).
                - params.function_name: Optional function name to extract.
                - params.line_range: Optional line range (e.g. "10-20").
                - params.mock: Whether demo mode is active.

        Returns:
            CommandOutput with the LLM-generated description.
        """
        if not params.path:
            return CommandOutput(
                title="📝 Code Description",
                content="Error: No file path provided.",
                is_demo_mode=params.mock,
            )

        file_path = Path(params.path)

        # Read the file content
        source_code = self._read_file(file_path)
        if source_code is None:
            return CommandOutput(
                title="📝 Code Description",
                content=(
                    f"Error: Unable to read file '{params.path}'. "
                    "The file may not exist, may be a binary file, or may not be accessible."
                ),
                is_demo_mode=params.mock,
            )

        # Detect language from file extension
        language = self._detect_language(file_path)

        # Filter to specific function or line range
        code_to_describe = self._extract_code(
            source_code, file_path, params.function_name, params.line_range
        )
        if code_to_describe is None:
            filter_desc = (
                f"function or class '{params.function_name}'"
                if params.function_name
                else f"lines {params.line_range}"
            )
            return CommandOutput(
                title="📝 Code Description",
                content=f"Error: Could not find {filter_desc} in '{params.path}'.",
                code_snippets=[
                    CodeSnippet(code=source_code, language=language, label=str(file_path))
                ],
                is_demo_mode=params.mock,
            )

        # Build prompt and call LLM
        prompt = self._build_prompt(code_to_describe)
        description = self._call_llm(prompt)

        # Build snippet label
        label = self._build_snippet_label(file_path, params.function_name, params.line_range)

        return CommandOutput(
            title="📝 Code Description",
            content=description,
            code_snippets=[
                CodeSnippet(code=code_to_describe, language=language, label=label)
            ],
            is_demo_mode=params.mock,
        )

    def _extract_code(
        self,
        source_code: str,
        file_path: Path,
        function_name: Optional[str],
        line_range: Optional[str],
    ) -> Optional[str]:
        """Extract code based on function name or line range.

        Uses ASTWalker for Python files when extracting by function name.
        Falls back to regex-based extraction for non-Python files.

        Args:
            source_code: Full file content.
            file_path: Path to the source file.
            function_name: Optional function name to extract.
            line_range: Optional line range string (e.g. "10-20").

        Returns:
            Extracted code string, or None if target not found.
            Returns full source if neither function nor lines specified.
        """
        if function_name:
            return self._extract_function(source_code, file_path, function_name)
        if line_range:
            return self._extract_line_range(source_code, line_range)
        return source_code

    def _extract_function(
        self, source_code: str, file_path: Path, function_name: str
    ) -> Optional[str]:
        """Extract a function's source code.

        For Python files, uses AST-based extraction to precisely locate the
        function boundaries. For non-Python files, falls back to regex/indentation.

        Args:
            source_code: Full file content.
            file_path: Path to determine file type.
            function_name: Name of the function to extract.

        Returns:
            Function source code, or None if not found.
        """
        if file_path.suffix == ".py":
            return self._extract_function_ast(source_code, function_name)
        return self._extract_function_regex(source_code, function_name)

    def _extract_function_ast(
        self, source_code: str, function_name: str
    ) -> Optional[str]:
        """Extract a Python function using AST parsing.

        Finds the function/method definition in the AST and extracts the
        corresponding source lines including decorators.

        Args:
            source_code: Full Python source code.
            function_name: Name of the function to find.

        Returns:
            Function source code, or None if not found or parse fails.
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            # Fall back to regex if AST parsing fails
            return self._extract_function_regex(source_code, function_name)

        source_lines = source_code.split("\n")

        # Walk the AST to find the function OR class (works for nested defs too)
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                if node.name == function_name:
                    # Get start line (accounting for decorators)
                    start_line = node.lineno - 1  # 0-indexed
                    if node.decorator_list:
                        start_line = node.decorator_list[0].lineno - 1
                    end_line = node.end_lineno  # 1-indexed, inclusive

                    if end_line is None:
                        # Fallback for older Python without end_lineno
                        return self._extract_function_regex(source_code, function_name)

                    return "\n".join(source_lines[start_line:end_line])

        return None

    def _extract_function_regex(
        self, source_code: str, function_name: str
    ) -> Optional[str]:
        """Extract a function using regex and indentation-based parsing.

        Fallback for non-Python files or when AST parsing fails.

        Args:
            source_code: Full file content.
            function_name: Name of the function to extract.

        Returns:
            Function source code, or None if not found.
        """
        source_lines = source_code.split("\n")
        pattern = re.compile(
            rf"^(\s*)(?:async\s+)?(?:def|class)\s+{re.escape(function_name)}\b"
        )

        start_idx = None
        base_indent = 0

        for i, line in enumerate(source_lines):
            match = pattern.match(line)
            if match:
                start_idx = i
                base_indent = len(match.group(1))
                break

        if start_idx is None:
            return None

        # Collect lines belonging to this function based on indentation
        end_idx = start_idx + 1
        for i in range(start_idx + 1, len(source_lines)):
            line = source_lines[i]
            if line.strip() == "":
                end_idx = i + 1
                continue
            line_indent = len(line) - len(line.lstrip())
            if line_indent > base_indent:
                end_idx = i + 1
            else:
                break

        return "\n".join(source_lines[start_idx:end_idx])

    def _extract_line_range(self, source_code: str, line_range: str) -> Optional[str]:
        """Extract a specific line range from source code.

        Args:
            source_code: Full file content.
            line_range: Line range string, e.g. "10-20" or "5".

        Returns:
            Extracted lines, or None if the range is invalid.
        """
        source_lines = source_code.split("\n")
        total_lines = len(source_lines)

        try:
            if "-" in line_range:
                parts = line_range.split("-", 1)
                start = int(parts[0]) - 1  # Convert to 0-indexed
                end = int(parts[1])
            else:
                start = int(line_range) - 1
                end = int(line_range)
        except (ValueError, IndexError):
            return None

        if start < 0 or end > total_lines or start >= end:
            return None

        return "\n".join(source_lines[start:end])

    def _build_prompt(self, code: str) -> str:
        """Build the LLM prompt for code description.

        Args:
            code: The source code to describe.

        Returns:
            Prompt string for the LLM.
        """
        return (
            "Describe what this code does in plain English, "
            "not why it exists. Do not explain history.\n\n"
            f"```\n{code}\n```"
        )

    def _call_llm(self, prompt: str) -> str:
        """Call GeminiService to generate the description.

        Args:
            prompt: The prompt to send to the LLM.

        Returns:
            Generated description text, or an error message on failure.
        """
        try:
            return self._gemini_service.generate(prompt)
        except Exception as e:
            return (
                f"Error generating description: {e}\n\n"
                "The code was read successfully but the LLM service was unavailable."
            )

    def _read_file(self, file_path: Path) -> Optional[str]:
        """Read file content, returning None if the file can't be read.

        Args:
            file_path: Path to the file to read.

        Returns:
            File content as string, or None on failure.
        """
        try:
            return file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension.

        Args:
            file_path: Path to determine language for.

        Returns:
            Language identifier string for syntax highlighting.
        """
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".jsx": "javascript",
            ".java": "java",
            ".rb": "ruby",
            ".go": "go",
            ".rs": "rust",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".sh": "bash",
            ".bash": "bash",
            ".zsh": "bash",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".toml": "toml",
            ".md": "markdown",
            ".sql": "sql",
            ".html": "html",
            ".css": "css",
            ".xml": "xml",
        }
        return extension_map.get(file_path.suffix.lower(), "text")

    def _build_snippet_label(
        self,
        file_path: Path,
        function_name: Optional[str],
        line_range: Optional[str],
    ) -> str:
        """Build a human-readable label for the code snippet.

        Args:
            file_path: File path.
            function_name: Function name if filtered.
            line_range: Line range if filtered.

        Returns:
            Label string like "path/to/file.py::my_function".
        """
        label = str(file_path)
        if function_name:
            label = f"{label}::{function_name}"
        elif line_range:
            label = f"{label} (lines {line_range})"
        return label

    @staticmethod
    def _create_default_service() -> object:
        """Create a default GeminiService using environment configuration.

        Returns:
            A configured GeminiService instance.

        Raises:
            RuntimeError: If no API keys are configured.
        """
        from codesense.llm import GeminiService, KeyRotator

        keys_str = os.environ.get("GEMINI_API_KEYS", "")
        if not keys_str:
            # Try single key fallback
            single_key = os.environ.get("GEMINI_API_KEY", "")
            if single_key:
                keys_str = single_key

        if not keys_str:
            raise RuntimeError(
                "No Gemini API keys configured. "
                "Set GEMINI_API_KEYS or GEMINI_API_KEY environment variable."
            )

        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        rotator = KeyRotator(api_keys=keys)
        return GeminiService(key_rotator=rotator)


class DescribeCapabilityHandler:
    """Legacy wrapper providing the old-style API for backward compatibility.

    Delegates to DescribeHandler internally. Provides the run(path, *, function,
    lines, mock) signature that existing code uses.
    """

    def __init__(self, gemini_service: object) -> None:
        self._handler = DescribeHandler(gemini_service=gemini_service)

    def run(
        self,
        path: str,
        *,
        function: Optional[str] = None,
        lines: Optional[str] = None,
        mock: bool = False,
    ) -> CommandOutput:
        """Generate a high-level code description (legacy API).

        Args:
            path: File path to describe.
            function: Optional function name to filter to.
            lines: Optional line range string (e.g. "10-20").
            mock: Whether demo/mock mode is active.

        Returns:
            CommandOutput with the LLM-generated description.
        """
        params = CommandParams(
            path=path,
            function_name=function,
            line_range=lines,
            mock=mock,
        )
        return self._handler.run(params)
