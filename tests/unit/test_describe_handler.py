"""Unit tests for the DescribeHandler (CapabilityHandler protocol-conformant)."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from codesense.capabilities.describe import DescribeHandler
from codesense.models.output import CommandOutput, CommandParams


@pytest.fixture
def mock_gemini_service():
    """Create a mocked GeminiService."""
    service = MagicMock()
    service.generate.return_value = "This code defines a utility function for adding numbers."
    return service


@pytest.fixture
def sample_python_file(tmp_path):
    """Create a sample Python file for testing."""
    content = '''"""Sample module for testing."""

import os
from pathlib import Path


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y


async def fetch_data(url: str) -> str:
    """Fetch data from a URL."""
    return f"data from {url}"


class Calculator:
    """A simple calculator."""

    def __init__(self):
        self.history = []

    def compute(self, a, b, op):
        result = op(a, b)
        self.history.append(result)
        return result
'''
    file_path = tmp_path / "sample.py"
    file_path.write_text(content, encoding="utf-8")
    return file_path


class TestDescribeHandler:
    """Tests for DescribeHandler following the CapabilityHandler protocol."""

    def test_run_with_command_params(self, mock_gemini_service, sample_python_file):
        """Test that run() accepts CommandParams and returns CommandOutput."""
        handler = DescribeHandler(gemini_service=mock_gemini_service)
        params = CommandParams(path=str(sample_python_file))
        result = handler.run(params)

        assert isinstance(result, CommandOutput)
        assert result.title == "📝 Code Description"
        assert result.content == "This code defines a utility function for adding numbers."
        assert len(result.code_snippets) == 1
        assert result.code_snippets[0].language == "python"

    def test_run_with_function_name(self, mock_gemini_service, sample_python_file):
        """Test extracting a specific function using AST (Python file)."""
        handler = DescribeHandler(gemini_service=mock_gemini_service)
        params = CommandParams(path=str(sample_python_file), function_name="add")
        result = handler.run(params)

        assert len(result.code_snippets) == 1
        assert "def add" in result.code_snippets[0].code
        assert "def multiply" not in result.code_snippets[0].code

    def test_run_with_line_range(self, mock_gemini_service, sample_python_file):
        """Test extracting a specific line range."""
        handler = DescribeHandler(gemini_service=mock_gemini_service)
        params = CommandParams(path=str(sample_python_file), line_range="7-10")
        result = handler.run(params)

        assert len(result.code_snippets) == 1
        assert "def add" in result.code_snippets[0].code

    def test_run_no_path(self, mock_gemini_service):
        """Test error when no path is provided."""
        handler = DescribeHandler(gemini_service=mock_gemini_service)
        params = CommandParams()
        result = handler.run(params)

        assert "Error: No file path provided" in result.content

    def test_run_nonexistent_file(self, mock_gemini_service, tmp_path):
        """Test error for a file that doesn't exist."""
        handler = DescribeHandler(gemini_service=mock_gemini_service)
        params = CommandParams(path=str(tmp_path / "nonexistent.py"))
        result = handler.run(params)

        assert "Error: Unable to read file" in result.content

    def test_run_nonexistent_function(self, mock_gemini_service, sample_python_file):
        """Test error when function is not found."""
        handler = DescribeHandler(gemini_service=mock_gemini_service)
        params = CommandParams(path=str(sample_python_file), function_name="no_such_func")
        result = handler.run(params)

        assert "Error: Could not find function 'no_such_func'" in result.content

    def test_run_demo_mode(self, mock_gemini_service, sample_python_file):
        """Test that mock flag sets is_demo_mode."""
        handler = DescribeHandler(gemini_service=mock_gemini_service)
        params = CommandParams(path=str(sample_python_file), mock=True)
        result = handler.run(params)

        assert result.is_demo_mode is True

    def test_prompt_content(self, mock_gemini_service, sample_python_file):
        """Test the LLM prompt includes required instructions."""
        handler = DescribeHandler(gemini_service=mock_gemini_service)
        params = CommandParams(path=str(sample_python_file))
        handler.run(params)

        call_args = mock_gemini_service.generate.call_args
        prompt = call_args[0][0]
        assert "Describe what this code does in plain English" in prompt
        assert "Do not explain history" in prompt

    def test_ast_extraction_for_async_function(self, mock_gemini_service, sample_python_file):
        """Test AST-based extraction works for async functions."""
        handler = DescribeHandler(gemini_service=mock_gemini_service)
        params = CommandParams(path=str(sample_python_file), function_name="fetch_data")
        result = handler.run(params)

        assert len(result.code_snippets) == 1
        assert "async def fetch_data" in result.code_snippets[0].code

    def test_ast_extraction_for_method(self, mock_gemini_service, sample_python_file):
        """Test AST-based extraction works for class methods."""
        handler = DescribeHandler(gemini_service=mock_gemini_service)
        params = CommandParams(path=str(sample_python_file), function_name="compute")
        result = handler.run(params)

        assert len(result.code_snippets) == 1
        assert "def compute" in result.code_snippets[0].code

    def test_llm_error_handling(self, sample_python_file):
        """Test graceful handling when LLM service fails."""
        service = MagicMock()
        service.generate.side_effect = RuntimeError("API unavailable")

        handler = DescribeHandler(gemini_service=service)
        params = CommandParams(path=str(sample_python_file))
        result = handler.run(params)

        assert "Error generating description" in result.content
        assert "LLM service was unavailable" in result.content
        # Code snippet is still present even on LLM failure
        assert len(result.code_snippets) == 1

    def test_no_git_history_needed(self, mock_gemini_service, sample_python_file):
        """Test that describe works without any git or credentials setup."""
        # This test verifies the fast path — no MCP tools, no RAG
        handler = DescribeHandler(gemini_service=mock_gemini_service)
        params = CommandParams(path=str(sample_python_file))
        result = handler.run(params)

        # Should succeed purely on file read + LLM
        assert result.content == "This code defines a utility function for adding numbers."
        assert result.confidence is None  # No confidence score for describe
        assert result.conflicts == []  # No conflicts for describe

    def test_constructor_accepts_optional_service(self):
        """Test that constructor works with explicitly provided service."""
        service = MagicMock()
        handler = DescribeHandler(gemini_service=service)
        assert handler._gemini_service is service
