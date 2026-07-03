"""Unit tests for the describe capability handler."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from codesense.capabilities.describe import DescribeCapabilityHandler


@pytest.fixture
def mock_gemini_service():
    """Create a mocked GeminiService."""
    service = MagicMock()
    service.generate.return_value = "This function adds two numbers and returns the sum."
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


class TestDescribeCapabilityHandler:
    """Tests for DescribeCapabilityHandler."""

    def test_describe_full_file(self, mock_gemini_service, sample_python_file):
        """Test describing a complete file."""
        handler = DescribeCapabilityHandler(mock_gemini_service)
        result = handler.run(str(sample_python_file))

        assert result.title == "📝 Code Description"
        assert result.content == "This function adds two numbers and returns the sum."
        assert len(result.code_snippets) == 1
        assert result.code_snippets[0].language == "python"
        assert "def add" in result.code_snippets[0].code
        assert result.is_demo_mode is False

    def test_describe_with_mock_flag(self, mock_gemini_service, sample_python_file):
        """Test that is_demo_mode is set when mock=True."""
        handler = DescribeCapabilityHandler(mock_gemini_service)
        result = handler.run(str(sample_python_file), mock=True)

        assert result.is_demo_mode is True

    def test_describe_specific_function(self, mock_gemini_service, sample_python_file):
        """Test describing a specific function."""
        handler = DescribeCapabilityHandler(mock_gemini_service)
        result = handler.run(str(sample_python_file), function="add")

        assert result.title == "📝 Code Description"
        assert len(result.code_snippets) == 1
        assert "def add" in result.code_snippets[0].code
        assert "def multiply" not in result.code_snippets[0].code

    def test_describe_async_function(self, mock_gemini_service, sample_python_file):
        """Test describing an async function."""
        handler = DescribeCapabilityHandler(mock_gemini_service)
        result = handler.run(str(sample_python_file), function="fetch_data")

        assert len(result.code_snippets) == 1
        assert "async def fetch_data" in result.code_snippets[0].code

    def test_describe_line_range(self, mock_gemini_service, sample_python_file):
        """Test describing a specific line range."""
        handler = DescribeCapabilityHandler(mock_gemini_service)
        # Line 7 is 'def add(a: int, b: int) -> int:' in the sample file
        result = handler.run(str(sample_python_file), lines="7-10")

        assert len(result.code_snippets) == 1
        assert "def add" in result.code_snippets[0].code

    def test_describe_nonexistent_file(self, mock_gemini_service, tmp_path):
        """Test error handling for a file that can't be read."""
        handler = DescribeCapabilityHandler(mock_gemini_service)
        fake_path = str(tmp_path / "nonexistent.py")
        result = handler.run(fake_path)

        assert result.title == "📝 Code Description"
        assert "Error: Unable to read file" in result.content
        assert len(result.code_snippets) == 0

    def test_describe_nonexistent_function(self, mock_gemini_service, sample_python_file):
        """Test error when a function is not found."""
        handler = DescribeCapabilityHandler(mock_gemini_service)
        result = handler.run(str(sample_python_file), function="nonexistent_func")

        assert "Error: Could not find function 'nonexistent_func'" in result.content

    def test_describe_invalid_line_range(self, mock_gemini_service, sample_python_file):
        """Test error for invalid line range."""
        handler = DescribeCapabilityHandler(mock_gemini_service)
        result = handler.run(str(sample_python_file), lines="999-1000")

        assert "Error: Could not find" in result.content

    def test_describe_llm_error(self, sample_python_file):
        """Test graceful handling of LLM service failure."""
        service = MagicMock()
        service.generate.side_effect = RuntimeError("All API keys rate-limited")

        handler = DescribeCapabilityHandler(service)
        result = handler.run(str(sample_python_file))

        assert "Error generating description" in result.content
        assert "LLM service was unavailable" in result.content
        # Even on LLM failure, we still get the code snippet
        assert len(result.code_snippets) == 1

    def test_describe_language_detection(self, mock_gemini_service, tmp_path):
        """Test language detection from file extension."""
        js_file = tmp_path / "app.js"
        js_file.write_text("function hello() { return 'hi'; }", encoding="utf-8")

        handler = DescribeCapabilityHandler(mock_gemini_service)
        result = handler.run(str(js_file))

        assert result.code_snippets[0].language == "javascript"

    def test_describe_unknown_extension(self, mock_gemini_service, tmp_path):
        """Test fallback to 'text' for unknown extensions."""
        weird_file = tmp_path / "data.xyz"
        weird_file.write_text("some content", encoding="utf-8")

        handler = DescribeCapabilityHandler(mock_gemini_service)
        result = handler.run(str(weird_file))

        assert result.code_snippets[0].language == "text"

    def test_describe_prompt_content(self, mock_gemini_service, sample_python_file):
        """Test that the LLM prompt instructs 'describe what, not why'."""
        handler = DescribeCapabilityHandler(mock_gemini_service)
        handler.run(str(sample_python_file))

        call_args = mock_gemini_service.generate.call_args
        prompt = call_args[0][0]
        assert "what this code does" in prompt.lower()
        assert "not why it exists" in prompt.lower()

    def test_describe_snippet_label_with_function(self, mock_gemini_service, sample_python_file):
        """Test code snippet label includes function name."""
        handler = DescribeCapabilityHandler(mock_gemini_service)
        result = handler.run(str(sample_python_file), function="add")

        assert "::add" in result.code_snippets[0].label

    def test_describe_snippet_label_with_lines(self, mock_gemini_service, sample_python_file):
        """Test code snippet label includes line range."""
        handler = DescribeCapabilityHandler(mock_gemini_service)
        result = handler.run(str(sample_python_file), lines="1-5")

        assert "(lines 1-5)" in result.code_snippets[0].label

    def test_describe_single_line(self, mock_gemini_service, sample_python_file):
        """Test describing a single line."""
        handler = DescribeCapabilityHandler(mock_gemini_service)
        # Line 7 is 'def add(a: int, b: int) -> int:'
        result = handler.run(str(sample_python_file), lines="7")

        assert len(result.code_snippets) == 1
        assert "def add" in result.code_snippets[0].code
