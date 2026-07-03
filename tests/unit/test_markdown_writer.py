"""Unit tests for MarkdownWriter."""

import os
import tempfile

import pytest

from codesense.output.markdown_writer import MarkdownWriter


@pytest.fixture
def writer():
    return MarkdownWriter()


class TestMarkdownWriterRender:
    """Tests for MarkdownWriter.render()."""

    def test_single_section(self, writer):
        sections = [{"heading": "Title", "content": "Some content.", "level": 1}]
        result = writer.render(sections)
        assert "# Title" in result
        assert "Some content." in result

    def test_multiple_sections(self, writer):
        sections = [
            {"heading": "Intro", "content": "Introduction text.", "level": 1},
            {"heading": "Details", "content": "Detail text.", "level": 2},
        ]
        result = writer.render(sections)
        assert "# Intro" in result
        assert "## Details" in result
        assert "Introduction text." in result
        assert "Detail text." in result

    def test_heading_levels(self, writer):
        sections = [
            {"heading": "H1", "content": "", "level": 1},
            {"heading": "H2", "content": "", "level": 2},
            {"heading": "H3", "content": "", "level": 3},
            {"heading": "H4", "content": "", "level": 4},
            {"heading": "H5", "content": "", "level": 5},
            {"heading": "H6", "content": "", "level": 6},
        ]
        result = writer.render(sections)
        assert "# H1" in result
        assert "## H2" in result
        assert "### H3" in result
        assert "#### H4" in result
        assert "##### H5" in result
        assert "###### H6" in result

    def test_level_clamped_below(self, writer):
        sections = [{"heading": "Test", "content": "body", "level": 0}]
        result = writer.render(sections)
        assert "# Test" in result

    def test_level_clamped_above(self, writer):
        sections = [{"heading": "Test", "content": "body", "level": 10}]
        result = writer.render(sections)
        assert "###### Test" in result

    def test_empty_content(self, writer):
        sections = [{"heading": "Empty", "content": "", "level": 2}]
        result = writer.render(sections)
        assert "## Empty" in result
        # No blank line with content follows
        lines = result.strip().split("\n")
        assert lines[0] == "## Empty"

    def test_result_ends_with_newline(self, writer):
        sections = [{"heading": "End", "content": "text", "level": 1}]
        result = writer.render(sections)
        assert result.endswith("\n")

    def test_empty_sections_list(self, writer):
        result = writer.render([])
        assert result == "\n"


class TestMarkdownWriterWriteFile:
    """Tests for MarkdownWriter.write_file()."""

    def test_write_to_file(self, writer):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.md")
            writer.write_file("# Test\n\nContent.\n", output_path)
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert content == "# Test\n\nContent.\n"

    def test_creates_parent_directories(self, writer):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "sub", "dir", "output.md")
            writer.write_file("hello", output_path)
            assert os.path.exists(output_path)
            with open(output_path, "r", encoding="utf-8") as f:
                assert f.read() == "hello"

    def test_overwrites_existing_file(self, writer):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.md")
            writer.write_file("first", output_path)
            writer.write_file("second", output_path)
            with open(output_path, "r", encoding="utf-8") as f:
                assert f.read() == "second"


class TestMarkdownWriterRenderOnboarding:
    """Tests for MarkdownWriter.render_onboarding()."""

    def test_full_project_info(self, writer):
        project_info = {
            "name": "MyProject",
            "purpose": "A tool for testing.",
            "structure": "src/ contains source code.",
            "execution_flow": "main.py → app.py → handlers",
            "design_decisions": [
                "Use async for IO",
                "REST over GraphQL",
            ],
            "setup": "pip install -e .",
            "safe_areas": ["tests/", "docs/"],
        }
        result = writer.render_onboarding(project_info)
        assert "# MyProject — Onboarding Guide" in result
        assert "## Purpose" in result
        assert "A tool for testing." in result
        assert "## Structure" in result
        assert "src/ contains source code." in result
        assert "## Execution Flow" in result
        assert "main.py → app.py → handlers" in result
        assert "## Design Decisions" in result
        assert "- Use async for IO" in result
        assert "- REST over GraphQL" in result
        assert "## Setup" in result
        assert "pip install -e ." in result
        assert "## Safe Areas to Modify" in result
        assert "- tests/" in result
        assert "- docs/" in result

    def test_minimal_project_info(self, writer):
        result = writer.render_onboarding({})
        assert "# Project — Onboarding Guide" in result
        assert "## Purpose" in result
        assert "No purpose description provided." in result
        assert "## Structure" in result
        assert "## Execution Flow" in result
        assert "## Design Decisions" in result
        assert "No design decisions documented." in result
        assert "## Setup" in result
        assert "## Safe Areas to Modify" in result
        assert "No safe-to-change areas identified." in result

    def test_empty_lists_show_fallback_text(self, writer):
        project_info = {
            "name": "EmptyProject",
            "design_decisions": [],
            "safe_areas": [],
        }
        result = writer.render_onboarding(project_info)
        assert "No design decisions documented." in result
        assert "No safe-to-change areas identified." in result

    def test_onboarding_has_all_six_sections(self, writer):
        result = writer.render_onboarding({"name": "Test"})
        assert "## Purpose" in result
        assert "## Structure" in result
        assert "## Execution Flow" in result
        assert "## Design Decisions" in result
        assert "## Setup" in result
        assert "## Safe Areas to Modify" in result
