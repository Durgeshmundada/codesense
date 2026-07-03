"""Unit tests for the onboard capability handler."""

import os
import tempfile
from pathlib import Path

import pytest

from codesense.capabilities.onboard import OnboardHandler, run_onboard
from codesense.models.output import CommandOutput, CommandParams


@pytest.fixture
def sample_project(tmp_path):
    """Create a sample project structure for testing."""
    # Create README
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Sample Project\n\n"
        "A sample project for testing the onboard handler.\n\n"
        "## Features\n\n"
        "- Feature A\n"
        "- Feature B\n",
        encoding="utf-8",
    )

    # Create pyproject.toml
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "sample-project"\nversion = "1.0.0"\n',
        encoding="utf-8",
    )

    # Create requirements.txt
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("requests>=2.28\ntyper>=0.9\n", encoding="utf-8")

    # Create .env.template
    env_template = tmp_path / ".env.template"
    env_template.write_text(
        "# API keys\nAPI_KEY=your-key-here\nDATABASE_URL=sqlite:///db.sqlite\n",
        encoding="utf-8",
    )

    # Create source directory
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    init = src_dir / "__init__.py"
    init.write_text('"""Core source module for sample project."""\n', encoding="utf-8")

    main = src_dir / "main.py"
    main.write_text(
        "def main():\n    print('hello')\n\nif __name__ == '__main__':\n    main()\n",
        encoding="utf-8",
    )

    # Create tests directory
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "__init__.py").write_text("", encoding="utf-8")

    # Create docs directory
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "guide.md").write_text("# Guide\nSome docs.\n", encoding="utf-8")

    return tmp_path


class TestOnboardHandler:
    """Tests for OnboardHandler."""

    def test_run_returns_command_output(self, sample_project):
        """Test that run() returns a CommandOutput instance."""
        handler = OnboardHandler(project_root=str(sample_project))
        params = CommandParams(path=None, output=None, mock=False)
        result = handler.run(params)

        assert isinstance(result, CommandOutput)
        assert "Onboarding Guide" in result.title
        assert result.is_demo_mode is False

    def test_run_demo_mode(self, sample_project):
        """Test that demo mode flag is propagated."""
        handler = OnboardHandler(project_root=str(sample_project))
        params = CommandParams(path=None, output=None, mock=True)
        result = handler.run(params)

        assert result.is_demo_mode is True

    def test_detects_project_name_from_pyproject(self, sample_project):
        """Test that the project name is read from pyproject.toml."""
        handler = OnboardHandler(project_root=str(sample_project))
        params = CommandParams(path=None, output=None, mock=False)
        result = handler.run(params)

        assert "sample-project" in result.content

    def test_includes_purpose_from_readme(self, sample_project):
        """Test that purpose section includes README content."""
        handler = OnboardHandler(project_root=str(sample_project))
        params = CommandParams(path=None, output=None, mock=False)
        result = handler.run(params)

        assert "sample project for testing" in result.content

    def test_includes_structure_section(self, sample_project):
        """Test that the output includes a project structure section."""
        handler = OnboardHandler(project_root=str(sample_project))
        params = CommandParams(path=None, output=None, mock=False)
        result = handler.run(params)

        assert "Structure" in result.content

    def test_includes_setup_instructions(self, sample_project):
        """Test that setup instructions are generated."""
        handler = OnboardHandler(project_root=str(sample_project))
        params = CommandParams(path=None, output=None, mock=False)
        result = handler.run(params)

        assert "pip install" in result.content
        assert "requirements.txt" in result.content

    def test_includes_env_vars_from_template(self, sample_project):
        """Test that environment variables from .env.template are listed."""
        handler = OnboardHandler(project_root=str(sample_project))
        params = CommandParams(path=None, output=None, mock=False)
        result = handler.run(params)

        assert "API_KEY" in result.content
        assert "DATABASE_URL" in result.content

    def test_includes_safe_areas(self, sample_project):
        """Test that safe-to-change areas are identified."""
        handler = OnboardHandler(project_root=str(sample_project))
        params = CommandParams(path=None, output=None, mock=False)
        result = handler.run(params)

        assert "Safe" in result.content
        assert "tests" in result.content.lower()

    def test_scoped_to_module(self, sample_project):
        """Test that onboarding can be scoped to a specific module."""
        handler = OnboardHandler(project_root=str(sample_project))
        src_path = str(sample_project / "src")
        params = CommandParams(path=src_path, output=None, mock=False)
        result = handler.run(params)

        assert isinstance(result, CommandOutput)
        # Module docstring should appear
        assert "Core source module" in result.content

    def test_output_writes_to_file(self, sample_project):
        """Test that --output writes the document to a file."""
        handler = OnboardHandler(project_root=str(sample_project))
        output_file = str(sample_project / "ONBOARDING.md")
        params = CommandParams(path=None, output=output_file, mock=False)
        result = handler.run(params)

        # File should be created
        assert Path(output_file).is_file()
        file_content = Path(output_file).read_text(encoding="utf-8")
        assert "Onboarding Guide" in file_content
        # Result content should confirm the write
        assert "Written to" in result.content

    def test_no_readme_fallback(self, tmp_path):
        """Test graceful handling when no README exists."""
        handler = OnboardHandler(project_root=str(tmp_path))
        params = CommandParams(path=None, output=None, mock=False)
        result = handler.run(params)

        assert isinstance(result, CommandOutput)
        assert "No project description found" in result.content

    def test_design_decisions_from_spec(self, sample_project):
        """Test that design decisions are extracted from .kiro/specs."""
        # Create a .kiro/specs structure
        spec_dir = sample_project / ".kiro" / "specs" / "feature"
        spec_dir.mkdir(parents=True)
        design_file = spec_dir / "design.md"
        design_file.write_text(
            "# Design\n\n"
            "### Design Decisions\n\n"
            "- **Use SQLite** for local storage\n"
            "- **Use REST** over GraphQL\n\n"
            "## Architecture\n\nSome text.\n",
            encoding="utf-8",
        )

        handler = OnboardHandler(project_root=str(sample_project))
        params = CommandParams(path=None, output=None, mock=False)
        result = handler.run(params)

        assert "SQLite" in result.content
        assert "REST" in result.content


class TestRunOnboard:
    """Tests for the convenience run_onboard function."""

    def test_run_onboard_returns_output(self, sample_project):
        """Test that run_onboard produces a CommandOutput."""
        result = run_onboard(project_root=str(sample_project))
        assert isinstance(result, CommandOutput)
        assert "Onboarding Guide" in result.title

    def test_run_onboard_with_module(self, sample_project):
        """Test run_onboard with a specific module path."""
        src_path = str(sample_project / "src")
        result = run_onboard(module_path=src_path, project_root=str(sample_project))
        assert isinstance(result, CommandOutput)

    def test_run_onboard_with_output(self, sample_project):
        """Test run_onboard with an output file path."""
        output_file = str(sample_project / "output.md")
        result = run_onboard(output_path=output_file, project_root=str(sample_project))
        assert Path(output_file).is_file()
