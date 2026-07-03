"""Unit tests for CLI commands using typer.testing.CliRunner.

Tests cover:
- Each command with valid paths (using the codesense/ project directory)
- Invalid path error handling for path-requiring commands
- --mock flag activation (output contains "[DEMO MODE]")
- CODESENSE_MOCK environment variable activation
- diagram --type validation (invalid type → error)
- deps --type validation (invalid type → error)

Requirements referenced: 5.1-5.12, 8.1, 8.3
"""

import os
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from codesense.main import app

runner = CliRunner()

# Use the codesense/ package directory as a valid path for testing
VALID_PATH = "codesense/main.py"
VALID_DIR = "codesense"
INVALID_PATH = "nonexistent/path/that/does/not/exist.py"


# ─── Tests: Commands with valid paths ─────────────────────────────────────────


class TestCommandsValidPaths:
    """Test each command invocation with valid paths using --mock mode."""

    def test_explain_valid_path(self):
        """explain command runs successfully with a valid path and --mock."""
        result = runner.invoke(app, ["explain", VALID_PATH, "--mock"])
        assert result.exit_code == 0

    def test_describe_valid_path(self):
        """describe command runs successfully with a valid path and --mock."""
        result = runner.invoke(
            app, ["describe", VALID_PATH, "--mock"],
            env={"GEMINI_API_KEY": "fake-key-for-test"},
        )
        assert result.exit_code == 0

    def test_tree_valid_path(self):
        """tree command runs successfully with a valid path and --mock."""
        result = runner.invoke(app, ["tree", VALID_DIR, "--mock"])
        assert result.exit_code == 0

    def test_tree_no_path(self):
        """tree command runs successfully with no path argument (defaults to cwd)."""
        result = runner.invoke(app, ["tree", "--mock"])
        assert result.exit_code == 0

    def test_flow_valid_path(self):
        """flow command runs successfully with a valid path and --mock."""
        result = runner.invoke(app, ["flow", VALID_PATH, "--mock"])
        assert result.exit_code == 0

    def test_diagram_valid_path(self):
        """diagram command runs successfully with a valid path and --mock."""
        result = runner.invoke(app, ["diagram", VALID_DIR, "--mock"])
        assert result.exit_code == 0

    def test_trace_valid_path(self):
        """trace command runs successfully with a valid path and --mock."""
        result = runner.invoke(app, ["trace", VALID_PATH, "--mock"])
        assert result.exit_code == 0

    def test_deps_valid_path(self):
        """deps command runs successfully with a valid path and --mock."""
        result = runner.invoke(app, ["deps", VALID_DIR, "--mock"])
        assert result.exit_code == 0

    def test_related_valid_path(self):
        """related command runs successfully with a valid path and --mock."""
        result = runner.invoke(app, ["related", VALID_PATH, "--mock"])
        assert result.exit_code == 0

    def test_risk_valid_path(self):
        """risk command runs successfully with a valid path and --mock."""
        result = runner.invoke(app, ["risk", VALID_PATH, "--mock"])
        assert result.exit_code == 0

    def test_onboard_no_module(self):
        """onboard command runs successfully with no module and --mock."""
        result = runner.invoke(app, ["onboard", "--mock"])
        assert result.exit_code == 0

    def test_onboard_with_module(self):
        """onboard command runs successfully with a valid module path."""
        result = runner.invoke(app, ["onboard", "--module", VALID_DIR, "--mock"])
        assert result.exit_code == 0

    def test_ingest_valid_path(self):
        """ingest command runs successfully with a valid folder and --mock."""
        result = runner.invoke(app, ["ingest", VALID_DIR, "--mock"])
        assert result.exit_code == 0


# ─── Tests: Invalid path error handling ───────────────────────────────────────


class TestInvalidPathErrors:
    """Test that commands requiring paths fail with appropriate error on invalid path."""

    def test_explain_invalid_path(self):
        """explain exits with error for non-existent path."""
        result = runner.invoke(app, ["explain", INVALID_PATH, "--mock"])
        assert result.exit_code == 1
        assert "Path not found" in result.output or "Error" in result.output

    def test_describe_invalid_path(self):
        """describe exits with error for non-existent path."""
        result = runner.invoke(app, ["describe", INVALID_PATH, "--mock"])
        assert result.exit_code == 1
        assert "Path not found" in result.output or "Error" in result.output

    def test_trace_invalid_path(self):
        """trace exits with error for non-existent path."""
        result = runner.invoke(app, ["trace", INVALID_PATH, "--mock"])
        assert result.exit_code == 1
        assert "Path not found" in result.output or "Error" in result.output

    def test_related_invalid_path(self):
        """related exits with error for non-existent path."""
        result = runner.invoke(app, ["related", INVALID_PATH, "--mock"])
        assert result.exit_code == 1
        assert "Path not found" in result.output or "Error" in result.output

    def test_risk_invalid_path(self):
        """risk exits with error for non-existent path."""
        result = runner.invoke(app, ["risk", INVALID_PATH, "--mock"])
        assert result.exit_code == 1
        assert "Path not found" in result.output or "Error" in result.output

    def test_flow_invalid_path(self):
        """flow exits with error for non-existent path."""
        result = runner.invoke(app, ["flow", INVALID_PATH, "--mock"])
        assert result.exit_code == 1
        assert "Path not found" in result.output or "Error" in result.output

    def test_ingest_invalid_path(self):
        """ingest exits with error for non-existent folder."""
        result = runner.invoke(app, ["ingest", INVALID_PATH, "--mock"])
        assert result.exit_code == 1
        assert "Path not found" in result.output or "Error" in result.output


# ─── Tests: --mock flag activation ───────────────────────────────────────────


class TestMockFlagActivation:
    """Test that --mock flag causes output to contain [DEMO MODE]."""

    def test_explain_mock_shows_demo_mode(self):
        """explain --mock output contains [DEMO MODE]."""
        result = runner.invoke(app, ["explain", VALID_PATH, "--mock"])
        assert "[DEMO MODE]" in result.output

    def test_describe_mock_shows_demo_mode(self):
        """describe --mock output contains [DEMO MODE]."""
        result = runner.invoke(
            app, ["describe", VALID_PATH, "--mock"],
            env={"GEMINI_API_KEY": "fake-key-for-test"},
        )
        assert "[DEMO MODE]" in result.output

    def test_tree_mock_shows_demo_mode(self):
        """tree --mock output contains [DEMO MODE]."""
        result = runner.invoke(app, ["tree", VALID_DIR, "--mock"])
        assert "[DEMO MODE]" in result.output

    def test_flow_mock_shows_demo_mode(self):
        """flow --mock output contains [DEMO MODE]."""
        result = runner.invoke(app, ["flow", VALID_PATH, "--mock"])
        assert "[DEMO MODE]" in result.output

    def test_diagram_mock_shows_demo_mode(self):
        """diagram --mock output contains [DEMO MODE]."""
        result = runner.invoke(app, ["diagram", VALID_DIR, "--mock"])
        assert "[DEMO MODE]" in result.output

    def test_trace_mock_shows_demo_mode(self):
        """trace --mock output contains [DEMO MODE]."""
        result = runner.invoke(app, ["trace", VALID_PATH, "--mock"])
        assert "[DEMO MODE]" in result.output

    def test_deps_mock_shows_demo_mode(self):
        """deps --mock output contains [DEMO MODE]."""
        result = runner.invoke(app, ["deps", VALID_DIR, "--mock"])
        assert "[DEMO MODE]" in result.output

    def test_related_mock_shows_demo_mode(self):
        """related --mock output contains [DEMO MODE]."""
        result = runner.invoke(app, ["related", VALID_PATH, "--mock"])
        assert "[DEMO MODE]" in result.output

    def test_risk_mock_shows_demo_mode(self):
        """risk --mock output contains [DEMO MODE]."""
        result = runner.invoke(app, ["risk", VALID_PATH, "--mock"])
        assert "[DEMO MODE]" in result.output

    def test_onboard_mock_shows_demo_mode(self):
        """onboard --mock output contains [DEMO MODE]."""
        result = runner.invoke(app, ["onboard", "--mock"])
        assert "[DEMO MODE]" in result.output

    def test_ingest_mock_shows_demo_mode(self):
        """ingest --mock output contains [DEMO MODE]."""
        result = runner.invoke(app, ["ingest", VALID_DIR, "--mock"])
        assert "[DEMO MODE]" in result.output


# ─── Tests: CODESENSE_MOCK environment variable ──────────────────────────────


class TestCodesenseMockEnvVar:
    """Test that CODESENSE_MOCK=true activates mock/demo mode without --mock flag."""

    def test_env_var_activates_demo_mode(self):
        """CODESENSE_MOCK=true shows [DEMO MODE] without --mock flag."""
        result = runner.invoke(
            app,
            ["tree", VALID_DIR],
            env={"CODESENSE_MOCK": "true"},
        )
        assert result.exit_code == 0
        assert "[DEMO MODE]" in result.output

    def test_env_var_case_insensitive(self):
        """CODESENSE_MOCK=True (mixed case) also activates demo mode."""
        result = runner.invoke(
            app,
            ["tree", VALID_DIR],
            env={"CODESENSE_MOCK": "True"},
        )
        assert result.exit_code == 0
        assert "[DEMO MODE]" in result.output

    def test_env_var_false_does_not_activate(self):
        """CODESENSE_MOCK=false does not show [DEMO MODE]."""
        result = runner.invoke(
            app,
            ["tree", VALID_DIR],
            env={"CODESENSE_MOCK": "false"},
        )
        assert result.exit_code == 0
        assert "[DEMO MODE]" not in result.output


# ─── Tests: diagram --type validation ────────────────────────────────────────


class TestDiagramTypeValidation:
    """Test that diagram command rejects invalid --type values."""

    def test_diagram_invalid_type_error(self):
        """diagram --type with invalid value exits with error."""
        result = runner.invoke(app, ["diagram", VALID_DIR, "--type", "invalid_type", "--mock"])
        assert result.exit_code == 1
        assert "Invalid diagram type" in result.output

    def test_diagram_type_flowchart_valid(self):
        """diagram --type flowchart is accepted."""
        result = runner.invoke(app, ["diagram", VALID_DIR, "--type", "flowchart", "--mock"])
        assert result.exit_code == 0

    def test_diagram_type_sequence_valid(self):
        """diagram --type sequence is accepted."""
        result = runner.invoke(app, ["diagram", VALID_DIR, "--type", "sequence", "--mock"])
        assert result.exit_code == 0

    def test_diagram_type_architecture_valid(self):
        """diagram --type architecture is accepted."""
        result = runner.invoke(app, ["diagram", VALID_DIR, "--type", "architecture", "--mock"])
        assert result.exit_code == 0

    def test_diagram_type_random_string_error(self):
        """diagram --type with an arbitrary string produces error."""
        result = runner.invoke(app, ["diagram", VALID_DIR, "--type", "pie_chart", "--mock"])
        assert result.exit_code == 1
        assert "Invalid diagram type" in result.output
        assert "flowchart" in result.output
        assert "sequence" in result.output
        assert "architecture" in result.output


# ─── Tests: deps --type validation ───────────────────────────────────────────


class TestDepsTypeValidation:
    """Test that deps command rejects invalid --type values."""

    def test_deps_invalid_type_error(self):
        """deps --type with invalid value exits with error."""
        result = runner.invoke(app, ["deps", VALID_DIR, "--type", "invalid_type", "--mock"])
        assert result.exit_code == 1
        assert "Invalid dependency type" in result.output

    def test_deps_type_env_valid(self):
        """deps --type env is accepted."""
        result = runner.invoke(app, ["deps", VALID_DIR, "--type", "env", "--mock"])
        assert result.exit_code == 0

    def test_deps_type_api_valid(self):
        """deps --type api is accepted."""
        result = runner.invoke(app, ["deps", VALID_DIR, "--type", "api", "--mock"])
        assert result.exit_code == 0

    def test_deps_type_packages_valid(self):
        """deps --type packages is accepted."""
        result = runner.invoke(app, ["deps", VALID_DIR, "--type", "packages", "--mock"])
        assert result.exit_code == 0

    def test_deps_type_all_valid(self):
        """deps --type all is accepted."""
        result = runner.invoke(app, ["deps", VALID_DIR, "--type", "all", "--mock"])
        assert result.exit_code == 0

    def test_deps_type_random_string_error(self):
        """deps --type with an arbitrary string produces error."""
        result = runner.invoke(app, ["deps", VALID_DIR, "--type", "runtime", "--mock"])
        assert result.exit_code == 1
        assert "Invalid dependency type" in result.output
        assert "env" in result.output
        assert "api" in result.output
        assert "packages" in result.output
        assert "all" in result.output
