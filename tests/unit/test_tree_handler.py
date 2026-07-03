"""Unit tests for the tree capability handler."""

import os
import tempfile
from pathlib import Path

from codesense.capabilities.tree import TreeHandler
from codesense.models.output import CommandParams, CommandOutput


class TestTreeHandler:
    """Tests for TreeHandler.run()."""

    def test_run_returns_command_output(self, tmp_path: Path) -> None:
        """run() returns a CommandOutput with correct title."""
        handler = TreeHandler()
        params = CommandParams(path=str(tmp_path), mock=False)
        result = handler.run(params)

        assert isinstance(result, CommandOutput)
        assert result.title == "🌳 Project Structure"

    def test_run_defaults_to_current_dir(self) -> None:
        """run() uses current directory when path is None."""
        handler = TreeHandler()
        params = CommandParams(path=None, mock=False)
        result = handler.run(params)

        assert isinstance(result, CommandOutput)
        assert result.title == "🌳 Project Structure"
        assert len(result.content) > 0

    def test_run_respects_depth(self, tmp_path: Path) -> None:
        """run() limits tree depth when limit is set."""
        # Create nested structure: alpha/beta/gamma
        nested = tmp_path / "alpha" / "beta" / "gamma"
        nested.mkdir(parents=True)
        (nested / "deep_file.txt").write_text("content")
        (tmp_path / "alpha" / "top.txt").write_text("top")

        handler = TreeHandler()
        params = CommandParams(path=str(tmp_path), mock=False, limit=1)
        result = handler.run(params)

        # With depth=1, we should see "alpha" but not "beta" or "gamma"
        assert "alpha" in result.content
        # "gamma" is at depth 3, should not appear
        assert "gamma" not in result.content
        assert "deep_file.txt" not in result.content

    def test_run_shows_files_and_dirs(self, tmp_path: Path) -> None:
        """run() includes both files and directories in tree output."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("# main")
        (tmp_path / "README.md").write_text("readme")

        handler = TreeHandler()
        params = CommandParams(path=str(tmp_path), mock=False)
        result = handler.run(params)

        assert "src" in result.content
        assert "main.py" in result.content
        assert "README.md" in result.content

    def test_run_respects_gitignore(self, tmp_path: Path) -> None:
        """run() excludes entries matched by .gitignore."""
        (tmp_path / ".gitignore").write_text("ignored_dir/\n*.log\n")
        (tmp_path / "ignored_dir").mkdir()
        (tmp_path / "ignored_dir" / "secret.txt").write_text("hidden")
        (tmp_path / "app.log").write_text("log data")
        (tmp_path / "visible.py").write_text("# visible")

        handler = TreeHandler()
        params = CommandParams(path=str(tmp_path), mock=False)
        result = handler.run(params)

        assert "visible.py" in result.content
        assert "ignored_dir" not in result.content
        assert "app.log" not in result.content

    def test_run_sets_demo_mode_flag(self, tmp_path: Path) -> None:
        """run() sets is_demo_mode on output when mock=True in params."""
        handler = TreeHandler()
        params = CommandParams(path=str(tmp_path), mock=True)
        result = handler.run(params)

        assert result.is_demo_mode is True

    def test_run_no_demo_mode_by_default(self, tmp_path: Path) -> None:
        """run() does not set is_demo_mode when mock=False."""
        handler = TreeHandler()
        params = CommandParams(path=str(tmp_path), mock=False)
        result = handler.run(params)

        assert result.is_demo_mode is False

    def test_run_with_nonexistent_path(self) -> None:
        """run() handles non-existent path gracefully (empty tree)."""
        handler = TreeHandler()
        params = CommandParams(path="/tmp/nonexistent_xyz_12345", mock=False)
        result = handler.run(params)

        # TreeFormatter returns a node with empty children for non-dir paths
        assert isinstance(result, CommandOutput)
        assert result.title == "🌳 Project Structure"

    def test_annotations_empty_without_service(self, tmp_path: Path) -> None:
        """Without GeminiService, annotations are empty."""
        (tmp_path / "file.txt").write_text("hello")

        handler = TreeHandler(gemini_service=None)
        params = CommandParams(path=str(tmp_path), mock=False)
        result = handler.run(params)

        # No annotation markers in output
        assert "←" not in result.content
