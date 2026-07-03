"""Unit tests for CallGraphBuilder."""

import os
import tempfile
from pathlib import Path

import pytest

from codesense.analysis.call_graph import CallGraphBuilder


@pytest.fixture
def project_dir():
    """Create a temporary project directory with sample Python files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple module structure
        pkg_dir = Path(tmpdir) / "pkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")

        # Main entry file with function calls
        (pkg_dir / "main.py").write_text(
            '''"""Main module."""
from pkg.helpers import helper_func
from pkg.utils import utility

def run():
    result = helper_func()
    utility()
    return result

def standalone():
    return 42
'''
        )

        # Helpers module
        (pkg_dir / "helpers.py").write_text(
            '''"""Helper functions."""
from pkg.utils import utility

def helper_func():
    utility()
    return "helped"

def unused_func():
    return "unused"
'''
        )

        # Utils module
        (pkg_dir / "utils.py").write_text(
            '''"""Utility functions."""

def utility():
    return "util"

def deep_func():
    return "deep"
'''
        )

        yield tmpdir


@pytest.fixture
def circular_project_dir():
    """Create a project with circular call dependencies."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_dir = Path(tmpdir) / "pkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")

        # Module A calls B, B calls A
        (pkg_dir / "a.py").write_text(
            '''"""Module A."""
from pkg.b import func_b

def func_a():
    func_b()
    return "a"
'''
        )

        (pkg_dir / "b.py").write_text(
            '''"""Module B."""
from pkg.a import func_a

def func_b():
    func_a()
    return "b"
'''
        )

        yield tmpdir


@pytest.fixture
def deep_project_dir():
    """Create a project with a deep call chain exceeding max_depth."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_dir = Path(tmpdir) / "pkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")

        # Create a chain of files: level0 -> level1 -> ... -> level12
        for i in range(13):
            if i < 12:
                content = f'''"""Level {i}."""
from pkg.level{i + 1} import func_{i + 1}

def func_{i}():
    func_{i + 1}()
    return {i}
'''
            else:
                content = f'''"""Level {i} (leaf)."""

def func_{i}():
    return {i}
'''
            (pkg_dir / f"level{i}.py").write_text(content)

        yield tmpdir


class TestCallGraphBuilder:
    """Tests for CallGraphBuilder."""

    def test_build_simple_entry_point(self, project_dir):
        """Test building a call graph from a simple file."""
        builder = CallGraphBuilder(project_root=project_dir)
        graph = builder.build("pkg/main.py::run")

        assert graph.root == "pkg/main.py::run"
        assert graph.max_depth_reached is False
        assert graph.depth >= 0

        # The run function calls helper_func and utility
        caller_names = [caller for caller, _ in graph.edges]
        assert any("run" in c for c in caller_names)

    def test_build_traces_imports(self, project_dir):
        """Test that the builder follows import chains."""
        builder = CallGraphBuilder(project_root=project_dir)
        graph = builder.build("pkg/main.py::run")

        # Should have edges from run to helper_func and utility
        edge_callees = [callee for _, callee in graph.edges]
        assert any("helper_func" in c for c in edge_callees)
        assert any("utility" in c for c in edge_callees)

    def test_build_whole_file(self, project_dir):
        """Test building from a whole file (no function specified)."""
        builder = CallGraphBuilder(project_root=project_dir)
        graph = builder.build("pkg/main.py")

        assert graph.root == "pkg/main.py"
        # Should trace from all top-level functions (run and standalone)
        assert len(graph.edges) >= 2  # run calls helper_func and utility

    def test_circular_dependency_handling(self, circular_project_dir):
        """Test that circular call chains don't cause infinite loops."""
        builder = CallGraphBuilder(project_root=circular_project_dir)
        graph = builder.build("pkg/a.py::func_a")

        assert graph.root == "pkg/a.py::func_a"
        # Should not hang — circular deps are detected and stopped
        # There should be edges but no infinite traversal
        assert len(graph.edges) >= 1

    def test_max_depth_enforcement(self, deep_project_dir):
        """Test that max_depth limit is enforced."""
        builder = CallGraphBuilder(project_root=deep_project_dir)
        graph = builder.build("pkg/level0.py::func_0", max_depth=5)

        assert graph.root == "pkg/level0.py::func_0"
        assert graph.max_depth_reached is True
        assert graph.depth <= 5

    def test_max_depth_not_reached_for_shallow_graph(self, project_dir):
        """Test that max_depth_reached is False for graphs within the limit."""
        builder = CallGraphBuilder(project_root=project_dir)
        graph = builder.build("pkg/utils.py::utility", max_depth=10)

        assert graph.max_depth_reached is False
        # utility() doesn't call anything else
        assert graph.depth == 0

    def test_missing_file_returns_empty_graph(self, project_dir):
        """Test that a missing entry point file returns an empty graph."""
        builder = CallGraphBuilder(project_root=project_dir)
        graph = builder.build("nonexistent/file.py::func")

        assert graph.root == "nonexistent/file.py::func"
        assert graph.edges == []
        assert graph.max_depth_reached is False
        assert graph.depth == 0

    def test_missing_function_returns_empty_graph(self, project_dir):
        """Test that a non-existent function returns an empty graph."""
        builder = CallGraphBuilder(project_root=project_dir)
        graph = builder.build("pkg/main.py::nonexistent_func")

        assert graph.root == "pkg/main.py::nonexistent_func"
        assert graph.edges == []

    def test_custom_ast_walker(self, project_dir):
        """Test that a custom ASTWalker can be provided."""
        from codesense.analysis.ast_walker import ASTWalker

        custom_walker = ASTWalker()
        builder = CallGraphBuilder(
            project_root=project_dir, ast_walker=custom_walker
        )
        graph = builder.build("pkg/main.py::run")

        assert graph.root == "pkg/main.py::run"
        assert len(graph.edges) >= 1

    def test_depth_tracks_actual_depth(self, deep_project_dir):
        """Test that depth field reports actual depth reached."""
        builder = CallGraphBuilder(project_root=deep_project_dir)
        # With max_depth=10, the chain goes 13 levels but should stop at 10
        graph = builder.build("pkg/level0.py::func_0", max_depth=10)

        assert graph.depth <= 10
        assert graph.max_depth_reached is True

    def test_default_max_depth_is_10(self, deep_project_dir):
        """Test that the default max_depth parameter is 10."""
        builder = CallGraphBuilder(project_root=deep_project_dir)
        graph = builder.build("pkg/level0.py::func_0")

        # Default max_depth=10, chain is 13, so should be truncated
        assert graph.max_depth_reached is True
        assert graph.depth <= 10

    def test_edges_are_tuples_of_strings(self, project_dir):
        """Test that edges are (caller, callee) string tuples."""
        builder = CallGraphBuilder(project_root=project_dir)
        graph = builder.build("pkg/main.py::run")

        for edge in graph.edges:
            assert isinstance(edge, tuple)
            assert len(edge) == 2
            assert isinstance(edge[0], str)
            assert isinstance(edge[1], str)
