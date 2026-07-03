"""Unit tests for the flow capability handler."""

import tempfile
from pathlib import Path

from codesense.capabilities.flow import FlowHandler
from codesense.models.analysis import CallGraph
from codesense.models.output import CommandParams


class TestBuildNumberedFlow:
    """Tests for FlowHandler._build_numbered_flow helper."""

    def setup_method(self):
        self.handler = FlowHandler(project_root=".")

    def test_empty_call_graph(self):
        """Empty call graph produces informative message."""
        cg = CallGraph(root="main.py", edges=[], max_depth_reached=False, depth=0)
        result = self.handler._build_numbered_flow(cg)
        assert "No execution flow detected" in result

    def test_single_edge(self):
        """Single edge produces one numbered step."""
        cg = CallGraph(
            root="main.py",
            edges=[("main.py::main", "main.py::helper")],
            max_depth_reached=False,
            depth=1,
        )
        result = self.handler._build_numbered_flow(cg)
        assert "Entry point: main.py" in result
        assert "1. main.py::main → main.py::helper" in result

    def test_multiple_edges(self):
        """Multiple edges produce sequential numbered steps."""
        cg = CallGraph(
            root="app.py",
            edges=[
                ("app.py::start", "app.py::init"),
                ("app.py::init", "db.py::connect"),
                ("db.py::connect", "db.py::ping"),
            ],
            max_depth_reached=False,
            depth=3,
        )
        result = self.handler._build_numbered_flow(cg)
        assert "1. app.py::start → app.py::init" in result
        assert "2. app.py::init → db.py::connect" in result
        assert "3. db.py::connect → db.py::ping" in result

    def test_truncation_indicator(self):
        """Truncation warning when max_depth_reached is True."""
        cg = CallGraph(
            root="deep.py",
            edges=[("deep.py::a", "deep.py::b")],
            max_depth_reached=True,
            depth=10,
        )
        result = self.handler._build_numbered_flow(cg)
        assert "truncated" in result.lower()
        assert "10" in result

    def test_no_truncation_indicator_when_not_reached(self):
        """No truncation warning when max_depth_reached is False."""
        cg = CallGraph(
            root="shallow.py",
            edges=[("shallow.py::a", "shallow.py::b")],
            max_depth_reached=False,
            depth=1,
        )
        result = self.handler._build_numbered_flow(cg)
        assert "truncated" not in result.lower()


class TestFlowHandler:
    """Tests for FlowHandler.run() with actual file analysis."""

    def test_run_with_simple_file(self, tmp_path):
        """Flow analysis on a simple Python file produces valid output."""
        source = tmp_path / "example.py"
        source.write_text(
            "def helper():\n"
            "    pass\n"
            "\n"
            "def main():\n"
            "    helper()\n"
        )

        handler = FlowHandler(project_root=str(tmp_path))
        params = CommandParams(path=str(source), mock=False)
        result = handler.run(params)

        assert result.title == "Execution Flow"
        assert result.content  # Non-empty content
        assert len(result.code_snippets) == 1
        assert result.code_snippets[0].language == "mermaid"
        assert "sequenceDiagram" in result.code_snippets[0].code

    def test_run_with_nonexistent_file(self, tmp_path):
        """Flow analysis on nonexistent file produces empty flow gracefully."""
        handler = FlowHandler(project_root=str(tmp_path))
        params = CommandParams(path="nonexistent.py", mock=False)
        result = handler.run(params)

        assert result.title == "Execution Flow"
        assert "No execution flow detected" in result.content
        assert len(result.code_snippets) == 1
        assert result.code_snippets[0].language == "mermaid"

    def test_run_with_demo_mode(self, tmp_path):
        """Demo mode flag is passed through to output."""
        source = tmp_path / "app.py"
        source.write_text("def run():\n    pass\n")

        handler = FlowHandler(project_root=str(tmp_path))
        params = CommandParams(path=str(source), mock=True)
        result = handler.run(params)

        assert result.is_demo_mode is True

    def test_run_without_demo_mode(self, tmp_path):
        """Non-demo mode flag is passed through to output."""
        source = tmp_path / "app.py"
        source.write_text("def run():\n    pass\n")

        handler = FlowHandler(project_root=str(tmp_path))
        params = CommandParams(path=str(source), mock=False)
        result = handler.run(params)

        assert result.is_demo_mode is False

    def test_run_with_specific_function(self, tmp_path):
        """Flow analysis targeting a specific function via :: notation."""
        source = tmp_path / "module.py"
        source.write_text(
            "def unrelated():\n"
            "    pass\n"
            "\n"
            "def target():\n"
            "    unrelated()\n"
        )

        handler = FlowHandler(project_root=str(tmp_path))
        params = CommandParams(path=f"{source}::target", mock=False)
        result = handler.run(params)

        assert result.title == "Execution Flow"
        assert "target" in result.content

    def test_mermaid_diagram_in_code_snippets(self, tmp_path):
        """Mermaid diagram is returned as a code snippet with correct metadata."""
        source = tmp_path / "flow.py"
        source.write_text(
            "def step1():\n"
            "    step2()\n"
            "\n"
            "def step2():\n"
            "    pass\n"
        )

        handler = FlowHandler(project_root=str(tmp_path))
        params = CommandParams(path=str(source), mock=False)
        result = handler.run(params)

        assert len(result.code_snippets) == 1
        snippet = result.code_snippets[0]
        assert snippet.language == "mermaid"
        assert snippet.label == "Execution Flow Sequence Diagram"
        assert "sequenceDiagram" in snippet.code

    def test_truncation_in_output(self, tmp_path):
        """When max depth is reached, output indicates truncation."""
        # Create a deeply nested call chain (11 levels deep to exceed max_depth=10)
        lines = []
        for i in range(12):
            if i < 11:
                lines.append(f"def func_{i}():\n    func_{i+1}()\n\n")
            else:
                lines.append(f"def func_{i}():\n    pass\n")

        source = tmp_path / "deep.py"
        source.write_text("".join(lines))

        handler = FlowHandler(project_root=str(tmp_path))
        params = CommandParams(path=f"{source}::func_0", mock=False)
        result = handler.run(params)

        # Should indicate truncation in content
        assert "truncated" in result.content.lower() or "Max depth" in result.code_snippets[0].code

    def test_constructor_defaults_to_cwd(self):
        """FlowHandler constructor defaults project_root to cwd."""
        handler = FlowHandler()
        assert handler._project_root == str(Path.cwd())

    def test_parse_failure_error_message(self, tmp_path):
        """Files with syntax errors produce informative error messages."""
        source = tmp_path / "broken.py"
        source.write_text("def broken(:\n    pass\n")  # Syntax error

        handler = FlowHandler(project_root=str(tmp_path))
        params = CommandParams(path=str(source), mock=False)
        result = handler.run(params)

        # Should handle gracefully with error info
        assert result.title == "Execution Flow"
        assert "could not be analyzed" in result.content or "No execution flow" in result.content
