"""Unit tests for MermaidFormatter."""

import re

import pytest

from codesense.models.analysis import CallGraph, ImportGraph
from codesense.output.mermaid_formatter import (
    MermaidFormatter,
    _escape_label,
    _sanitize_id,
)


@pytest.fixture
def formatter():
    return MermaidFormatter()


class TestSanitizeId:
    def test_simple_name(self):
        assert _sanitize_id("MyClass") == "MyClass"

    def test_dots_replaced(self):
        assert _sanitize_id("module.submodule") == "module_submodule"

    def test_starts_with_digit(self):
        result = _sanitize_id("3rdParty")
        assert not result[0].isdigit()
        assert result == "n_3rdParty"

    def test_special_chars(self):
        result = _sanitize_id("my-class<T>")
        assert re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", result)

    def test_empty_string(self):
        assert _sanitize_id("") == "node"


class TestEscapeLabel:
    def test_quotes_escaped(self):
        assert _escape_label('He said "hello"') == "He said 'hello'"

    def test_angle_brackets(self):
        assert _escape_label("List<int>") == "List&lt;int&gt;"


class TestClassDiagramDict:
    """Tests for the dict-based class_diagram API (legacy format)."""

    def test_basic_inheritance(self, formatter):
        relationships = [
            {"from": "Dog", "to": "Animal", "type": "inherits"}
        ]
        result = formatter.class_diagram(relationships)
        assert "classDiagram" in result
        assert "Animal" in result
        assert "Dog" in result
        assert "<|--" in result

    def test_uses_relationship(self, formatter):
        relationships = [
            {"from": "Service", "to": "Repository", "type": "uses"}
        ]
        result = formatter.class_diagram(relationships)
        assert "..>" in result

    def test_calls_relationship(self, formatter):
        relationships = [
            {"from": "Controller", "to": "Service", "type": "calls"}
        ]
        result = formatter.class_diagram(relationships)
        assert "-->" in result

    def test_all_entities_appear(self, formatter):
        relationships = [
            {"from": "A", "to": "B", "type": "inherits"},
            {"from": "C", "to": "D", "type": "uses"},
        ]
        result = formatter.class_diagram(relationships)
        assert "A" in result
        assert "B" in result
        assert "C" in result
        assert "D" in result

    def test_empty_relationships(self, formatter):
        result = formatter.class_diagram([])
        assert "classDiagram" in result

    def test_special_chars_in_names(self, formatter):
        relationships = [
            {"from": "List<int>", "to": "Collection<T>", "type": "inherits"}
        ]
        result = formatter.class_diagram(relationships)
        # Should contain escaped labels
        assert "classDiagram" in result
        assert "&lt;int&gt;" in result or "List" in result


class TestClassDiagramTuple:
    """Tests for the tuple-based class_diagram API (task 13.2 spec)."""

    def test_inheritance(self, formatter):
        relationships = [("Child", "Parent", "inheritance")]
        result = formatter.class_diagram(relationships)
        assert "%% Class Diagram" in result
        assert "classDiagram" in result
        assert "Child <|-- Parent" in result

    def test_composition(self, formatter):
        relationships = [("Car", "Engine", "composition")]
        result = formatter.class_diagram(relationships)
        assert "Car *-- Engine" in result

    def test_call(self, formatter):
        relationships = [("ServiceA", "ServiceB", "call")]
        result = formatter.class_diagram(relationships)
        assert "ServiceA ..> ServiceB" in result

    def test_dependency(self, formatter):
        relationships = [("ModuleA", "ModuleB", "dependency")]
        result = formatter.class_diagram(relationships)
        assert "ModuleA --> ModuleB" in result

    def test_all_entities_present(self, formatter):
        relationships = [
            ("A", "B", "inheritance"),
            ("C", "D", "composition"),
            ("A", "D", "call"),
        ]
        result = formatter.class_diagram(relationships)
        for entity in ["A", "B", "C", "D"]:
            assert entity in result

    def test_header_comment(self, formatter):
        relationships = [("A", "B", "inheritance")]
        result = formatter.class_diagram(relationships)
        assert "%% Class Diagram" in result

    def test_special_chars_escaped(self, formatter):
        relationships = [("my.module", "base.class", "inheritance")]
        result = formatter.class_diagram(relationships)
        assert "my_module" in result
        assert "base_class" in result
        # Should have class declarations with original names
        assert 'class my_module["my.module"]' in result
        assert 'class base_class["base.class"]' in result

    def test_unknown_type_defaults_to_dependency(self, formatter):
        relationships = [("A", "B", "unknown")]
        result = formatter.class_diagram(relationships)
        assert "A --> B" in result


class TestSequenceDiagram:
    def test_basic_call_chain(self, formatter):
        call_chain = CallGraph(
            root="main",
            edges=[("main", "service"), ("service", "repository")],
            max_depth_reached=False,
            depth=2,
        )
        result = formatter.sequence_diagram(call_chain)
        assert "sequenceDiagram" in result
        assert "main" in result
        assert "service" in result
        assert "repository" in result
        assert "->>" in result

    def test_max_depth_truncation(self, formatter):
        call_chain = CallGraph(
            root="entry",
            edges=[("entry", "middle"), ("middle", "deep")],
            max_depth_reached=True,
            depth=10,
        )
        result = formatter.sequence_diagram(call_chain)
        assert "Max depth reached" in result
        assert "truncated" in result

    def test_no_truncation_note_when_not_maxed(self, formatter):
        call_chain = CallGraph(
            root="a",
            edges=[("a", "b")],
            max_depth_reached=False,
            depth=1,
        )
        result = formatter.sequence_diagram(call_chain)
        assert "truncated" not in result

    def test_empty_edges(self, formatter):
        call_chain = CallGraph(root="solo", edges=[], max_depth_reached=False, depth=0)
        result = formatter.sequence_diagram(call_chain)
        assert "sequenceDiagram" in result
        assert "solo" in result

    def test_all_participants_declared(self, formatter):
        call_chain = CallGraph(
            root="a",
            edges=[("a", "b"), ("b", "c"), ("a", "d")],
            max_depth_reached=False,
            depth=2,
        )
        result = formatter.sequence_diagram(call_chain)
        assert "participant" in result
        # All nodes appear
        for name in ["a", "b", "c", "d"]:
            assert name in result

    def test_header_comment(self, formatter):
        call_chain = CallGraph(root="main", edges=[], depth=0)
        result = formatter.sequence_diagram(call_chain)
        assert "%% Sequence Diagram" in result

    def test_special_chars_in_participant_names(self, formatter):
        call_chain = CallGraph(
            root="my.module:func",
            edges=[("my.module:func", "other.module:bar")],
            depth=1,
        )
        result = formatter.sequence_diagram(call_chain)
        # Should use alias syntax for participants with special chars
        assert "my_module_func" in result
        assert "other_module_bar" in result


class TestDependencyGraph:
    def test_basic_dependencies(self, formatter):
        imports = ImportGraph(
            module="codesense.agent",
            internal_deps=["codesense.models", "codesense.llm"],
            external_deps=["langgraph", "langchain"],
            env_vars=[],
            external_apis=[],
        )
        result = formatter.dependency_graph(imports)
        assert "flowchart LR" in result
        assert "codesense.agent" in result or "codesense_agent" in result
        assert "codesense.models" in result or "codesense_models" in result
        assert "langgraph" in result
        assert "-->" in result

    def test_no_dependencies(self, formatter):
        imports = ImportGraph(
            module="standalone",
            internal_deps=[],
            external_deps=[],
            env_vars=[],
            external_apis=[],
        )
        result = formatter.dependency_graph(imports)
        assert "flowchart LR" in result
        assert "standalone" in result

    def test_external_deps_different_shape(self, formatter):
        imports = ImportGraph(
            module="app",
            internal_deps=["internal"],
            external_deps=["external_lib"],
            env_vars=[],
            external_apis=[],
        )
        result = formatter.dependency_graph(imports)
        # Internal uses square brackets, external uses stadium shape
        assert "internal" in result
        assert "external_lib" in result

    def test_header_comment(self, formatter):
        imports = ImportGraph(
            module="mod",
            internal_deps=[],
            external_deps=[],
            env_vars=[],
            external_apis=[],
        )
        result = formatter.dependency_graph(imports)
        assert "%% Dependency Graph" in result

    def test_internal_deps_solid_arrow(self, formatter):
        imports = ImportGraph(
            module="myapp",
            internal_deps=["myapp.utils"],
            external_deps=[],
            env_vars=[],
            external_apis=[],
        )
        result = formatter.dependency_graph(imports)
        assert "myapp --> myapp_utils" in result

    def test_all_nodes_have_labels(self, formatter):
        imports = ImportGraph(
            module="my.app",
            internal_deps=["my.utils"],
            external_deps=["some.lib"],
            env_vars=[],
            external_apis=[],
        )
        result = formatter.dependency_graph(imports)
        assert '"my.app"' in result
        assert '"my.utils"' in result
        assert '"some.lib"' in result


class TestFlowchart:
    def test_basic_flowchart(self, formatter):
        nodes = [
            {"id": "start", "label": "Start"},
            {"id": "process", "label": "Process Data"},
            {"id": "end", "label": "End"},
        ]
        edges = [("start", "process"), ("process", "end")]
        result = formatter.flowchart(nodes, edges)
        assert result.startswith("flowchart TD")
        assert "Start" in result
        assert "Process Data" in result
        assert "End" in result
        assert "-->" in result

    def test_diamond_shape(self, formatter):
        nodes = [
            {"id": "decision", "label": "Is valid?", "shape": "diamond"},
        ]
        result = formatter.flowchart(nodes, [])
        assert "{" in result  # diamond uses curly braces

    def test_round_shape(self, formatter):
        nodes = [
            {"id": "step", "label": "Step 1", "shape": "round"},
        ]
        result = formatter.flowchart(nodes, [])
        assert '("Step 1")' in result

    def test_empty_flowchart(self, formatter):
        result = formatter.flowchart([], [])
        assert result == "flowchart TD"

    def test_special_chars_in_labels(self, formatter):
        nodes = [
            {"id": "node1", "label": 'Say "hi" & <bye>'},
        ]
        result = formatter.flowchart(nodes, [])
        # Quotes should be escaped
        assert "'" in result or "&lt;" in result
