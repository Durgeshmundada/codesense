"""Property-based tests for MermaidFormatter.

Tests Property 15 from the design document using Hypothesis.

**Validates: Requirements 5.5, 12.2, 12.3, 12.4**
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from codesense.models.analysis import CallGraph, ImportGraph
from codesense.output.mermaid_formatter import MermaidFormatter, _sanitize_id


# --- Strategies ---

# Strategy for valid identifiers used as entity names
identifier_st = st.from_regex(r"[a-zA-Z][a-zA-Z0-9_]{0,29}", fullmatch=True)

# Strategy for relationship types supported by class_diagram (tuple format)
rel_type_st = st.sampled_from(["inheritance", "composition", "call", "dependency"])

# Strategy for a single relationship tuple (source, target, rel_type)
relationship_tuple_st = st.tuples(identifier_st, identifier_st, rel_type_st)

# Strategy for a non-empty list of relationship tuples
relationships_st = st.lists(relationship_tuple_st, min_size=1, max_size=20)


# --- Property 15 Tests ---


# Feature: codesense, Property 15: Mermaid output syntactic validity
@settings(max_examples=100)
@given(relationships=relationships_st)
def test_class_diagram_contains_all_entities(
    relationships: list[tuple[str, str, str]],
) -> None:
    """For any list of (source, target, rel_type) tuples, the generated
    classDiagram contains both source and target names (either as raw name
    or as sanitized ID).

    **Validates: Requirements 5.5, 12.2**
    """
    formatter = MermaidFormatter()
    output = formatter.class_diagram(relationships)

    # Collect all entities from input
    entities: set[str] = set()
    for source, target, _ in relationships:
        entities.add(source)
        entities.add(target)

    # Every entity must be present in the output (either as raw name or sanitized ID)
    for entity in entities:
        sanitized = _sanitize_id(entity)
        assert sanitized in output or entity in output, (
            f"Entity '{entity}' (sanitized: '{sanitized}') not found in class diagram output:\n{output}"
        )


# Feature: codesense, Property 15: Mermaid output syntactic validity
@settings(max_examples=100)
@given(
    root=identifier_st,
    edges=st.lists(
        st.tuples(identifier_st, identifier_st),
        min_size=1,
        max_size=15,
    ),
)
def test_sequence_diagram_contains_all_participants(
    root: str, edges: list[tuple[str, str]]
) -> None:
    """For any CallGraph with edges, the generated sequenceDiagram declares all
    nodes as participants.

    **Validates: Requirements 5.5, 12.3**
    """
    call_graph = CallGraph(root=root, edges=edges)
    formatter = MermaidFormatter()
    output = formatter.sequence_diagram(call_graph)

    # Collect all nodes from the CallGraph
    nodes: set[str] = {root}
    for caller, callee in edges:
        nodes.add(caller)
        nodes.add(callee)

    # Every node must appear as a participant in the output
    for node in nodes:
        sanitized = _sanitize_id(node)
        assert sanitized in output or node in output, (
            f"Node '{node}' (sanitized: '{sanitized}') not found as participant in sequence diagram:\n{output}"
        )


# Feature: codesense, Property 15: Mermaid output syntactic validity
@settings(max_examples=100)
@given(
    module=identifier_st,
    internal_deps=st.lists(identifier_st, min_size=0, max_size=10),
    external_deps=st.lists(identifier_st, min_size=0, max_size=10),
)
def test_dependency_graph_contains_all_nodes(
    module: str, internal_deps: list[str], external_deps: list[str]
) -> None:
    """For any ImportGraph, the generated flowchart LR contains all internal
    and external dependencies.

    **Validates: Requirements 5.5, 12.4**
    """
    import_graph = ImportGraph(
        module=module,
        internal_deps=internal_deps,
        external_deps=external_deps,
        env_vars=[],
        external_apis=[],
    )
    formatter = MermaidFormatter()
    output = formatter.dependency_graph(import_graph)

    # The module itself must appear
    module_sanitized = _sanitize_id(module)
    assert module_sanitized in output or module in output, (
        f"Module '{module}' (sanitized: '{module_sanitized}') not found in dependency graph:\n{output}"
    )

    # All internal deps must appear
    for dep in internal_deps:
        sanitized = _sanitize_id(dep)
        assert sanitized in output or dep in output, (
            f"Internal dep '{dep}' (sanitized: '{sanitized}') not found in dependency graph:\n{output}"
        )

    # All external deps must appear
    for dep in external_deps:
        sanitized = _sanitize_id(dep)
        assert sanitized in output or dep in output, (
            f"External dep '{dep}' (sanitized: '{sanitized}') not found in dependency graph:\n{output}"
        )


# Feature: codesense, Property 15: Mermaid output syntactic validity
@settings(max_examples=100)
@given(
    diagram_type=st.sampled_from(["class", "sequence", "dependency"]),
    data=st.data(),
)
def test_mermaid_starts_with_valid_directive(
    diagram_type: str, data: st.DataObject
) -> None:
    """All generated diagrams start with a valid Mermaid directive
    (classDiagram, sequenceDiagram, or flowchart).

    **Validates: Requirements 5.5, 12.2, 12.3, 12.4**
    """
    formatter = MermaidFormatter()

    valid_directives = ("classDiagram", "sequenceDiagram", "flowchart")

    if diagram_type == "class":
        relationships = data.draw(relationships_st)
        output = formatter.class_diagram(relationships)
    elif diagram_type == "sequence":
        root = data.draw(identifier_st)
        edges = data.draw(
            st.lists(st.tuples(identifier_st, identifier_st), min_size=1, max_size=10)
        )
        call_graph = CallGraph(root=root, edges=edges)
        output = formatter.sequence_diagram(call_graph)
    else:  # dependency
        module = data.draw(identifier_st)
        internal = data.draw(st.lists(identifier_st, min_size=0, max_size=5))
        external = data.draw(st.lists(identifier_st, min_size=0, max_size=5))
        import_graph = ImportGraph(
            module=module,
            internal_deps=internal,
            external_deps=external,
            env_vars=[],
            external_apis=[],
        )
        output = formatter.dependency_graph(import_graph)

    # The output (after the optional comment line) must contain a valid directive
    lines = output.strip().split("\n")
    # First line is typically a comment (%% ...), directive is on the second line
    directive_line = lines[1] if len(lines) > 1 else lines[0]
    directive_found = any(
        directive_line.strip().startswith(d) for d in valid_directives
    )
    assert directive_found, (
        f"Diagram output does not start with a valid Mermaid directive.\n"
        f"Expected one of {valid_directives}, got line: '{directive_line}'\n"
        f"Full output:\n{output}"
    )
