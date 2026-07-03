"""Mermaid diagram generation for structural and flow visualizations."""

import re
from typing import Union

from codesense.models.analysis import CallGraph, ImportGraph


def _sanitize_id(name: str) -> str:
    """Sanitize a string to be a valid Mermaid node ID.

    Mermaid IDs must be alphanumeric (plus underscores).
    Replace any non-alphanumeric/underscore characters with underscores.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    # Ensure it doesn't start with a digit
    if sanitized and sanitized[0].isdigit():
        sanitized = f"n_{sanitized}"
    # Ensure non-empty
    if not sanitized:
        sanitized = "node"
    return sanitized


def _escape_label(text: str) -> str:
    """Escape special characters in Mermaid labels."""
    # Escape quotes and other special chars for display labels
    text = text.replace('"', "'")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


class MermaidFormatter:
    """Generates valid Mermaid diagram syntax from structured data."""

    # Arrow map for tuple-based relationship types (task 13.2 spec)
    _TUPLE_ARROW_MAP = {
        "inheritance": "<|--",
        "composition": "*--",
        "call": "..>",
        "dependency": "-->",
    }

    # Arrow map for dict-based relationship types (legacy/existing)
    _DICT_ARROW_MAP = {
        "inherits": "<|--",
        "uses": "..>",
        "calls": "-->",
    }

    def class_diagram(
        self, relationships: Union[list[tuple[str, str, str]], list[dict]]
    ) -> str:
        """Generate a Mermaid class diagram from relationship data.

        Supports two input formats:
        1. Tuple format: list of (source, target, relationship_type) tuples
           relationship_type: "inheritance", "composition", "call", "dependency"
        2. Dict format: list of dicts with "from", "to", "type" keys
           type: "inherits", "uses", "calls"

        Returns:
            Valid Mermaid classDiagram syntax string with title comment.
        """
        lines = ["%% Class Diagram", "classDiagram"]

        # Collect all entities to ensure they appear in output
        entities: set[str] = set()
        rel_lines: list[str] = []

        for rel in relationships:
            if isinstance(rel, tuple):
                source, target, rel_type = rel
                entities.add(source)
                entities.add(target)
                arrow = self._TUPLE_ARROW_MAP.get(rel_type, "-->")
                from_id = _sanitize_id(source)
                to_id = _sanitize_id(target)
                rel_lines.append(f"    {from_id} {arrow} {to_id}")
            else:
                # Dict format (legacy)
                from_name = rel["from"]
                to_name = rel["to"]
                entities.add(from_name)
                entities.add(to_name)
                rel_type = rel.get("type", "uses")
                arrow = self._DICT_ARROW_MAP.get(rel_type, "-->")
                from_id = _sanitize_id(from_name)
                to_id = _sanitize_id(to_name)
                rel_lines.append(f"    {to_id} {arrow} {from_id}")

        # Declare all classes
        for entity in sorted(entities):
            safe_id = _sanitize_id(entity)
            if safe_id != entity:
                lines.append(
                    f'    class {safe_id}["{_escape_label(entity)}"]'
                )
            else:
                lines.append(f"    class {safe_id}")

        # Add relationships
        lines.extend(rel_lines)

        return "\n".join(lines)

    def sequence_diagram(self, call_chain: CallGraph) -> str:
        """Generate a Mermaid sequence diagram from a CallGraph.

        Args:
            call_chain: CallGraph with root, edges (caller, callee), and
                max_depth_reached flag.

        Returns:
            Valid Mermaid sequenceDiagram syntax string with title comment.
            Includes truncation note if max_depth_reached is True.
        """
        lines = ["%% Sequence Diagram", "sequenceDiagram"]

        # Collect all participants from edges
        participants: list[str] = []
        seen: set[str] = set()

        # Add root first
        if call_chain.root:
            participants.append(call_chain.root)
            seen.add(call_chain.root)

        # Add remaining participants in order of appearance
        for caller, callee in call_chain.edges:
            if caller not in seen:
                participants.append(caller)
                seen.add(caller)
            if callee not in seen:
                participants.append(callee)
                seen.add(callee)

        # Declare participants
        for participant in participants:
            safe_name = _escape_label(participant)
            lines.append(
                f"    participant {_sanitize_id(participant)} as {safe_name}"
            )

        # Add message arrows for each edge
        for caller, callee in call_chain.edges:
            caller_id = _sanitize_id(caller)
            callee_id = _sanitize_id(callee)
            lines.append(f"    {caller_id}->>+{callee_id}: calls")

        # Indicate truncation if max depth reached
        if call_chain.max_depth_reached:
            last_participant = (
                _sanitize_id(participants[-1]) if participants else "root"
            )
            lines.append(
                f"    Note right of {last_participant}: "
                "Max depth reached, trace truncated"
            )

        return "\n".join(lines)

    def dependency_graph(self, imports: ImportGraph) -> str:
        """Generate a Mermaid flowchart LR from an ImportGraph.

        Args:
            imports: ImportGraph with module, internal_deps, and external_deps.

        Returns:
            Valid Mermaid flowchart LR syntax string with title comment,
            showing the module connected to its internal and external
            dependencies.
        """
        lines = ["%% Dependency Graph", "flowchart LR"]

        module_id = _sanitize_id(imports.module)
        module_label = _escape_label(imports.module)
        lines.append(f'    {module_id}["{module_label}"]')

        # Internal dependencies
        for dep in imports.internal_deps:
            dep_id = _sanitize_id(dep)
            dep_label = _escape_label(dep)
            lines.append(f'    {dep_id}["{dep_label}"]')
            lines.append(f"    {module_id} --> {dep_id}")

        # External dependencies (different shape - stadium/rounded)
        for dep in imports.external_deps:
            dep_id = _sanitize_id(dep)
            dep_label = _escape_label(dep)
            lines.append(f'    {dep_id}(["{dep_label}"])')
            lines.append(f"    {module_id} --> {dep_id}")

        return "\n".join(lines)

    def flowchart(self, nodes: list[dict], edges: list[tuple[str, str]]) -> str:
        """Generate a generic Mermaid flowchart TD from nodes and edges.

        Args:
            nodes: List of dicts with at minimum an "id" key. May also have
                "label" for display text and "shape" (default "box").
            edges: List of (source_id, target_id) tuples.

        Returns:
            Valid Mermaid flowchart TD syntax string.
        """
        lines = ["flowchart TD"]

        # Declare nodes
        for node in nodes:
            node_id = _sanitize_id(node["id"])
            label = _escape_label(node.get("label", node["id"]))
            shape = node.get("shape", "box")

            if shape == "round":
                lines.append(f'    {node_id}("{label}")')
            elif shape == "diamond":
                lines.append(f'    {node_id}{{"{label}"}}')
            elif shape == "stadium":
                lines.append(f'    {node_id}(["{label}"])')
            else:  # default box
                lines.append(f'    {node_id}["{label}"]')

        # Add edges
        for source, target in edges:
            source_id = _sanitize_id(source)
            target_id = _sanitize_id(target)
            lines.append(f"    {source_id} --> {target_id}")

        return "\n".join(lines)
