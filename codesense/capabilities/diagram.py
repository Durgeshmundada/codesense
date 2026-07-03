"""Diagram capability handler — generates Mermaid structural diagrams.

Parses code scope to extract module, class, and function relationships
(inheritance, composition, calls) and generates Mermaid diagrams.

Supports diagram types:
- flowchart (default): Module/function dependency flowchart
- sequence: Call sequence between functions/methods using CallGraphBuilder
- architecture: Class diagram showing inheritance, composition, calls

Requirements: 5.5, 12.1, 12.2, 12.5
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from codesense.analysis.ast_walker import ASTWalker
from codesense.analysis.call_graph import CallGraphBuilder
from codesense.models.analysis import CallGraph, ModuleInfo
from codesense.models.output import CodeSnippet, CommandOutput, CommandParams
from codesense.output.markdown_writer import MarkdownWriter
from codesense.output.mermaid_formatter import MermaidFormatter

logger = logging.getLogger(__name__)

DIAGRAM_TITLE = "Code Diagram"


class DiagramHandler:
    """Capability handler for the 'diagram' command.

    Analyzes Python source files using ASTWalker to extract structural
    relationships (inheritance, composition, calls) and generates Mermaid
    diagrams via MermaidFormatter.

    Supports three diagram types:
    - flowchart: Shows module/function call dependencies as a directed graph
    - sequence: Shows call sequences using CallGraphBuilder
    - architecture: Shows class relationships (inheritance, composition, method calls)

    Args:
        project_root: Optional root directory for resolving relative paths.
    """

    def __init__(self, project_root: Optional[str] = None) -> None:
        self._ast_walker = ASTWalker()
        self._mermaid = MermaidFormatter()
        self._markdown_writer = MarkdownWriter()
        self._project_root = project_root or "."

    def run(self, params: CommandParams) -> CommandOutput:
        """Execute the diagram capability.

        Args:
            params: Parsed CLI arguments. Uses:
                - params.path: File or directory path to diagram.
                - params.query: Optional diagram type (flowchart/sequence/architecture).
                - params.output: Optional output file path for the diagram.
                - params.mock: Whether demo mode is active.

        Returns:
            CommandOutput with:
                - title: "Code Diagram"
                - content: the Mermaid diagram as a code block
                - is_demo_mode: from params.mock
        """
        target_path = params.path or "."
        diagram_type = self._resolve_diagram_type(params)
        output_path = params.output

        # Collect Python files to analyze
        files = self._collect_python_files(target_path)

        if not files:
            return CommandOutput(
                title=DIAGRAM_TITLE,
                content=f"No Python files found at path: {target_path}",
                is_demo_mode=params.mock,
            )

        # Parse all files and collect structural info
        modules, parse_errors = self._parse_files(files)

        if not modules and parse_errors:
            # All files failed to parse — return descriptive error
            error_details = "\n".join(
                f"- {path}: {error}" for path, error in parse_errors
            )
            return CommandOutput(
                title=DIAGRAM_TITLE,
                content=(
                    f"Failed to parse files for structural analysis:\n"
                    f"{error_details}"
                ),
                is_demo_mode=params.mock,
            )

        # Generate the diagram based on type
        mermaid_diagram = self._generate_diagram(modules, diagram_type)

        # The content is the Mermaid diagram as a code block
        content = f"```mermaid\n{mermaid_diagram}\n```"

        if parse_errors:
            error_info = self._format_parse_errors(parse_errors)
            content = f"{error_info}\n\n{content}"

        # Write to file if output path specified
        if output_path:
            self._write_output(mermaid_diagram, output_path)
            content += f"\n\nDiagram written to: {output_path}"

        return CommandOutput(
            title=DIAGRAM_TITLE,
            content=content,
            code_snippets=[
                CodeSnippet(
                    code=mermaid_diagram,
                    language="mermaid",
                    label=f"{diagram_type} diagram",
                )
            ],
            is_demo_mode=params.mock,
        )

    def _resolve_diagram_type(self, params: CommandParams) -> str:
        """Resolve the diagram type from params.

        The diagram type is passed via the query field from the CLI
        (since CommandParams doesn't have a dedicated type field).
        Defaults to 'flowchart' if not specified.

        Supported types: flowchart, sequence, architecture
        """
        if params.query and params.query in ("flowchart", "sequence", "architecture"):
            return params.query
        return "flowchart"

    def _collect_python_files(self, target_path: str) -> list[str]:
        """Collect Python files from the target path.

        If target_path is a file, returns it as a single-element list.
        If it's a directory, recursively finds all .py files.

        Returns:
            List of absolute file path strings.
        """
        path = Path(target_path)

        if not path.exists():
            # Try resolving relative to project root
            path = Path(self._project_root) / target_path
            if not path.exists():
                return []

        if path.is_file():
            if path.suffix == ".py":
                return [str(path.resolve())]
            return []

        # Directory: collect all .py files recursively
        py_files: list[str] = []
        for py_file in sorted(path.rglob("*.py")):
            # Skip __pycache__ and hidden directories
            parts = py_file.parts
            if any(
                part.startswith(".") or part == "__pycache__" for part in parts
            ):
                continue
            py_files.append(str(py_file.resolve()))

        return py_files

    def _parse_files(
        self, files: list[str]
    ) -> tuple[list[ModuleInfo], list[tuple[str, str]]]:
        """Parse Python files using ASTWalker.

        Args:
            files: List of file paths to parse.

        Returns:
            Tuple of (successfully parsed modules, list of (path, error) for failures).
        """
        modules: list[ModuleInfo] = []
        errors: list[tuple[str, str]] = []

        for file_path in files:
            try:
                module_info = self._ast_walker.parse_module(file_path)
                # Only include modules that have meaningful content
                if (
                    module_info.classes
                    or module_info.functions
                    or module_info.imports
                ):
                    modules.append(module_info)
                else:
                    # Check if file has actual content
                    file_content = Path(file_path).read_text(
                        encoding="utf-8"
                    ).strip()
                    if file_content:
                        errors.append(
                            (
                                file_path,
                                "File could not be parsed or contains no "
                                "analyzable code",
                            )
                        )
            except ValueError as e:
                errors.append((file_path, str(e)))
            except Exception as e:
                logger.warning("Unexpected error parsing %s: %s", file_path, e)
                errors.append((file_path, f"Unexpected error: {e}"))

        return modules, errors

    def _generate_diagram(
        self, modules: list[ModuleInfo], diagram_type: str
    ) -> str:
        """Generate a Mermaid diagram from parsed module information.

        Args:
            modules: List of parsed ModuleInfo objects.
            diagram_type: One of 'flowchart', 'sequence', 'architecture'.

        Returns:
            Mermaid diagram string.
        """
        if diagram_type == "sequence":
            return self._generate_sequence_diagram(modules)
        elif diagram_type == "architecture":
            return self._generate_architecture_diagram(modules)
        else:
            return self._generate_flowchart_diagram(modules)

    def _generate_flowchart_diagram(self, modules: list[ModuleInfo]) -> str:
        """Generate a flowchart showing module and function relationships.

        Shows modules as nodes with edges representing function calls
        and imports between them. Uses MermaidFormatter.flowchart().
        """
        nodes: list[dict] = []
        edges: list[tuple[str, str]] = []
        seen_nodes: set[str] = set()

        for module in modules:
            module_name = Path(module.path).stem
            if module_name not in seen_nodes:
                nodes.append({
                    "id": module_name,
                    "label": module_name,
                    "shape": "box",
                })
                seen_nodes.add(module_name)

            # Add function nodes and their call edges
            for func in module.functions:
                func_id = f"{module_name}.{func.name}"
                if func_id not in seen_nodes:
                    nodes.append({
                        "id": func_id,
                        "label": f"{func.name}()",
                        "shape": "round",
                    })
                    seen_nodes.add(func_id)
                    edges.append((module_name, func_id))

                # Add edges for function calls
                for call in func.calls:
                    call_id = call
                    if call_id not in seen_nodes:
                        nodes.append({
                            "id": call_id,
                            "label": call,
                            "shape": "round",
                        })
                        seen_nodes.add(call_id)
                    edges.append((func_id, call_id))

            # Add class nodes
            for cls in module.classes:
                cls_id = f"{module_name}.{cls.name}"
                if cls_id not in seen_nodes:
                    nodes.append({
                        "id": cls_id,
                        "label": cls.name,
                        "shape": "stadium",
                    })
                    seen_nodes.add(cls_id)
                    edges.append((module_name, cls_id))

        return self._mermaid.flowchart(nodes, edges)

    def _generate_sequence_diagram(self, modules: list[ModuleInfo]) -> str:
        """Generate a sequence diagram showing call chains.

        Uses CallGraphBuilder to trace static execution paths from the first
        module's entry point and produces a Mermaid sequence diagram via
        MermaidFormatter.sequence_diagram().
        """
        if modules:
            entry_path = modules[0].path
            builder = CallGraphBuilder(
                project_root=self._project_root,
                ast_walker=self._ast_walker,
            )
            call_graph = builder.build(entry_point=entry_path, max_depth=10)
        else:
            call_graph = CallGraph(
                root="main", edges=[], max_depth_reached=False, depth=0
            )

        return self._mermaid.sequence_diagram(call_graph)

    def _generate_architecture_diagram(self, modules: list[ModuleInfo]) -> str:
        """Generate a class diagram showing inheritance, composition, and calls.

        Uses MermaidFormatter.class_diagram() with module relationships
        expressed as tuples of (source, target, relationship_type).
        """
        relationships: list[tuple[str, str, str]] = []

        for module in modules:
            for cls in module.classes:
                # Inheritance relationships
                for base in cls.bases:
                    relationships.append((cls.name, base, "inheritance"))

                # Detect composition and call relationships from method calls
                for method in cls.methods:
                    for call in method.calls:
                        if "." in call:
                            parts = call.split(".")
                            # self.attr.method() pattern → composition
                            if parts[0] == "self" and len(parts) > 2:
                                target_attr = parts[1]
                                for other_module in modules:
                                    for other_cls in other_module.classes:
                                        if (
                                            other_cls.name.lower()
                                            == target_attr.lower()
                                            or (
                                                target_attr.startswith("_")
                                                and other_cls.name.lower()
                                                == target_attr.lstrip(
                                                    "_"
                                                ).lower()
                                            )
                                        ):
                                            rel = (
                                                cls.name,
                                                other_cls.name,
                                                "composition",
                                            )
                                            if rel not in relationships:
                                                relationships.append(rel)
                            elif parts[0] != "self":
                                # Direct class reference: ClassName.method()
                                target_class = parts[0]
                                for other_module in modules:
                                    for other_cls in other_module.classes:
                                        if other_cls.name == target_class:
                                            rel = (
                                                cls.name,
                                                other_cls.name,
                                                "call",
                                            )
                                            if rel not in relationships:
                                                relationships.append(rel)

            # Module-level function calls to classes
            for func in module.functions:
                for call in func.calls:
                    if "." in call:
                        parts = call.split(".")
                        target_class = parts[0]
                        for other_module in modules:
                            for other_cls in other_module.classes:
                                if other_cls.name == target_class:
                                    rel = (func.name, other_cls.name, "call")
                                    if rel not in relationships:
                                        relationships.append(rel)

        # If no relationships found but we have classes, show them
        if not relationships:
            for module in modules:
                for cls in module.classes:
                    if cls.bases:
                        for base in cls.bases:
                            relationships.append(
                                (cls.name, base, "inheritance")
                            )
                    else:
                        module_name = Path(module.path).stem
                        relationships.append(
                            (module_name, cls.name, "dependency")
                        )

        if not relationships:
            # Fallback: create relationships from functions
            for module in modules:
                module_name = Path(module.path).stem
                for func in module.functions:
                    relationships.append(
                        (module_name, func.name, "dependency")
                    )

        return self._mermaid.class_diagram(relationships)

    def _format_parse_errors(
        self,
        parse_errors: list[tuple[str, str]],
    ) -> str:
        """Format parse errors into a human-readable warning string.

        Args:
            parse_errors: List of (path, error) tuples for files that failed.

        Returns:
            Warning string listing failed files and their error reasons.
        """
        lines = [f"⚠️ {len(parse_errors)} file(s) could not be parsed:"]
        for path, error in parse_errors:
            short_path = Path(path).name
            lines.append(f"- {short_path}: {error}")
        return "\n".join(lines)

    def _write_output(self, mermaid_content: str, output_path: str) -> None:
        """Write the Mermaid diagram to a file using MarkdownWriter.

        Uses MarkdownWriter.build_diagram_doc() to wrap the Mermaid content
        in a proper markdown document with title and code fence.

        Args:
            mermaid_content: The Mermaid diagram string to write.
            output_path: Destination file path.
        """
        try:
            doc_content = self._markdown_writer.build_diagram_doc(
                title=DIAGRAM_TITLE, mermaid_content=mermaid_content
            )
            self._markdown_writer.write_file(doc_content, output_path)
            logger.info("Diagram written to %s", output_path)
        except OSError as e:
            logger.error("Failed to write diagram to %s: %s", output_path, e)
