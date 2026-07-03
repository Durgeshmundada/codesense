"""Flow capability handler — displays execution flow with call sequence and Mermaid diagram.

Traces the static execution path from an entry point file/function using
CallGraphBuilder, then renders a numbered text description of the flow
plus a Mermaid sequence diagram.

Requirements: 5.4, 12.3, 12.6
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from codesense.analysis.call_graph import CallGraphBuilder
from codesense.models.analysis import CallGraph
from codesense.models.output import CodeSnippet, CommandOutput, CommandParams
from codesense.output.mermaid_formatter import MermaidFormatter

logger = logging.getLogger(__name__)

FLOW_TITLE = "Execution Flow"


class FlowHandler:
    """Capability handler for the 'flow' command.

    Traces static execution paths from an entry point using CallGraphBuilder
    with max_depth=10, then formats the result as a numbered step list and
    a Mermaid sequence diagram.

    Implements the CapabilityHandler protocol: run(params) -> CommandOutput.

    Args:
        project_root: Root directory of the project to analyze.
            Defaults to the current working directory.
    """

    def __init__(self, project_root: Optional[str] = None) -> None:
        self._project_root = project_root or str(Path.cwd())

    def run(self, params: CommandParams) -> CommandOutput:
        """Execute the flow capability with the given parameters.

        Uses CallGraphBuilder to trace the execution path from params.path
        (the entry point), builds a CallGraph with max_depth=10, then formats
        the result as a numbered step list and Mermaid sequence diagram.

        Args:
            params: Parsed CLI arguments. Uses:
                - params.path: Entry point (file path or file::function format).
                - params.mock: Whether demo mode is active.

        Returns:
            CommandOutput with:
                - title: "Execution Flow"
                - content: numbered step list + Mermaid code block
                - is_demo_mode: from params.mock
        """
        entry_point = params.path or "."
        is_demo = params.mock

        # Build the call graph using CallGraphBuilder with max_depth=10
        try:
            builder = CallGraphBuilder(project_root=self._project_root)
            call_graph = builder.build(entry_point=entry_point, max_depth=10)
        except Exception as e:
            logger.error("Failed to build call graph for '%s': %s", entry_point, e)
            return CommandOutput(
                title=FLOW_TITLE,
                content=(
                    f"Error: Unable to analyze execution flow for '{entry_point}'.\n"
                    f"Could not parse the entry point file: {e}"
                ),
                is_demo_mode=is_demo,
            )

        # Check for parse failures (empty graph with a valid entry point file)
        parse_errors = self._check_parse_errors(entry_point, call_graph)

        # Generate numbered text flow
        numbered_flow = self._build_numbered_flow(call_graph)

        # Generate Mermaid sequence diagram
        formatter = MermaidFormatter()
        mermaid_diagram = formatter.sequence_diagram(call_graph)

        # Compose content: numbered flow + Mermaid code block
        content_parts: list[str] = [numbered_flow]

        if parse_errors:
            content_parts.append("")
            content_parts.append("⚠️ Some files could not be analyzed:")
            for error in parse_errors:
                content_parts.append(f"  - {error}")

        content_parts.append("")
        content_parts.append("### Sequence Diagram")
        content_parts.append("")
        content_parts.append(f"```mermaid\n{mermaid_diagram}\n```")

        content = "\n".join(content_parts)

        return CommandOutput(
            title=FLOW_TITLE,
            content=content,
            code_snippets=[
                CodeSnippet(
                    code=mermaid_diagram,
                    language="mermaid",
                    label="Execution Flow Sequence Diagram",
                )
            ],
            is_demo_mode=is_demo,
        )

    def _build_numbered_flow(self, call_graph: CallGraph) -> str:
        """Build a numbered text description of the execution flow.

        Each edge in the call graph becomes a numbered step showing
        the caller-to-callee relationship.

        Args:
            call_graph: The call graph produced by CallGraphBuilder.

        Returns:
            A multi-line string with numbered execution steps.
        """
        if not call_graph.edges:
            return "No execution flow detected from the entry point."

        lines: list[str] = []
        lines.append(f"Entry point: {call_graph.root}")
        lines.append("")

        for i, (caller, callee) in enumerate(call_graph.edges, start=1):
            lines.append(f"{i}. {caller} → {callee}")

        if call_graph.max_depth_reached:
            lines.append("")
            lines.append(
                f"⚠️  Trace truncated at depth {call_graph.depth} "
                f"(max depth of 10 reached). Deeper calls exist but are not shown."
            )

        return "\n".join(lines)

    def _check_parse_errors(
        self, entry_point: str, call_graph: CallGraph
    ) -> list[str]:
        """Check for files that could not be parsed during call graph construction.

        Inspects the entry point path and reports if the file couldn't be read
        or parsed, based on the call graph result.

        Args:
            entry_point: The original entry point string.
            call_graph: The resulting call graph.

        Returns:
            List of error message strings for files that couldn't be analyzed.
        """
        errors: list[str] = []

        # Parse the entry point to get the file path
        file_path = entry_point.split("::")[0] if "::" in entry_point else entry_point

        # If we have no edges and the entry point references a file that exists,
        # it may have failed to parse
        if not call_graph.edges and file_path != ".":
            path = Path(file_path)
            if not path.is_absolute():
                path = Path(self._project_root) / path

            if path.exists() and path.is_file():
                # File exists but produced no edges — could be parse failure or
                # simply no calls in the file. Check if it's a valid Python file.
                if path.suffix == ".py":
                    try:
                        source = path.read_text(encoding="utf-8")
                        import ast
                        ast.parse(source, filename=str(path))
                    except SyntaxError as e:
                        errors.append(
                            f"{file_path}: Syntax error at line {e.lineno} — {e.msg}"
                        )
                    except (OSError, UnicodeDecodeError) as e:
                        errors.append(f"{file_path}: Unable to read file — {e}")
            elif not path.exists():
                errors.append(f"{file_path}: File not found")

        return errors
