"""Deps capability handler — displays dependency analysis for a module.

Uses ImportScanner to analyze the target module and displays external packages,
environment variables, external APIs, and internal module dependencies formatted
as Rich tables. Optionally generates a Mermaid dependency graph.

Requirements: 5.7, 12.4
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from codesense.analysis.import_scanner import ImportScanner
from codesense.models.output import CodeSnippet, CommandOutput, CommandParams, TableData
from codesense.output.mermaid_formatter import MermaidFormatter

logger = logging.getLogger(__name__)

DEPS_TITLE = "Dependency Analysis"


class DepsHandler:
    """Capability handler for the 'deps' command.

    Analyzes a module's dependencies using ImportScanner and presents the
    results as categorized Rich tables (external packages, environment variables,
    external APIs, internal module dependencies) plus an optional Mermaid
    dependency graph.

    Implements the CapabilityHandler protocol: run(params) -> CommandOutput.

    Args:
        project_root: Root directory of the project to analyze.
            Defaults to the current working directory.
    """

    def __init__(self, project_root: Optional[str] = None) -> None:
        self._project_root = project_root or str(Path.cwd())

    def run(self, params: CommandParams) -> CommandOutput:
        """Execute the deps capability with the given parameters.

        Uses ImportScanner to analyze the module at params.path (or cwd if None),
        then formats the results into categorized tables and a Mermaid dependency
        graph.

        Args:
            params: Parsed CLI arguments. Uses:
                - params.path: Module path to analyze (file or directory).
                    Defaults to the project root if not provided.
                - params.mock: Whether demo mode is active.

        Returns:
            CommandOutput with:
                - title: "Dependency Analysis"
                - content: formatted dependency info in markdown
                - tables: one TableData per category (external packages, env vars,
                    external APIs, internal deps)
                - code_snippets: Mermaid dependency graph
                - is_demo_mode: from params.mock
        """
        target_path = params.path or self._project_root
        is_demo = params.mock

        # Resolve the target path relative to project root if not absolute
        resolved_path = Path(target_path)
        if not resolved_path.is_absolute():
            resolved_path = Path(self._project_root) / resolved_path

        # Check that the path exists
        if not resolved_path.exists():
            return CommandOutput(
                title=DEPS_TITLE,
                content=(
                    f"Error: Path '{target_path}' does not exist.\n"
                    f"Please provide a valid module path to analyze."
                ),
                is_demo_mode=is_demo,
            )

        # Perform import scanning
        try:
            scanner = ImportScanner(project_root=self._project_root)
            import_graph = scanner.scan(str(resolved_path))
        except Exception as e:
            logger.error("Failed to scan dependencies for '%s': %s", target_path, e)
            return CommandOutput(
                title=DEPS_TITLE,
                content=(
                    f"Error: Unable to analyze dependencies for '{target_path}'.\n"
                    f"Reason: {e}"
                ),
                is_demo_mode=is_demo,
            )

        # Build tables for each category
        tables: list[TableData] = []

        # External packages table
        if import_graph.external_deps:
            tables.append(
                TableData(
                    headers=["Package"],
                    rows=[[dep] for dep in import_graph.external_deps],
                    title="External Packages",
                )
            )

        # Environment variables table
        if import_graph.env_vars:
            tables.append(
                TableData(
                    headers=["Variable"],
                    rows=[[var] for var in import_graph.env_vars],
                    title="Environment Variables",
                )
            )

        # External APIs table
        if import_graph.external_apis:
            tables.append(
                TableData(
                    headers=["API Call"],
                    rows=[[api] for api in import_graph.external_apis],
                    title="External APIs",
                )
            )

        # Internal module dependencies table
        if import_graph.internal_deps:
            tables.append(
                TableData(
                    headers=["Module"],
                    rows=[[dep] for dep in import_graph.internal_deps],
                    title="Internal Dependencies",
                )
            )

        # Build markdown content summary
        content = self._build_content(import_graph, target_path)

        # Generate Mermaid dependency graph
        formatter = MermaidFormatter()
        mermaid_diagram = formatter.dependency_graph(import_graph)

        content += "\n\n### Dependency Graph\n\n"
        content += f"```mermaid\n{mermaid_diagram}\n```"

        return CommandOutput(
            title=DEPS_TITLE,
            content=content,
            tables=tables,
            code_snippets=[
                CodeSnippet(
                    code=mermaid_diagram,
                    language="mermaid",
                    label="Dependency Graph",
                )
            ],
            is_demo_mode=is_demo,
        )

    def _build_content(self, import_graph, target_path: str) -> str:
        """Build markdown content summarizing the dependency analysis.

        Args:
            import_graph: The ImportGraph result from scanning.
            target_path: The original target path string for display.

        Returns:
            Markdown-formatted summary of dependencies.
        """
        lines: list[str] = []
        lines.append(f"**Module:** `{target_path}`")
        lines.append("")

        # Summary counts
        lines.append("### Summary")
        lines.append("")
        lines.append(f"- External packages: {len(import_graph.external_deps)}")
        lines.append(f"- Environment variables: {len(import_graph.env_vars)}")
        lines.append(f"- External APIs: {len(import_graph.external_apis)}")
        lines.append(f"- Internal dependencies: {len(import_graph.internal_deps)}")

        # Detailed sections
        if import_graph.external_deps:
            lines.append("")
            lines.append("### External Packages")
            lines.append("")
            for dep in import_graph.external_deps:
                lines.append(f"- `{dep}`")

        if import_graph.env_vars:
            lines.append("")
            lines.append("### Environment Variables")
            lines.append("")
            for var in import_graph.env_vars:
                lines.append(f"- `{var}`")

        if import_graph.external_apis:
            lines.append("")
            lines.append("### External APIs")
            lines.append("")
            for api in import_graph.external_apis:
                lines.append(f"- `{api}`")

        if import_graph.internal_deps:
            lines.append("")
            lines.append("### Internal Dependencies")
            lines.append("")
            for dep in import_graph.internal_deps:
                lines.append(f"- `{dep}`")

        return "\n".join(lines)
