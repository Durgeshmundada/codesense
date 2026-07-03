"""Related capability handler — displays file impact analysis.

Shows dependents (files that import the target), dependencies (files the target
imports from), and files frequently co-modified with the target. Provides impact
analysis indicating how many files would be affected by a change.

Uses ImportScanner for static import analysis and MCP get_related_changes for
co-modification history.

Requirements: 5.8
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from codesense.analysis.import_scanner import ImportScanner
from codesense.models.output import CommandOutput, CommandParams, TableData

logger = logging.getLogger(__name__)

RELATED_TITLE = "🔗 Impact Analysis"


class RelatedHandler:
    """Capability handler for the 'related' command.

    Implements the CapabilityHandler protocol. Analyzes a file path to find:
    - Dependents: files that import the target file
    - Dependencies: files the target imports from
    - Co-modified: files frequently changed together (from git history)

    Produces an impact summary with tables for each direction.

    Args:
        project_root: Root directory of the project to analyze.
            Defaults to the current working directory.
        mock: Whether to use mock data sources for MCP tools.
    """

    def __init__(
        self, project_root: Optional[str] = None, mock: bool = False
    ) -> None:
        self._project_root = project_root or str(Path.cwd())
        self._mock = mock

    def run(self, params: CommandParams) -> CommandOutput:
        """Execute the related capability with the given parameters.

        Scans for dependents and dependencies using ImportScanner, retrieves
        co-modified files via MCP tools, and formats the results with impact
        analysis.

        Args:
            params: Parsed CLI arguments. Uses:
                - params.path: File path to analyze relationships for.
                - params.mock: Whether demo mode is active.

        Returns:
            CommandOutput with:
                - title: "🔗 Impact Analysis"
                - content: markdown summary with impact counts
                - tables: tables for dependents, dependencies, co-modified files
                - is_demo_mode: from params.mock
        """
        file_path = params.path or ""
        is_demo = params.mock or self._mock

        if not file_path:
            return CommandOutput(
                title=RELATED_TITLE,
                content="**Error:** No file path provided. Usage: `related <file_path>`",
                is_demo_mode=is_demo,
            )

        # Resolve path for display
        resolved = Path(file_path)
        display_path = str(resolved)

        # 1. Find dependencies (files this module imports from)
        dependencies = self._find_dependencies(file_path)

        # 2. Find dependents (files that import this module)
        dependents = self._find_dependents(file_path)

        # 3. Find co-modified files via MCP tools
        co_modified = self._find_co_modified(file_path, is_demo)

        # Build content with impact analysis
        content = self._build_content(
            display_path, dependents, dependencies, co_modified
        )

        # Build tables
        tables = self._build_tables(dependents, dependencies, co_modified)

        return CommandOutput(
            title=RELATED_TITLE,
            content=content,
            tables=tables,
            is_demo_mode=is_demo,
        )

    def _find_dependencies(self, file_path: str) -> list[str]:
        """Find files that the target imports from (its dependencies).

        Uses ImportScanner.scan_file to get the internal deps of the target.

        Args:
            file_path: Path to the file to analyze.

        Returns:
            List of internal module dependency names.
        """
        try:
            scanner = ImportScanner(project_root=self._project_root)
            import_graph = scanner.scan_file(file_path)
            return import_graph.internal_deps
        except Exception as e:
            logger.error("Failed to scan dependencies for '%s': %s", file_path, e)
            return []

    def _find_dependents(self, file_path: str) -> list[str]:
        """Find files that import the target file (its dependents).

        Scans all Python files in the project and checks if any of their
        internal imports reference the target module.

        Args:
            file_path: Path to the file to find dependents for.

        Returns:
            List of file paths that import the target.
        """
        try:
            scanner = ImportScanner(project_root=self._project_root)
            target_path = Path(file_path).resolve()
            project_root = Path(self._project_root).resolve()

            # Derive the module name for the target file
            target_module_name = self._path_to_module_name(target_path, project_root)
            if not target_module_name:
                return []

            # Scan all Python files in the project
            dependents: list[str] = []
            for py_file in project_root.rglob("*.py"):
                # Skip the target file itself
                if py_file.resolve() == target_path:
                    continue

                # Skip __pycache__ and hidden directories
                parts = py_file.parts
                if any(
                    part.startswith(".") or part == "__pycache__" for part in parts
                ):
                    continue

                try:
                    import_graph = scanner.scan_file(str(py_file))
                    # Check if any internal dep matches the target module
                    for dep in import_graph.internal_deps:
                        if self._module_matches_target(dep, target_module_name):
                            # Use relative path for display
                            try:
                                rel_path = str(py_file.relative_to(project_root))
                            except ValueError:
                                rel_path = str(py_file)
                            dependents.append(rel_path)
                            break
                except Exception:
                    # Skip files that can't be parsed
                    continue

            return sorted(dependents)
        except Exception as e:
            logger.error("Failed to find dependents for '%s': %s", file_path, e)
            return []

    def _path_to_module_name(self, file_path: Path, project_root: Path) -> str:
        """Convert a file path to its Python module name.

        Args:
            file_path: Absolute path to a Python file.
            project_root: Absolute path to the project root.

        Returns:
            Dotted module name (e.g., 'codesense.capabilities.related'),
            or empty string if conversion fails.
        """
        try:
            relative = file_path.relative_to(project_root)
        except ValueError:
            return ""

        # Remove .py extension and convert path separators to dots
        parts = list(relative.parts)
        if not parts:
            return ""

        # Remove .py suffix from the last part
        if parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]

        # Remove __init__ — the module name is the package
        if parts[-1] == "__init__":
            parts = parts[:-1]

        if not parts:
            return ""

        return ".".join(parts)

    def _module_matches_target(self, dep: str, target_module: str) -> bool:
        """Check if an import dependency references the target module.

        Handles both exact matches and prefix matches (e.g., importing
        a submodule of the target package).

        Args:
            dep: The dependency module name found in an import.
            target_module: The target file's module name.

        Returns:
            True if the dep references the target module.
        """
        # Exact match
        if dep == target_module:
            return True
        # The dep is a parent of the target (e.g., dep="codesense.analysis"
        # matches target="codesense.analysis.import_scanner")
        if target_module.startswith(dep + "."):
            return True
        # The dep is a child of the target (unlikely for "depends on" but check)
        if dep.startswith(target_module + "."):
            return True
        return False

    def _find_co_modified(self, file_path: str, mock: bool) -> list[dict]:
        """Find files frequently co-modified with the target via MCP tools.

        Calls get_related_changes from the MCP server.

        Args:
            file_path: Path to the file to find co-modifications for.
            mock: Whether to use mock data sources.

        Returns:
            List of dicts with 'path', 'co_commit_count', 'last_co_modified'.
        """
        try:
            from codesense.mcp_server.server import get_related_changes

            result = get_related_changes(code_path=file_path, mock=mock)

            if "error" in result:
                logger.warning("get_related_changes error: %s", result["error"])
                return []

            return result.get("related_files", [])
        except Exception as e:
            logger.error("Failed to get co-modified files for '%s': %s", file_path, e)
            return []

    def _build_content(
        self,
        display_path: str,
        dependents: list[str],
        dependencies: list[str],
        co_modified: list[dict],
    ) -> str:
        """Build the markdown content with impact analysis.

        Args:
            display_path: The file path being analyzed (for display).
            dependents: List of files that import the target.
            dependencies: List of modules the target imports.
            co_modified: List of co-modified file records.

        Returns:
            Formatted markdown content string.
        """
        parts: list[str] = []
        parts.append(f"**File:** `{display_path}`\n")

        # Impact summary
        total_affected = len(dependents) + len(co_modified)
        parts.append("### ⚠️ Impact Summary\n")
        parts.append(
            f"If you change this file, **{total_affected}** files may be affected:"
        )
        parts.append(f"  - **{len(dependents)}** files directly import this module")
        parts.append(
            f"  - **{len(co_modified)}** files are frequently co-modified in git history"
        )
        parts.append("")

        # Dependents section
        parts.append("### ⬆️ Dependents (files that import this)\n")
        if dependents:
            for dep in dependents:
                parts.append(f"  - `{dep}`")
        else:
            parts.append("  _No files import this module._")
        parts.append("")

        # Dependencies section
        parts.append("### ⬇️ Dependencies (files this imports)\n")
        if dependencies:
            for dep in dependencies:
                parts.append(f"  - `{dep}`")
        else:
            parts.append("  _No internal dependencies detected._")
        parts.append("")

        # Co-modified section
        parts.append("### 🔄 Frequently Co-Modified Files\n")
        if co_modified:
            for item in co_modified:
                path = item.get("path", "unknown")
                count = item.get("co_commit_count", 0)
                parts.append(f"  - `{path}` ({count} co-commits)")
        else:
            parts.append("  _No co-modification data available._")
        parts.append("")

        return "\n".join(parts)

    def _build_tables(
        self,
        dependents: list[str],
        dependencies: list[str],
        co_modified: list[dict],
    ) -> list[TableData]:
        """Build structured tables for the output.

        Args:
            dependents: List of files that import the target.
            dependencies: List of modules the target imports.
            co_modified: List of co-modified file records.

        Returns:
            List of TableData objects for Rich rendering.
        """
        tables: list[TableData] = []

        # Dependents table
        if dependents:
            tables.append(
                TableData(
                    headers=["File", "Relationship"],
                    rows=[[dep, "imports this"] for dep in dependents],
                    title="Dependents",
                )
            )

        # Dependencies table
        if dependencies:
            tables.append(
                TableData(
                    headers=["Module", "Relationship"],
                    rows=[[dep, "imported by this"] for dep in dependencies],
                    title="Dependencies",
                )
            )

        # Co-modified table
        if co_modified:
            rows = []
            for item in co_modified:
                path = item.get("path", "unknown")
                count = str(item.get("co_commit_count", 0))
                last = item.get("last_co_modified", "unknown")
                rows.append([path, count, last])
            tables.append(
                TableData(
                    headers=["File", "Co-Commits", "Last Co-Modified"],
                    rows=rows,
                    title="Co-Modified Files",
                )
            )

        return tables


def run_related(
    file_path: Optional[str] = None,
    project_root: str = ".",
    mock: bool = False,
) -> CommandOutput:
    """Convenience function to run the related capability handler.

    Args:
        file_path: File path to analyze relationships for.
        project_root: Root directory of the project.
        mock: Whether demo mode is active.

    Returns:
        CommandOutput with impact analysis results.
    """
    handler = RelatedHandler(project_root=project_root, mock=mock)
    params = CommandParams(path=file_path, mock=mock)
    return handler.run(params)
