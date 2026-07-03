"""Onboard capability handler — generates a complete onboarding markdown document.

Produces an onboarding guide covering:
- Project purpose (from README if available)
- Structure (from tree analysis)
- Execution flow (from flow capability)
- Design decisions (from Decision Memory / ADR docs if ingested)
- Setup instructions (from README, requirements.txt, pyproject.toml)
- Safe-to-change areas (from risk assessment — areas with low risk score)

Accepts optional module path (scope to a specific module) and output flag
(write to file).

Requirements: 5.10
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from codesense.analysis.call_graph import CallGraphBuilder
from codesense.models.output import CommandOutput, CommandParams
from codesense.output.markdown_writer import MarkdownWriter
from codesense.output.tree_formatter import TreeFormatter

logger = logging.getLogger(__name__)

ONBOARD_TITLE = "\U0001f4da Onboarding Guide"


class OnboardHandler:
    """Capability handler for the 'onboard' CLI command.

    Generates a complete onboarding markdown document for a project or
    a specific module within it.

    Implements the CapabilityHandler protocol: run(params) -> CommandOutput.

    Args:
        project_root: Root directory of the project to analyze.
            Defaults to the current working directory.
    """

    def __init__(self, project_root: Optional[str] = None) -> None:
        self._project_root = project_root or os.getcwd()
        self._markdown_writer = MarkdownWriter()
        self._tree_formatter = TreeFormatter()

    def run(self, params: CommandParams) -> CommandOutput:
        """Execute the onboard capability with the given parameters.

        Generates an onboarding guide by gathering project information from
        various sources (README, tree structure, call graph, setup files).
        Optionally scoped to a specific module path and optionally writes
        to a file.

        Args:
            params: Parsed CLI arguments. Uses:
                - params.path: Optional module path to scope the guide.
                - params.output: Optional file path to write the document.
                - params.mock: Whether demo mode is active.

        Returns:
            CommandOutput with:
                - title: "Onboarding Guide"
                - content: the generated onboarding markdown document
                - is_demo_mode: from params.mock
        """
        module_path = params.path
        output_path = params.output
        is_demo = params.mock

        # Determine the scope: specific module or entire project
        scope_root = self._resolve_scope(module_path)
        project_name = self._detect_project_name(scope_root)

        # Gather all onboarding information
        project_info = {
            "name": project_name,
            "purpose": self._gather_purpose(scope_root, module_path),
            "structure": self._gather_structure(scope_root),
            "execution_flow": self._gather_execution_flow(scope_root, module_path),
            "design_decisions": self._gather_design_decisions(),
            "setup": self._gather_setup_instructions(scope_root),
            "safe_areas": self._gather_safe_areas(scope_root),
        }

        # Render the onboarding document using MarkdownWriter
        document = self._markdown_writer.render_onboarding(project_info)

        # Write to file if --output specified
        if output_path:
            try:
                self._markdown_writer.write_file(document, output_path)
                logger.info("Onboarding document written to: %s", output_path)
            except OSError as e:
                logger.error("Failed to write onboarding document: %s", e)
                return CommandOutput(
                    title=ONBOARD_TITLE,
                    content=(
                        f"Error: Unable to write onboarding document to "
                        f"'{output_path}'.\n"
                        f"Reason: {e}\n\n"
                        "The generated document is shown below:\n\n"
                        f"{document}"
                    ),
                    is_demo_mode=is_demo,
                )

        # Build content with optional write confirmation
        content = document
        if output_path:
            content = (
                f"Written to: `{output_path}`\n\n{document}"
            )

        return CommandOutput(
            title=ONBOARD_TITLE,
            content=content,
            is_demo_mode=is_demo,
        )

    def _resolve_scope(self, module_path: Optional[str]) -> str:
        """Resolve the root directory for analysis.

        If a module path is provided, resolves it relative to project root.
        Otherwise returns the project root.

        Args:
            module_path: Optional path to a specific module/directory.

        Returns:
            Absolute path to the analysis root directory.
        """
        if module_path:
            path = Path(module_path)
            if not path.is_absolute():
                path = Path(self._project_root) / path
            # If it's a file, use its parent directory
            if path.is_file():
                return str(path.parent)
            return str(path)
        return self._project_root

    def _detect_project_name(self, scope_root: str) -> str:
        """Detect the project name from pyproject.toml or directory name.

        Tries to read the project name from pyproject.toml first,
        then falls back to the directory name.

        Args:
            scope_root: Root directory of the scope being analyzed.

        Returns:
            Detected project name string.
        """
        # Try pyproject.toml in project root
        pyproject_path = Path(self._project_root) / "pyproject.toml"
        if pyproject_path.is_file():
            try:
                content = pyproject_path.read_text(encoding="utf-8")
                for line in content.split("\n"):
                    line_stripped = line.strip()
                    if line_stripped.startswith("name") and "=" in line_stripped:
                        value = (
                            line_stripped.split("=", 1)[1]
                            .strip()
                            .strip('"')
                            .strip("'")
                        )
                        if value:
                            return value
            except (OSError, UnicodeDecodeError):
                pass

        # Fall back to directory name
        return Path(scope_root).name

    def _gather_purpose(
        self, scope_root: str, module_path: Optional[str]
    ) -> str:
        """Gather project purpose from README files.

        Reads the first paragraph or introduction section from README.md
        or README.rst at the project root.

        If scoped to a module, also checks for a module-level docstring
        in __init__.py.

        Args:
            scope_root: Root directory being analyzed.
            module_path: Optional module path for scoped analysis.

        Returns:
            Purpose description string.
        """
        purpose_parts: list[str] = []

        # Try README at project root
        readme_content = self._read_readme(self._project_root)
        if readme_content:
            purpose_parts.append(readme_content)

        # If scoped to a module, try module docstring
        if module_path:
            module_doc = self._read_module_docstring(scope_root)
            if module_doc:
                purpose_parts.append(f"\n**Module focus:** {module_doc}")

        if purpose_parts:
            return "\n".join(purpose_parts)
        return (
            "No project description found. "
            "Add a README.md to provide project context."
        )

    def _read_readme(self, directory: str) -> Optional[str]:
        """Read and extract the introduction from a README file.

        Looks for README.md or README.rst in the given directory and
        extracts the first meaningful section (up to the second heading
        or first 500 characters).

        Args:
            directory: Directory to search for README.

        Returns:
            Extracted introduction text, or None if no README found.
        """
        root = Path(directory)
        readme_names = ["README.md", "README.rst", "readme.md", "Readme.md"]

        for name in readme_names:
            readme_path = root / name
            if readme_path.is_file():
                try:
                    content = readme_path.read_text(encoding="utf-8")
                    return self._extract_intro(content)
                except (OSError, UnicodeDecodeError):
                    continue

        return None

    def _extract_intro(self, readme_content: str) -> str:
        """Extract the introduction section from README content.

        Takes content up to the second heading or up to 500 characters,
        whichever comes first.

        Args:
            readme_content: Full README file content.

        Returns:
            Introduction text extracted from the README.
        """
        lines = readme_content.split("\n")
        intro_lines: list[str] = []
        heading_count = 0

        for line in lines:
            # Count markdown headings
            if line.startswith("#"):
                heading_count += 1
                if heading_count > 1:
                    break
                # Skip the first heading (usually the project name)
                continue
            intro_lines.append(line)

        intro = "\n".join(intro_lines).strip()

        # Cap at reasonable length
        if len(intro) > 500:
            truncated = intro[:500]
            last_period = truncated.rfind(".")
            if last_period > 200:
                intro = truncated[: last_period + 1]
            else:
                intro = truncated + "..."

        return intro if intro else None

    def _read_module_docstring(self, scope_root: str) -> Optional[str]:
        """Read the module-level docstring from __init__.py.

        Args:
            scope_root: Directory containing the module's __init__.py.

        Returns:
            First line of the module docstring, or None.
        """
        init_path = Path(scope_root) / "__init__.py"
        if not init_path.is_file():
            return None

        try:
            content = init_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return None

        if not content:
            return None

        for quote in ('"""', "'''"):
            if content.startswith(quote):
                end_idx = content.find(quote, len(quote))
                if end_idx != -1:
                    docstring = content[len(quote):end_idx].strip()
                    return docstring.split("\n")[0].strip() or None
                else:
                    # Multi-line: take first non-empty line
                    lines = content[len(quote):].split("\n")
                    for ln in lines:
                        stripped = ln.strip()
                        if stripped:
                            return stripped
        return None

    def _gather_structure(self, scope_root: str) -> str:
        """Gather project structure using TreeFormatter.

        Builds a tree representation of the directory structure.

        Args:
            scope_root: Root directory to analyze.

        Returns:
            Formatted tree string showing project structure.
        """
        try:
            tree = self._tree_formatter.build_tree(scope_root, depth=3)
            content = self._tree_formatter.format(tree)
            return f"```\n{content}\n```"
        except Exception as e:
            logger.warning("Failed to build project tree: %s", e)
            return "Unable to generate project structure."

    def _gather_execution_flow(
        self, scope_root: str, module_path: Optional[str]
    ) -> str:
        """Gather execution flow information using CallGraphBuilder.

        Attempts to trace the main entry point of the project or module.

        Args:
            scope_root: Root directory being analyzed.
            module_path: Optional module path for scoped analysis.

        Returns:
            Execution flow description string.
        """
        entry_points = self._find_entry_points(scope_root)

        if not entry_points:
            return (
                "No clear entry point detected. "
                "Check `main.py`, `__main__.py`, or `app.py` for execution flow."
            )

        flow_parts: list[str] = []
        builder = CallGraphBuilder(project_root=self._project_root)

        for entry_point in entry_points[:3]:
            try:
                call_graph = builder.build(
                    entry_point=entry_point, max_depth=5
                )
                if call_graph.edges:
                    flow_parts.append(f"**Entry: `{entry_point}`**")
                    for i, (caller, callee) in enumerate(
                        call_graph.edges[:10], 1
                    ):
                        flow_parts.append(f"  {i}. `{caller}` -> `{callee}`")
                    if call_graph.max_depth_reached:
                        flow_parts.append(
                            "  _(truncated - deeper calls exist)_"
                        )
                    flow_parts.append("")
            except Exception as e:
                logger.debug(
                    "Could not trace entry point '%s': %s", entry_point, e
                )
                continue

        if flow_parts:
            return "\n".join(flow_parts)
        return "No execution flow could be traced from detected entry points."

    def _find_entry_points(self, scope_root: str) -> list[str]:
        """Find likely entry points in the project or module.

        Searches for common patterns: main.py, __main__.py, app.py, cli.py.

        Args:
            scope_root: Root directory to search.

        Returns:
            List of entry point file paths (relative to project root).
        """
        entry_point_names = [
            "main.py",
            "__main__.py",
            "app.py",
            "cli.py",
            "server.py",
        ]

        root = Path(scope_root)
        found: list[str] = []

        for name in entry_point_names:
            candidate = root / name
            if candidate.is_file():
                try:
                    rel = candidate.relative_to(Path(self._project_root))
                    found.append(str(rel).replace("\\", "/"))
                except ValueError:
                    found.append(str(candidate))

        return found

    def _gather_design_decisions(self) -> list[str]:
        """Gather design decisions from ADR documents.

        Looks for ADR documents in common locations:
        - docs/adr/
        - docs/decisions/
        - adr/
        - .kiro/specs/ design.md files

        Returns:
            List of design decision summary strings.
        """
        decisions: list[str] = []
        adr_dirs = [
            "docs/adr",
            "docs/decisions",
            "adr",
            "decisions",
        ]

        root = Path(self._project_root)

        for adr_dir in adr_dirs:
            adr_path = root / adr_dir
            if adr_path.is_dir():
                try:
                    for f in sorted(adr_path.iterdir()):
                        if f.suffix in (".md", ".rst", ".txt") and f.is_file():
                            decision = self._extract_decision_title(f)
                            if decision:
                                decisions.append(decision)
                except (OSError, PermissionError):
                    continue

        # Check for design.md in .kiro/specs
        specs_dir = root / ".kiro" / "specs"
        if specs_dir.is_dir():
            try:
                for spec_folder in specs_dir.iterdir():
                    if spec_folder.is_dir():
                        design_file = spec_folder / "design.md"
                        if design_file.is_file():
                            extracted = (
                                self._extract_design_decisions_from_spec(
                                    design_file
                                )
                            )
                            decisions.extend(extracted)
            except (OSError, PermissionError):
                pass

        return decisions

    def _extract_decision_title(self, file_path: Path) -> Optional[str]:
        """Extract the title/summary of an ADR document.

        Reads the first heading from a markdown ADR file.

        Args:
            file_path: Path to the ADR document.

        Returns:
            Decision title string, or None if not extractable.
        """
        try:
            content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

        for line in content.split("\n"):
            line_stripped = line.strip()
            if line_stripped.startswith("#"):
                title = line_stripped.lstrip("#").strip()
                if title:
                    return f"{file_path.stem}: {title}"
        return None

    def _extract_design_decisions_from_spec(
        self, design_file: Path
    ) -> list[str]:
        """Extract design decisions from a spec design.md file.

        Looks for a "Design Decisions" section and extracts bullet points.

        Args:
            design_file: Path to the design.md file.

        Returns:
            List of design decision strings.
        """
        decisions: list[str] = []
        try:
            content = design_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return decisions

        in_decisions_section = False
        for line in content.split("\n"):
            if "Design Decisions" in line and line.strip().startswith("#"):
                in_decisions_section = True
                continue
            if in_decisions_section:
                if line.strip().startswith("#"):
                    break
                if line.strip().startswith(("-", "*")):
                    text = line.strip().lstrip("-*").strip()
                    text = text.replace("**", "")
                    if text:
                        decisions.append(text)

        return decisions

    def _gather_setup_instructions(self, scope_root: str) -> str:
        """Gather setup instructions from project configuration files.

        Reads from requirements.txt, pyproject.toml, and .env templates.

        Args:
            scope_root: Root directory being analyzed.

        Returns:
            Setup instructions string in markdown format.
        """
        instructions: list[str] = []
        root = Path(self._project_root)

        # Check for requirements.txt
        req_path = root / "requirements.txt"
        if req_path.is_file():
            instructions.append("**Install dependencies:**")
            instructions.append("```bash")
            instructions.append("pip install -r requirements.txt")
            instructions.append("```")

        # Check for pyproject.toml
        pyproject_path = root / "pyproject.toml"
        if pyproject_path.is_file():
            instructions.append("**Install with pip (editable):**")
            instructions.append("```bash")
            instructions.append("pip install -e .")
            instructions.append("```")

        # Check for .env.template or .env.example
        env_template = None
        for name in [".env.template", ".env.example", ".env.sample"]:
            env_path = root / name
            if env_path.is_file():
                env_template = env_path
                break

        if env_template:
            instructions.append("\n**Environment setup:**")
            instructions.append("```bash")
            instructions.append(f"cp {env_template.name} .env")
            instructions.append("# Edit .env with your configuration")
            instructions.append("```")
            env_vars = self._extract_env_vars(env_template)
            if env_vars:
                instructions.append("\nRequired environment variables:")
                for var in env_vars:
                    instructions.append(f"- `{var}`")

        # Check for Makefile
        makefile_path = root / "Makefile"
        if makefile_path.is_file():
            instructions.append("\n**Available make targets:**")
            targets = self._extract_make_targets(makefile_path)
            for target in targets[:10]:
                instructions.append(f"- `make {target}`")

        if instructions:
            return "\n".join(instructions)
        return (
            "No setup configuration files found. "
            "Check with the project maintainer for setup instructions."
        )

    def _extract_env_vars(self, env_path: Path) -> list[str]:
        """Extract environment variable names from a .env template file.

        Args:
            env_path: Path to the .env template file.

        Returns:
            List of environment variable names.
        """
        env_vars: list[str] = []
        try:
            content = env_path.read_text(encoding="utf-8")
            for line in content.split("\n"):
                line_stripped = line.strip()
                if (
                    line_stripped
                    and not line_stripped.startswith("#")
                    and "=" in line_stripped
                ):
                    var_name = line_stripped.split("=", 1)[0].strip()
                    if var_name:
                        env_vars.append(var_name)
        except (OSError, UnicodeDecodeError):
            pass
        return env_vars

    def _extract_make_targets(self, makefile_path: Path) -> list[str]:
        """Extract target names from a Makefile.

        Args:
            makefile_path: Path to the Makefile.

        Returns:
            List of make target names.
        """
        targets: list[str] = []
        try:
            content = makefile_path.read_text(encoding="utf-8")
            for line in content.split("\n"):
                if (
                    ":" in line
                    and not line.startswith("\t")
                    and not line.startswith(" ")
                ):
                    target = line.split(":", 1)[0].strip()
                    if (
                        target
                        and not target.startswith(".")
                        and not target.startswith("#")
                    ):
                        targets.append(target)
        except (OSError, UnicodeDecodeError):
            pass
        return targets

    def _gather_safe_areas(self, scope_root: str) -> list[str]:
        """Identify safe-to-change areas based on risk heuristics.

        Uses simple heuristics to identify areas that are likely safe:
        - Test directories
        - Documentation directories
        - Configuration files
        - Utility/helper modules

        Args:
            scope_root: Root directory being analyzed.

        Returns:
            List of safe-to-change area descriptions.
        """
        safe_areas: list[str] = []
        root = Path(scope_root)

        # Tests are always safe to modify
        test_dirs = ["tests", "test", "spec", "specs"]
        for test_dir in test_dirs:
            if (root / test_dir).is_dir():
                safe_areas.append(
                    f"`{test_dir}/` — Test files can be freely modified"
                )

        # Documentation directories
        doc_dirs = ["docs", "doc", "documentation"]
        for doc_dir in doc_dirs:
            if (root / doc_dir).is_dir():
                safe_areas.append(
                    f"`{doc_dir}/` — Documentation, no runtime impact"
                )

        # Configuration files that are typically safe
        safe_configs = [
            (".gitignore", "Git ignore patterns"),
            ("pyproject.toml", "Build configuration (non-breaking changes)"),
            (".env.template", "Environment variable template"),
        ]
        for filename, description in safe_configs:
            if (root / filename).is_file():
                safe_areas.append(f"`{filename}` — {description}")

        # Look for utility/helper modules (low coupling)
        utils_patterns = ["utils", "helpers", "constants", "config"]
        try:
            for item in root.iterdir():
                if item.is_dir() and item.name in utils_patterns:
                    safe_areas.append(
                        f"`{item.name}/` — Utility module, low coupling"
                    )
                elif (
                    item.is_file()
                    and item.stem in utils_patterns
                    and item.suffix == ".py"
                ):
                    safe_areas.append(
                        f"`{item.name}` — Utility file, low coupling"
                    )
        except (OSError, PermissionError):
            pass

        if not safe_areas:
            safe_areas.append(
                "No clearly safe areas identified — "
                "review coupling before modifying"
            )

        return safe_areas


def run_onboard(
    module_path: Optional[str] = None,
    output_path: Optional[str] = None,
    project_root: Optional[str] = None,
) -> CommandOutput:
    """Convenience function to run the onboard capability.

    Creates an OnboardHandler and runs it with the provided parameters.

    Args:
        module_path: Optional module path to scope the onboarding guide.
        output_path: Optional file path to write the generated document.
        project_root: Root directory of the project (defaults to cwd).

    Returns:
        CommandOutput with the generated onboarding document.
    """
    handler = OnboardHandler(project_root=project_root or os.getcwd())
    params = CommandParams(
        path=module_path,
        output=output_path,
        mock=False,
    )
    return handler.run(params)
