"""Tree capability handler — displays project structure with annotations.

Uses TreeFormatter.build_tree() to walk the directory at params.path (or cwd)
and TreeFormatter.format() to produce the annotated tree string.

Requirements: 5.3
"""

import os
from pathlib import Path
from typing import Optional

from codesense.models.output import CommandOutput, CommandParams
from codesense.output.tree_formatter import TreeFormatter


class TreeHandler:
    """Capability handler for the 'tree' CLI command.

    Displays a project directory structure using box-drawing characters,
    with optional depth limiting and file annotations derived from
    __init__.py docstrings.

    The handler follows the CapabilityHandler protocol:
        run(params: CommandParams) -> CommandOutput

    Args:
        gemini_service: Optional GeminiService for richer annotations.
            Currently unused; annotations come from __init__.py docstrings.
    """

    TITLE = "🌳 Project Structure"

    def __init__(self, gemini_service=None) -> None:
        self._formatter = TreeFormatter()
        self._gemini_service = gemini_service

    def run(self, params: CommandParams) -> CommandOutput:
        """Execute the tree capability with the given parameters.

        Walks the directory at params.path (or cwd if None), applies the
        optional depth limit from params.limit, and produces a formatted
        tree string with simple annotations.

        Args:
            params: Parsed CLI arguments.
                - path: Root directory to display (defaults to cwd).
                - limit: Maximum depth to recurse (None = unlimited).
                - mock: Whether demo mode is active.

        Returns:
            CommandOutput with:
                - title: "Project Structure"
                - content: the formatted tree string
                - is_demo_mode: from params.mock
        """
        # Determine root path: use params.path or fall back to cwd
        root_path = params.path if params.path else os.getcwd()

        # Determine depth: use params.limit or -1 for unlimited
        depth = params.limit if params.limit is not None else -1

        # Build the directory tree structure
        tree = self._formatter.build_tree(root_path, depth=depth)

        # Gather simple annotations from __init__.py docstrings
        annotations = self._gather_annotations(root_path, tree)

        # Format the tree with annotations
        content = self._formatter.format(tree, annotations)

        return CommandOutput(
            title=self.TITLE,
            content=content,
            is_demo_mode=params.mock,
        )

    def _gather_annotations(
        self, root_path: str, tree: dict
    ) -> dict[str, str]:
        """Gather one-line annotations from __init__.py docstrings.

        Walks the tree structure and for each directory that contains an
        __init__.py, extracts the module-level docstring's first line as
        an annotation.

        Args:
            root_path: Absolute path to the root directory.
            tree: The nested dict tree structure from TreeFormatter.build_tree().

        Returns:
            Dict mapping relative tree paths to annotation strings.
        """
        annotations: dict[str, str] = {}
        root = Path(root_path).resolve()

        self._collect_annotations_recursive(root, tree, annotations)

        return annotations

    def _collect_annotations_recursive(
        self,
        root: Path,
        node: dict,
        annotations: dict[str, str],
        parent_path: str = "",
    ) -> None:
        """Recursively collect annotations for directories with __init__.py.

        Args:
            root: The absolute root path of the project.
            node: Current tree node dict.
            annotations: Accumulator dict to fill with path -> annotation.
            parent_path: The tree-relative path to the current node.
        """
        node_name = node.get("name", "")
        if parent_path:
            current_path = f"{parent_path}/{node_name}"
        else:
            current_path = node_name

        children = node.get("children", [])

        # Check if this is a directory (has children) with an __init__.py
        if children:
            # Build the filesystem path for this directory
            if parent_path:
                # Strip the root name from current_path to get relative fs path
                parts = current_path.split("/")
                rel_fs_path = "/".join(parts[1:]) if len(parts) > 1 else ""
            else:
                rel_fs_path = ""

            if rel_fs_path:
                dir_path = root / rel_fs_path
            else:
                dir_path = root

            init_path = dir_path / "__init__.py"
            if init_path.is_file():
                docstring = self._extract_first_docstring_line(init_path)
                if docstring:
                    annotations[current_path] = docstring

            # Recurse into children
            for child in children:
                self._collect_annotations_recursive(
                    root, child, annotations, current_path
                )

    def _extract_first_docstring_line(self, init_path: Path) -> Optional[str]:
        """Extract the first line of the module-level docstring from an __init__.py.

        Args:
            init_path: Path to the __init__.py file.

        Returns:
            First line of the docstring (stripped of quotes), or None if
            no docstring is found.
        """
        try:
            content = init_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

        # Simple extraction: look for the first triple-quoted string
        content = content.strip()
        if not content:
            return None

        for quote in ('"""', "'''"):
            if content.startswith(quote):
                # Find the closing quote
                end_idx = content.find(quote, len(quote))
                if end_idx == -1:
                    # Multi-line: take first line after opening quotes
                    lines = content[len(quote):].split("\n")
                    first_line = lines[0].strip()
                    return first_line if first_line else (lines[1].strip() if len(lines) > 1 else None)
                else:
                    # Single-line docstring
                    docstring = content[len(quote):end_idx].strip()
                    # Return only the first line
                    return docstring.split("\n")[0].strip() or None

        return None
