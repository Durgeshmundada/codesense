"""Tree formatter for displaying project structure with box-drawing characters."""

import os
from fnmatch import fnmatch
from pathlib import Path


class TreeFormatter:
    """Formats directory trees with box-drawing characters and optional annotations."""

    PIPE = "│   "
    TEE = "├── "
    ELBOW = "└── "
    SPACE = "    "

    def format(self, tree: dict, annotations: dict[str, str] | None = None) -> str:
        """Format a nested dict tree structure into a string with box-drawing characters.

        Args:
            tree: Nested dict representing directory structure.
                  Format: {"name": "root", "children": [...]}
                  Each child is also a dict with "name" and optionally "children".
            annotations: Maps file/dir paths to one-line descriptions.
                         Paths should use forward slashes relative to tree root.

        Returns:
            Formatted tree string with box-drawing characters and inline annotations.
        """
        if annotations is None:
            annotations = {}

        lines: list[str] = []
        root_name = tree.get("name", "")
        root_annotation = annotations.get(root_name, "")
        if root_annotation:
            lines.append(f"{root_name}  ← {root_annotation}")
        else:
            lines.append(root_name)

        children = tree.get("children", [])
        self._format_children(children, "", lines, annotations, root_name)

        return "\n".join(lines)

    def _format_children(
        self,
        children: list[dict],
        prefix: str,
        lines: list[str],
        annotations: dict[str, str],
        parent_path: str,
    ) -> None:
        """Recursively format children nodes."""
        for i, child in enumerate(children):
            is_last = i == len(children) - 1
            connector = self.ELBOW if is_last else self.TEE
            child_name = child.get("name", "")

            # Build the path for annotation lookup
            if parent_path:
                child_path = f"{parent_path}/{child_name}"
            else:
                child_path = child_name

            annotation = annotations.get(child_path, "")
            annotation_suffix = f"  ← {annotation}" if annotation else ""

            lines.append(f"{prefix}{connector}{child_name}{annotation_suffix}")

            # Recurse into children
            child_children = child.get("children", [])
            if child_children:
                extension = self.SPACE if is_last else self.PIPE
                self._format_children(
                    child_children, prefix + extension, lines, annotations, child_path
                )

    def build_tree(
        self, root_path: str, depth: int = -1, gitignore: bool = True
    ) -> dict:
        """Walk a directory and build a nested dict structure.

        Args:
            root_path: Path to the root directory to walk.
            depth: Maximum depth to recurse. -1 means unlimited.
            gitignore: If True, respect .gitignore patterns found in the root.

        Returns:
            Nested dict: {"name": "<dir_name>", "children": [...]}
        """
        root = Path(root_path).resolve()
        if not root.is_dir():
            return {"name": root.name, "children": []}

        ignore_patterns = self._load_gitignore(root) if gitignore else []

        return self._walk_directory(root, root, depth, 0, ignore_patterns)

    def _walk_directory(
        self,
        current: Path,
        root: Path,
        max_depth: int,
        current_depth: int,
        ignore_patterns: list[str],
    ) -> dict:
        """Recursively walk a directory building the tree dict."""
        node: dict = {"name": current.name, "children": []}

        if max_depth != -1 and current_depth >= max_depth:
            return node

        try:
            entries = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return node

        for entry in entries:
            # Skip hidden files/dirs starting with .
            if entry.name.startswith("."):
                continue

            # Check gitignore patterns
            rel_path = entry.relative_to(root)
            if self._is_ignored(rel_path, entry.is_dir(), ignore_patterns):
                continue

            if entry.is_dir():
                child = self._walk_directory(
                    entry, root, max_depth, current_depth + 1, ignore_patterns
                )
                node["children"].append(child)
            else:
                node["children"].append({"name": entry.name})

        return node

    def _load_gitignore(self, root: Path) -> list[str]:
        """Load .gitignore patterns from the root directory."""
        gitignore_path = root / ".gitignore"
        if not gitignore_path.is_file():
            return []

        patterns: list[str] = []
        try:
            with open(gitignore_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith("#"):
                        continue
                    patterns.append(line)
        except (OSError, UnicodeDecodeError):
            pass

        return patterns

    def render_rich(self, tree: dict, annotations: dict[str, str] | None = None) -> str:
        """Render tree using Rich Tree for terminal display.

        Falls back to plain format() if Rich is not available.

        Args:
            tree: Tree dict with structure like format().
            annotations: Optional annotations dict.

        Returns:
            Rich-rendered tree string for terminal output.
        """
        try:
            from rich.tree import Tree as RichTree
            from rich.console import Console
            from io import StringIO

            if annotations is None:
                annotations = {}

            root_name = tree.get("name", "")
            root_annotation = annotations.get(root_name, "")
            root_label = f"[bold]{root_name}[/bold]"
            if root_annotation:
                root_label += f"  [dim]← {root_annotation}[/dim]"

            rich_tree = RichTree(root_label)
            self._build_rich_tree(
                rich_tree, tree.get("children", []), annotations, root_name
            )

            buffer = StringIO()
            console = Console(file=buffer, force_terminal=True, width=120)
            console.print(rich_tree)
            return buffer.getvalue().rstrip()
        except ImportError:
            return self.format(tree, annotations)

    def _build_rich_tree(
        self,
        parent_node,
        children: list[dict],
        annotations: dict[str, str],
        parent_path: str,
    ) -> None:
        """Recursively build Rich Tree nodes."""
        for child in children:
            child_name = child.get("name", "")
            child_path = f"{parent_path}/{child_name}" if parent_path else child_name
            is_dir = bool(child.get("children"))

            annotation = annotations.get(child_path, "")
            if is_dir:
                label = f"[bold blue]{child_name}/[/bold blue]"
            else:
                label = child_name
            if annotation:
                label += f"  [dim]← {annotation}[/dim]"

            child_node = parent_node.add(label)
            child_children = child.get("children", [])
            if child_children:
                self._build_rich_tree(child_node, child_children, annotations, child_path)

    def _is_ignored(
        self, rel_path: Path, is_dir: bool, patterns: list[str]
    ) -> bool:
        """Check if a path matches any gitignore pattern."""
        path_str = rel_path.as_posix()
        name = rel_path.name

        for pattern in patterns:
            # Handle negation patterns (not fully supported, just skip)
            if pattern.startswith("!"):
                continue

            # Strip trailing slash (directory-only pattern)
            dir_only = pattern.endswith("/")
            clean_pattern = pattern.rstrip("/")

            if dir_only and not is_dir:
                continue

            # Match against name only (simple pattern without /)
            if "/" not in clean_pattern:
                if fnmatch(name, clean_pattern):
                    return True
                # For directories, also try matching with trailing content
                if is_dir and fnmatch(name, clean_pattern):
                    return True
            else:
                # Match against full relative path
                if fnmatch(path_str, clean_pattern):
                    return True
                # Also try matching with leading wildcard
                if fnmatch(path_str, f"**/{clean_pattern}"):
                    return True

        return False
