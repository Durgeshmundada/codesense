"""Call graph builder for tracing static execution paths through Python code."""

import ast
import logging
from pathlib import Path

from codesense.analysis.ast_walker import ASTWalker
from codesense.models.analysis import CallGraph, FunctionInfo, ModuleInfo

logger = logging.getLogger(__name__)


class CallGraphBuilder:
    """Builds a call graph by tracing static function calls from an entry point.

    Starting from a given entry point (file path or qualified function name),
    the builder traces which functions call which other functions by analyzing
    AST-level call expressions and resolving them through import chains.

    Args:
        project_root: Root directory of the project to analyze.
        ast_walker: Optional ASTWalker instance; creates a default if not provided.
    """

    def __init__(self, project_root: str, ast_walker: ASTWalker | None = None) -> None:
        self._project_root = Path(project_root)
        self._ast_walker = ast_walker or ASTWalker()

    def build(self, entry_point: str, max_depth: int = 10) -> CallGraph:
        """Build a call graph starting from the given entry point.

        The entry point can be:
        - A file path (e.g., "src/main.py") — traces all top-level functions in that file
        - A qualified function name (e.g., "src/main.py::run") — traces from that specific function

        Args:
            entry_point: File path or "file_path::function_name" to start tracing from.
            max_depth: Maximum depth of call levels to trace. Defaults to 10.

        Returns:
            CallGraph with root, edges, depth reached, and whether max_depth was hit.
        """
        file_path, function_name = self._parse_entry_point(entry_point)
        resolved_path = self._resolve_path(file_path)

        if resolved_path is None:
            logger.warning("Entry point file not found: %s", file_path)
            return CallGraph(root=entry_point, edges=[], max_depth_reached=False, depth=0)

        # Cache for parsed modules to avoid re-parsing
        self._module_cache: dict[str, ModuleInfo] = {}
        # Track visited (caller, callee) to avoid infinite loops from circular calls
        self._visited_edges: set[tuple[str, str]] = set()
        # Track visited functions at each depth to detect circular call chains
        self._visiting: set[str] = set()

        edges: list[tuple[str, str]] = []
        max_depth_reached = False
        actual_depth = 0

        # Get the module info for the entry file
        module_info = self._get_module_info(str(resolved_path))
        if module_info is None:
            return CallGraph(root=entry_point, edges=[], max_depth_reached=False, depth=0)

        # Determine which functions to trace from
        if function_name:
            start_functions = [
                f for f in module_info.functions if f.name == function_name
            ]
            # Also check class methods
            if not start_functions:
                for cls in module_info.classes:
                    for method in cls.methods:
                        if method.name == function_name:
                            start_functions.append(method)
                            break
            if not start_functions:
                logger.warning(
                    "Function '%s' not found in %s", function_name, file_path
                )
                return CallGraph(
                    root=entry_point, edges=[], max_depth_reached=False, depth=0
                )
        else:
            start_functions = module_info.functions

        # Build the import resolution map for the entry file
        import_map = self._build_import_map(str(resolved_path), module_info)

        # Trace calls from each start function
        for func in start_functions:
            qualified_name = self._qualify_name(str(resolved_path), func.name)
            depth_reached, hit_max = self._trace_calls(
                qualified_name=qualified_name,
                func_info=func,
                file_path=str(resolved_path),
                import_map=import_map,
                edges=edges,
                current_depth=0,
                max_depth=max_depth,
            )
            actual_depth = max(actual_depth, depth_reached)
            if hit_max:
                max_depth_reached = True

        return CallGraph(
            root=entry_point,
            edges=edges,
            max_depth_reached=max_depth_reached,
            depth=actual_depth,
        )

    def _trace_calls(
        self,
        qualified_name: str,
        func_info: FunctionInfo,
        file_path: str,
        import_map: dict[str, str],
        edges: list[tuple[str, str]],
        current_depth: int,
        max_depth: int,
    ) -> tuple[int, bool]:
        """Recursively trace calls from a function.

        Returns:
            Tuple of (max depth reached during tracing, whether max_depth limit was hit).
        """
        if current_depth >= max_depth:
            return current_depth, True

        # Guard against circular call chains
        if qualified_name in self._visiting:
            return current_depth, False
        self._visiting.add(qualified_name)

        deepest = current_depth
        hit_max = False

        try:
            for call_name in func_info.calls:
                # Resolve the callee to a qualified name
                callee_qualified, callee_file = self._resolve_call(
                    call_name, file_path, import_map
                )

                if callee_qualified is None:
                    continue

                edge = (qualified_name, callee_qualified)
                if edge in self._visited_edges:
                    continue
                self._visited_edges.add(edge)
                edges.append(edge)

                # Try to trace deeper into the callee
                callee_func_info = self._find_function_info(
                    callee_qualified, callee_file
                )
                if callee_func_info and callee_file:
                    callee_import_map = self._build_import_map_for_file(callee_file)
                    sub_depth, sub_hit_max = self._trace_calls(
                        qualified_name=callee_qualified,
                        func_info=callee_func_info,
                        file_path=callee_file,
                        import_map=callee_import_map,
                        edges=edges,
                        current_depth=current_depth + 1,
                        max_depth=max_depth,
                    )
                    deepest = max(deepest, sub_depth)
                    if sub_hit_max:
                        hit_max = True
        finally:
            self._visiting.discard(qualified_name)

        return deepest, hit_max

    def _parse_entry_point(self, entry_point: str) -> tuple[str, str | None]:
        """Parse entry point into (file_path, optional_function_name).

        Supports formats:
        - "path/to/file.py" → (path/to/file.py, None)
        - "path/to/file.py::function_name" → (path/to/file.py, function_name)
        """
        if "::" in entry_point:
            parts = entry_point.split("::", 1)
            return parts[0], parts[1]
        return entry_point, None

    def _resolve_path(self, file_path: str) -> Path | None:
        """Resolve a file path relative to the project root.

        Returns None if the file does not exist.
        """
        path = Path(file_path)
        if path.is_absolute():
            if path.exists():
                return path
            return None

        # Try relative to project root
        resolved = self._project_root / path
        if resolved.exists():
            return resolved

        # Try as-is (already absolute or cwd-relative)
        if path.exists():
            return path

        return None

    def _get_module_info(self, path: str) -> ModuleInfo | None:
        """Get ModuleInfo for a file, using cache to avoid re-parsing."""
        if path in self._module_cache:
            return self._module_cache[path]

        try:
            module_info = self._ast_walker.parse_module(path)
            # Check if the parse returned meaningful data (non-empty or valid path)
            self._module_cache[path] = module_info
            return module_info
        except Exception as e:
            logger.error("Failed to parse module %s: %s", path, e)
            return None

    def _build_import_map(
        self, file_path: str, module_info: ModuleInfo
    ) -> dict[str, str]:
        """Build a map from imported names to their resolved file paths.

        Returns a dict mapping call-name prefixes to file paths where they're defined.
        """
        import_map: dict[str, str] = {}
        source_path = Path(file_path)

        tree = self._parse_file_ast(file_path)
        if tree is None:
            return import_map

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                resolved = self._resolve_import_module(
                    node.module or "", source_path, node.level
                )
                if resolved:
                    for alias in node.names:
                        name = alias.asname or alias.name
                        import_map[name] = resolved
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    resolved = self._resolve_import_module(alias.name, source_path, 0)
                    if resolved:
                        name = alias.asname or alias.name
                        import_map[name] = resolved

        return import_map

    def _build_import_map_for_file(self, file_path: str) -> dict[str, str]:
        """Build import map for a given file path."""
        module_info = self._get_module_info(file_path)
        if module_info is None:
            return {}
        return self._build_import_map(file_path, module_info)

    def _resolve_import_module(
        self, module_name: str, source_path: Path, level: int
    ) -> str | None:
        """Resolve an import module name to a file path within the project.

        Args:
            module_name: The dotted module path (e.g., "codesense.analysis.ast_walker").
            source_path: The file that contains the import statement.
            level: Relative import level (0 for absolute, 1+ for relative).

        Returns:
            Resolved file path string, or None if not found in the project.
        """
        if level > 0:
            # Relative import
            base = source_path.parent
            for _ in range(level - 1):
                base = base.parent
            if module_name:
                parts = module_name.split(".")
                candidate = base / "/".join(parts)
            else:
                candidate = base
        else:
            # Absolute import — try to resolve within the project root
            parts = module_name.split(".")
            candidate = self._project_root / "/".join(parts)

        # Check if it's a package (directory with __init__.py)
        if candidate.is_dir() and (candidate / "__init__.py").exists():
            return str(candidate / "__init__.py")

        # Check if it's a module file
        py_file = candidate.with_suffix(".py")
        if py_file.exists():
            return str(py_file)

        return None

    def _resolve_call(
        self, call_name: str, file_path: str, import_map: dict[str, str]
    ) -> tuple[str | None, str | None]:
        """Resolve a call name to a qualified name and its defining file.

        Args:
            call_name: The name as it appears in the call (e.g., "foo", "bar.baz").
            file_path: The file where the call appears.
            import_map: Map from imported names to file paths.

        Returns:
            Tuple of (qualified_name, file_path) or (None, None) if unresolvable.
        """
        # Split dotted call names (e.g., "self.method" or "module.func")
        parts = call_name.split(".")

        # Skip self/cls calls - they refer to methods in the same class
        if parts[0] in ("self", "cls"):
            if len(parts) > 1:
                method_name = parts[1]
                qualified = self._qualify_name(file_path, method_name)
                return qualified, file_path
            return None, None

        # Check if the first part is an imported name
        root_name = parts[0]
        if root_name in import_map:
            target_file = import_map[root_name]
            if len(parts) == 1:
                # Direct imported function call
                qualified = self._qualify_name(target_file, root_name)
                return qualified, target_file
            else:
                # Attribute access on imported module/class
                attr_name = parts[-1]
                qualified = self._qualify_name(target_file, attr_name)
                return qualified, target_file

        # Check if it's a function defined in the same file
        module_info = self._get_module_info(file_path)
        if module_info:
            for func in module_info.functions:
                if func.name == call_name:
                    qualified = self._qualify_name(file_path, call_name)
                    return qualified, file_path
            # Check class methods
            for cls in module_info.classes:
                for method in cls.methods:
                    if method.name == call_name:
                        qualified = self._qualify_name(file_path, call_name)
                        return qualified, file_path

        # Could be a builtin or unresolvable external call — skip
        return None, None

    def _find_function_info(
        self, qualified_name: str, file_path: str | None
    ) -> FunctionInfo | None:
        """Find FunctionInfo for a given qualified function name.

        Args:
            qualified_name: The fully qualified name (file::function).
            file_path: The file where the function is defined.

        Returns:
            FunctionInfo if found, None otherwise.
        """
        if file_path is None:
            return None

        module_info = self._get_module_info(file_path)
        if module_info is None:
            return None

        # Extract function name from qualified name
        func_name = qualified_name.rsplit("::", 1)[-1] if "::" in qualified_name else qualified_name

        # Search top-level functions
        for func in module_info.functions:
            if func.name == func_name:
                return func

        # Search class methods
        for cls in module_info.classes:
            for method in cls.methods:
                if method.name == func_name:
                    return method

        return None

    def _qualify_name(self, file_path: str, function_name: str) -> str:
        """Create a qualified name from file path and function name.

        Produces names like "path/to/file.py::function_name" using paths
        relative to the project root when possible.
        """
        try:
            rel_path = str(Path(file_path).relative_to(self._project_root))
        except ValueError:
            rel_path = file_path

        return f"{rel_path}::{function_name}"

    def _parse_file_ast(self, path: str) -> ast.Module | None:
        """Parse a Python file into an AST tree.

        Returns None if the file cannot be read or parsed.
        """
        try:
            source = Path(path).read_text(encoding="utf-8")
            return ast.parse(source, filename=path)
        except (OSError, SyntaxError, UnicodeDecodeError) as e:
            logger.error("Failed to parse AST for %s: %s", path, e)
            return None
