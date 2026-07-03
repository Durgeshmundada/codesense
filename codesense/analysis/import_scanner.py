"""Import scanner for analyzing module dependencies using AST parsing."""

import ast
import logging
from pathlib import Path

from codesense.models.analysis import ImportGraph

logger = logging.getLogger(__name__)


class ImportScanner:
    """Scans Python modules to build import dependency graphs.

    Categorizes dependencies into internal (project) imports, external (third-party)
    imports, environment variable accesses, and external API calls.
    """

    def __init__(self, project_root: str) -> None:
        """Initialize the ImportScanner.

        Args:
            project_root: The root directory of the project. Used to distinguish
                internal imports from external (third-party) ones.
        """
        self.project_root = Path(project_root).resolve()

    def scan(self, module_path: str) -> ImportGraph:
        """Scan the given module path for dependencies.

        Args:
            module_path: Path to a Python file or directory to scan.

        Returns:
            ImportGraph with categorized dependencies.
        """
        path = Path(module_path).resolve()

        internal_deps: set[str] = set()
        external_deps: set[str] = set()
        env_vars: set[str] = set()
        external_apis: set[str] = set()

        if path.is_file():
            py_files = [path] if path.suffix == ".py" else []
        elif path.is_dir():
            py_files = list(path.rglob("*.py"))
        else:
            # Path doesn't exist, return empty graph
            return ImportGraph(
                module=module_path,
                internal_deps=[],
                external_deps=[],
                env_vars=[],
                external_apis=[],
            )

        for py_file in py_files:
            self._scan_file(py_file, internal_deps, external_deps, env_vars, external_apis)

        return ImportGraph(
            module=module_path,
            internal_deps=sorted(internal_deps),
            external_deps=sorted(external_deps),
            env_vars=sorted(env_vars),
            external_apis=sorted(external_apis),
        )

    def scan_file(self, file_path: str) -> ImportGraph:
        """Scan a single Python file for dependencies.

        Args:
            file_path: Path to a Python file to scan.

        Returns:
            ImportGraph with categorized dependencies for the single file.
        """
        path = Path(file_path).resolve()

        internal_deps: set[str] = set()
        external_deps: set[str] = set()
        env_vars: set[str] = set()
        external_apis: set[str] = set()

        if path.is_file() and path.suffix == ".py":
            self._scan_file(path, internal_deps, external_deps, env_vars, external_apis)

        return ImportGraph(
            module=file_path,
            internal_deps=sorted(internal_deps),
            external_deps=sorted(external_deps),
            env_vars=sorted(env_vars),
            external_apis=sorted(external_apis),
        )

    def _scan_file(
        self,
        file_path: Path,
        internal_deps: set[str],
        external_deps: set[str],
        env_vars: set[str],
        external_apis: set[str],
    ) -> None:
        """Parse a single Python file and extract dependencies.

        Handles parse failures gracefully by logging a warning and skipping unparseable files.
        """
        try:
            source = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("Could not read file %s: %s", file_path, e)
            return

        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError as e:
            logger.warning("Could not parse file %s: %s", file_path, e)
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self._process_import(node, internal_deps, external_deps)
            elif isinstance(node, ast.ImportFrom):
                self._process_import_from(node, file_path, internal_deps, external_deps)
            elif isinstance(node, ast.Call):
                self._check_env_var_access(node, env_vars)
                self._check_external_api_call(node, external_apis)
            elif isinstance(node, ast.Subscript):
                self._check_environ_subscript(node, env_vars)

    def _process_import(
        self,
        node: ast.Import,
        internal_deps: set[str],
        external_deps: set[str],
    ) -> None:
        """Process `import X` statements."""
        for alias in node.names:
            module_name = alias.name
            top_level = module_name.split(".")[0]
            if self._is_internal_module(top_level):
                internal_deps.add(module_name)
            else:
                external_deps.add(module_name)

    def _process_import_from(
        self,
        node: ast.ImportFrom,
        file_path: Path,
        internal_deps: set[str],
        external_deps: set[str],
    ) -> None:
        """Process `from X import Y` statements."""
        if node.module is None:
            # Relative import with no module (e.g., `from . import something`)
            # These are always internal
            for alias in node.names:
                internal_deps.add(alias.name)
            return

        if node.level > 0:
            # Relative import (e.g., `from .foo import bar`)
            # Always internal
            internal_deps.add(node.module)
        else:
            # Absolute import
            top_level = node.module.split(".")[0]
            if self._is_internal_module(top_level):
                internal_deps.add(node.module)
            else:
                external_deps.add(node.module)

    def _is_internal_module(self, top_level_name: str) -> bool:
        """Check if a top-level module name exists under the project root.

        A module is considered internal if a directory or .py file with
        that name exists in the project root.
        """
        # Check for package directory
        package_dir = self.project_root / top_level_name
        if package_dir.is_dir():
            return True

        # Check for module file
        module_file = self.project_root / f"{top_level_name}.py"
        if module_file.is_file():
            return True

        return False

    def _check_env_var_access(self, node: ast.Call, env_vars: set[str]) -> None:
        """Detect os.getenv(), os.environ.get(), and os.environ[...] calls."""
        # Check os.getenv("VAR")
        if self._is_call_to(node, "os", "getenv"):
            var_name = self._extract_first_string_arg(node)
            if var_name:
                env_vars.add(var_name)
            return

        # Check os.environ.get("VAR")
        if self._is_method_call_on(node, "os", "environ", "get"):
            var_name = self._extract_first_string_arg(node)
            if var_name:
                env_vars.add(var_name)
            return

    def _check_external_api_call(self, node: ast.Call, external_apis: set[str]) -> None:
        """Detect HTTP calls like requests.get/post, httpx.get/post, etc."""
        http_libraries = ("requests", "httpx", "aiohttp", "urllib3")
        http_methods = ("get", "post", "put", "patch", "delete", "head", "options", "request")

        func = node.func

        # Match pattern: library.method (e.g., requests.get)
        if isinstance(func, ast.Attribute) and func.attr in http_methods:
            if isinstance(func.value, ast.Name) and func.value.id in http_libraries:
                url = self._extract_first_string_arg(node)
                call_repr = f"{func.value.id}.{func.attr}"
                if url:
                    external_apis.add(f"{call_repr}({url})")
                else:
                    external_apis.add(call_repr)
                return

            # Match pattern: library.Client().method or session.method
            # e.g., httpx.AsyncClient().get(...)
            if isinstance(func.value, ast.Call):
                inner = func.value.func
                if isinstance(inner, ast.Attribute) and isinstance(inner.value, ast.Name):
                    if inner.value.id in http_libraries:
                        url = self._extract_first_string_arg(node)
                        call_repr = f"{inner.value.id}.{inner.attr}().{func.attr}"
                        if url:
                            external_apis.add(f"{call_repr}({url})")
                        else:
                            external_apis.add(call_repr)

    def _is_call_to(self, node: ast.Call, obj_name: str, method_name: str) -> bool:
        """Check if node is a call like `obj.method(...)`."""
        func = node.func
        return (
            isinstance(func, ast.Attribute)
            and func.attr == method_name
            and isinstance(func.value, ast.Name)
            and func.value.id == obj_name
        )

    def _is_method_call_on(
        self, node: ast.Call, obj_name: str, attr_name: str, method_name: str
    ) -> bool:
        """Check if node is a call like `obj.attr.method(...)`."""
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != method_name:
            return False
        value = func.value
        if not isinstance(value, ast.Attribute) or value.attr != attr_name:
            return False
        return isinstance(value.value, ast.Name) and value.value.id == obj_name

    def _check_environ_subscript(self, node: ast.Subscript, env_vars: set[str]) -> None:
        """Detect os.environ["VAR"] subscript access."""
        value = node.value
        if (
            isinstance(value, ast.Attribute)
            and value.attr == "environ"
            and isinstance(value.value, ast.Name)
            and value.value.id == "os"
        ):
            # Extract the key from the subscript slice
            slice_node = node.slice
            if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
                env_vars.add(slice_node.value)

    def _extract_first_string_arg(self, node: ast.Call) -> str | None:
        """Extract the first positional string argument from a call node."""
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(
            node.args[0].value, str
        ):
            return node.args[0].value
        return None
