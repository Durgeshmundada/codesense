"""AST-based static analysis walker for Python source files."""

import ast
import logging
from pathlib import Path

from codesense.models.analysis import ClassInfo, FunctionInfo, ModuleInfo

logger = logging.getLogger(__name__)


class ASTWalker:
    """Walks Python AST to extract structural information from source files."""

    def parse_module(self, path: str) -> ModuleInfo:
        """Parse a Python file and extract classes, functions, and imports.

        Args:
            path: Path to the Python source file.

        Returns:
            ModuleInfo with extracted classes, functions, and imports.
            Returns empty ModuleInfo if the file cannot be parsed.

        Raises:
            ValueError: If the file does not exist.
        """
        if not path.endswith(".py"):
            return ModuleInfo(path=path, classes=[], functions=[], imports=[])

        self._validate_file_exists(path)

        tree = self._parse_file(path)
        if tree is None:
            return ModuleInfo(path=path, classes=[], functions=[], imports=[])

        classes = self._extract_classes_from_tree(tree)
        functions = self._extract_top_level_functions(tree)
        imports = self._extract_imports_from_tree(tree)

        return ModuleInfo(
            path=path,
            classes=classes,
            functions=functions,
            imports=imports,
        )

    def extract_classes(self, path: str) -> list[ClassInfo]:
        """Extract class definitions from a Python file.

        Args:
            path: Path to the Python source file.

        Returns:
            List of ClassInfo with name, bases, methods, attributes, and line number.

        Raises:
            ValueError: If the file does not exist.
        """
        if not path.endswith(".py"):
            return []

        self._validate_file_exists(path)

        tree = self._parse_file(path)
        if tree is None:
            return []

        return self._extract_classes_from_tree(tree)

    def extract_functions(self, path: str) -> list[FunctionInfo]:
        """Extract top-level function definitions from a Python file.

        Args:
            path: Path to the Python source file.

        Returns:
            List of FunctionInfo with name, parameters, return type, calls, and line number.

        Raises:
            ValueError: If the file does not exist.
        """
        if not path.endswith(".py"):
            return []

        self._validate_file_exists(path)

        tree = self._parse_file(path)
        if tree is None:
            return []

        return self._extract_top_level_functions(tree)

    def extract_imports(self, path: str) -> list[str]:
        """Extract all import statements from a Python file.

        Args:
            path: Path to the Python source file.

        Returns:
            List of module path strings for all imports.

        Raises:
            ValueError: If the file does not exist.
        """
        if not path.endswith(".py"):
            return []

        self._validate_file_exists(path)

        tree = self._parse_file(path)
        if tree is None:
            return []

        return self._extract_imports_from_tree(tree)

    def _validate_file_exists(self, path: str) -> None:
        """Validate that the file exists, raising ValueError if not.

        Raises:
            ValueError: If the file does not exist.
        """
        if not Path(path).exists():
            raise ValueError(f"File does not exist: {path}")

    def _parse_file(self, path: str) -> ast.Module | None:
        """Parse a Python file into an AST.

        Returns None if the file cannot be read or parsed.
        Logs a warning on parse failure.
        """
        try:
            source = Path(path).read_text(encoding="utf-8")
            return ast.parse(source, filename=path)
        except (SyntaxError, UnicodeDecodeError) as e:
            logger.warning("Failed to parse %s: %s", path, e)
            return None

    def _extract_classes_from_tree(self, tree: ast.Module) -> list[ClassInfo]:
        """Extract all class definitions from the AST."""
        classes: list[ClassInfo] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = [self._get_name(base) for base in node.bases]
                methods = self._extract_methods(node)
                attributes = self._extract_class_attributes(node)

                classes.append(
                    ClassInfo(
                        name=node.name,
                        bases=bases,
                        methods=methods,
                        attributes=attributes,
                        line_number=node.lineno,
                    )
                )

        return classes

    def _extract_top_level_functions(self, tree: ast.Module) -> list[FunctionInfo]:
        """Extract only top-level function definitions (not methods)."""
        functions: list[FunctionInfo] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(self._build_function_info(node))

        return functions

    def _extract_methods(self, class_node: ast.ClassDef) -> list[FunctionInfo]:
        """Extract method definitions from a class."""
        methods: list[FunctionInfo] = []

        for node in ast.iter_child_nodes(class_node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(self._build_function_info(node))

        return methods

    def _build_function_info(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> FunctionInfo:
        """Build a FunctionInfo from a function/method AST node."""
        parameters = self._extract_parameters(node)
        return_type = self._get_return_annotation(node)
        calls = self._extract_calls(node)
        is_async = isinstance(node, ast.AsyncFunctionDef)

        return FunctionInfo(
            name=node.name,
            parameters=parameters,
            return_type=return_type,
            calls=calls,
            line_number=node.lineno,
            is_async=is_async,
        )

    def _extract_parameters(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> list[str]:
        """Extract parameter names from a function definition."""
        params: list[str] = []

        for arg in node.args.posonlyargs:
            params.append(arg.arg)
        for arg in node.args.args:
            params.append(arg.arg)
        if node.args.vararg:
            params.append(f"*{node.args.vararg.arg}")
        for arg in node.args.kwonlyargs:
            params.append(arg.arg)
        if node.args.kwarg:
            params.append(f"**{node.args.kwarg.arg}")

        return params

    def _get_return_annotation(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> str | None:
        """Extract return type annotation as a string, if present."""
        if node.returns is None:
            return None
        return ast.unparse(node.returns)

    def _extract_calls(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> list[str]:
        """Extract all function/method call names within a function body."""
        calls: list[str] = []

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_name = self._get_call_name(child)
                if call_name and call_name not in calls:
                    calls.append(call_name)

        return calls

    def _get_call_name(self, node: ast.Call) -> str | None:
        """Get the name of a function call."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            # Handle chained attribute access like obj.method()
            parts = self._get_attribute_parts(node.func)
            if parts:
                return ".".join(parts)
        return None

    def _get_attribute_parts(self, node: ast.Attribute) -> list[str]:
        """Recursively extract dotted attribute access parts."""
        parts: list[str] = []

        if isinstance(node.value, ast.Name):
            parts.append(node.value.id)
        elif isinstance(node.value, ast.Attribute):
            parts.extend(self._get_attribute_parts(node.value))
        else:
            return []

        parts.append(node.attr)
        return parts

    def _extract_class_attributes(self, class_node: ast.ClassDef) -> list[str]:
        """Extract class-level and instance attributes."""
        attributes: list[str] = []

        for node in ast.walk(class_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    # Class-level: x = value
                    if isinstance(target, ast.Name) and target.id not in attributes:
                        attributes.append(target.id)
                    # Instance-level: self.x = value
                    elif (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and target.attr not in attributes
                    ):
                        attributes.append(target.attr)
            elif isinstance(node, ast.AnnAssign):
                # Annotated assignments: x: int = value or self.x: int = value
                if (
                    isinstance(node.target, ast.Name)
                    and node.target.id not in attributes
                ):
                    attributes.append(node.target.id)
                elif (
                    isinstance(node.target, ast.Attribute)
                    and isinstance(node.target.value, ast.Name)
                    and node.target.value.id == "self"
                    and node.target.attr not in attributes
                ):
                    attributes.append(node.target.attr)

        return attributes

    def _extract_imports_from_tree(self, tree: ast.Module) -> list[str]:
        """Extract all import module paths from the AST."""
        imports: list[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in imports:
                        imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module not in imports:
                    imports.append(node.module)

        return imports

    def _get_name(self, node: ast.expr) -> str:
        """Get a string representation of an AST expression (for base classes, etc.)."""
        return ast.unparse(node)
