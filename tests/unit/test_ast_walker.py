"""Unit tests for ASTWalker."""

import os
import tempfile

import pytest

from codesense.analysis.ast_walker import ASTWalker


@pytest.fixture
def walker():
    return ASTWalker()


@pytest.fixture
def sample_python_file():
    """Create a temporary Python file with various constructs."""
    code = '''\
import os
from pathlib import Path
from typing import Optional

class Animal:
    """Base class."""
    species: str = "unknown"

    def __init__(self, name: str):
        self.name = name

    def speak(self) -> str:
        return ""

class Dog(Animal):
    """A dog."""
    breed: str = "mixed"

    def __init__(self, name: str, breed: str):
        super().__init__(name)
        self.breed = breed

    def speak(self) -> str:
        return "woof"

    async def fetch(self, item: str) -> bool:
        result = await self.find(item)
        return result is not None

def create_dog(name: str, breed: str = "labrador") -> Dog:
    """Factory function."""
    dog = Dog(name, breed)
    print(f"Created {dog.name}")
    return dog

async def async_create(name: str, **kwargs) -> Dog:
    """Async factory."""
    return create_dog(name, kwargs.get("breed", "unknown"))
'''
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def syntax_error_file():
    """Create a temporary Python file with a syntax error."""
    code = "def broken(\n    this is not valid python\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    yield path
    os.unlink(path)


class TestParseModule:
    def test_extracts_classes(self, walker, sample_python_file):
        info = walker.parse_module(sample_python_file)
        class_names = [c.name for c in info.classes]
        assert "Animal" in class_names
        assert "Dog" in class_names

    def test_extracts_top_level_functions(self, walker, sample_python_file):
        info = walker.parse_module(sample_python_file)
        func_names = [f.name for f in info.functions]
        assert "create_dog" in func_names
        assert "async_create" in func_names
        # Methods should NOT appear as top-level functions
        assert "__init__" not in func_names
        assert "speak" not in func_names

    def test_extracts_imports(self, walker, sample_python_file):
        info = walker.parse_module(sample_python_file)
        assert "os" in info.imports
        assert "pathlib" in info.imports
        assert "typing" in info.imports

    def test_returns_correct_path(self, walker, sample_python_file):
        info = walker.parse_module(sample_python_file)
        assert info.path == sample_python_file

    def test_non_python_file_returns_empty(self, walker, tmp_path):
        non_py = tmp_path / "README.md"
        non_py.write_text("# Hello")
        info = walker.parse_module(str(non_py))
        assert info.path == str(non_py)
        assert info.classes == []
        assert info.functions == []
        assert info.imports == []

    def test_nonexistent_file_raises_value_error(self, walker):
        with pytest.raises(ValueError, match="does not exist"):
            walker.parse_module("does_not_exist.py")

    def test_syntax_error_returns_empty(self, walker, syntax_error_file):
        info = walker.parse_module(syntax_error_file)
        assert info.path == syntax_error_file
        assert info.classes == []
        assert info.functions == []
        assert info.imports == []


class TestExtractClasses:
    def test_extracts_class_names(self, walker, sample_python_file):
        classes = walker.extract_classes(sample_python_file)
        names = [c.name for c in classes]
        assert "Animal" in names
        assert "Dog" in names

    def test_extracts_base_classes(self, walker, sample_python_file):
        classes = walker.extract_classes(sample_python_file)
        dog = next(c for c in classes if c.name == "Dog")
        assert "Animal" in dog.bases

    def test_extracts_methods(self, walker, sample_python_file):
        classes = walker.extract_classes(sample_python_file)
        dog = next(c for c in classes if c.name == "Dog")
        method_names = [m.name for m in dog.methods]
        assert "__init__" in method_names
        assert "speak" in method_names
        assert "fetch" in method_names

    def test_extracts_async_methods(self, walker, sample_python_file):
        classes = walker.extract_classes(sample_python_file)
        dog = next(c for c in classes if c.name == "Dog")
        fetch = next(m for m in dog.methods if m.name == "fetch")
        assert fetch.is_async is True

    def test_extracts_attributes(self, walker, sample_python_file):
        classes = walker.extract_classes(sample_python_file)
        animal = next(c for c in classes if c.name == "Animal")
        assert "species" in animal.attributes
        assert "name" in animal.attributes

    def test_extracts_line_numbers(self, walker, sample_python_file):
        classes = walker.extract_classes(sample_python_file)
        # Animal is defined first, Dog second
        animal = next(c for c in classes if c.name == "Animal")
        dog = next(c for c in classes if c.name == "Dog")
        assert animal.line_number < dog.line_number

    def test_non_python_file_returns_empty(self, walker, tmp_path):
        non_py = tmp_path / "config.yaml"
        non_py.write_text("key: value")
        assert walker.extract_classes(str(non_py)) == []


class TestExtractFunctions:
    def test_extracts_function_names(self, walker, sample_python_file):
        funcs = walker.extract_functions(sample_python_file)
        names = [f.name for f in funcs]
        assert "create_dog" in names
        assert "async_create" in names

    def test_extracts_parameters(self, walker, sample_python_file):
        funcs = walker.extract_functions(sample_python_file)
        create_dog = next(f for f in funcs if f.name == "create_dog")
        assert "name" in create_dog.parameters
        assert "breed" in create_dog.parameters

    def test_extracts_kwargs(self, walker, sample_python_file):
        funcs = walker.extract_functions(sample_python_file)
        async_create = next(f for f in funcs if f.name == "async_create")
        assert "**kwargs" in async_create.parameters

    def test_extracts_return_type(self, walker, sample_python_file):
        funcs = walker.extract_functions(sample_python_file)
        create_dog = next(f for f in funcs if f.name == "create_dog")
        assert create_dog.return_type == "Dog"

    def test_extracts_calls(self, walker, sample_python_file):
        funcs = walker.extract_functions(sample_python_file)
        create_dog = next(f for f in funcs if f.name == "create_dog")
        assert "Dog" in create_dog.calls
        assert "print" in create_dog.calls

    def test_detects_async(self, walker, sample_python_file):
        funcs = walker.extract_functions(sample_python_file)
        async_create = next(f for f in funcs if f.name == "async_create")
        assert async_create.is_async is True
        create_dog = next(f for f in funcs if f.name == "create_dog")
        assert create_dog.is_async is False

    def test_non_python_file_returns_empty(self, walker, tmp_path):
        non_py = tmp_path / "data.json"
        non_py.write_text("{}")
        assert walker.extract_functions(str(non_py)) == []


class TestExtractImports:
    def test_extracts_import_statements(self, walker, sample_python_file):
        imports = walker.extract_imports(sample_python_file)
        assert "os" in imports

    def test_extracts_from_imports(self, walker, sample_python_file):
        imports = walker.extract_imports(sample_python_file)
        assert "pathlib" in imports
        assert "typing" in imports

    def test_no_duplicates(self, walker, sample_python_file):
        imports = walker.extract_imports(sample_python_file)
        assert len(imports) == len(set(imports))

    def test_non_python_file_returns_empty(self, walker, tmp_path):
        non_py = tmp_path / "Makefile"
        non_py.write_text("all: build")
        assert walker.extract_imports(str(non_py)) == []

    def test_nonexistent_file_raises_value_error(self, walker):
        with pytest.raises(ValueError, match="does not exist"):
            walker.extract_imports("ghost.py")
