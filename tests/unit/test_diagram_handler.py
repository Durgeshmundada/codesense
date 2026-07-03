"""Unit tests for the diagram capability handler."""

import tempfile
from pathlib import Path

import pytest

from codesense.capabilities.diagram import DiagramHandler
from codesense.models.output import CommandParams


@pytest.fixture
def sample_python_file(tmp_path: Path) -> str:
    """Create a sample Python file for testing."""
    code = '''"""Sample module for testing diagram generation."""

class Animal:
    """Base class for animals."""

    def __init__(self, name: str) -> None:
        self.name = name

    def speak(self) -> str:
        return ""


class Dog(Animal):
    """A dog is an animal."""

    def __init__(self, name: str, breed: str) -> None:
        super().__init__(name)
        self.breed = breed

    def speak(self) -> str:
        return "Woof!"

    def fetch(self, item: str) -> str:
        return f"{self.name} fetches {item}"


class Cat(Animal):
    """A cat is an animal."""

    def speak(self) -> str:
        return "Meow!"


def create_pet(animal_type: str, name: str) -> Animal:
    """Factory function for creating pets."""
    if animal_type == "dog":
        return Dog(name, "Unknown")
    return Cat(name)
'''
    file_path = tmp_path / "animals.py"
    file_path.write_text(code, encoding="utf-8")
    return str(file_path)


@pytest.fixture
def sample_directory(tmp_path: Path) -> str:
    """Create a directory with multiple Python files."""
    # Module 1: models
    models_code = '''"""Data models."""

class User:
    def __init__(self, name: str, email: str) -> None:
        self.name = name
        self.email = email


class Admin(User):
    def __init__(self, name: str, email: str, role: str) -> None:
        super().__init__(name, email)
        self.role = role
'''
    # Module 2: service
    service_code = '''"""Service layer."""

from models import User

class UserService:
    def get_user(self, user_id: int) -> User:
        pass

    def create_user(self, name: str, email: str) -> User:
        return User(name, email)
'''
    models_file = tmp_path / "models.py"
    models_file.write_text(models_code, encoding="utf-8")

    service_file = tmp_path / "service.py"
    service_file.write_text(service_code, encoding="utf-8")

    return str(tmp_path)


@pytest.fixture
def handler() -> DiagramHandler:
    """Create a DiagramHandler instance."""
    return DiagramHandler()


class TestDiagramHandlerRun:
    """Test the main run() method."""

    def test_returns_command_output_with_correct_title(
        self, handler: DiagramHandler, sample_python_file: str
    ):
        params = CommandParams(path=sample_python_file)
        result = handler.run(params)
        assert result.title == "Code Diagram"

    def test_returns_mermaid_code_snippet(
        self, handler: DiagramHandler, sample_python_file: str
    ):
        params = CommandParams(path=sample_python_file)
        result = handler.run(params)
        assert len(result.code_snippets) == 1
        assert result.code_snippets[0].language == "mermaid"
        assert result.code_snippets[0].code  # non-empty

    def test_flowchart_is_default_type(
        self, handler: DiagramHandler, sample_python_file: str
    ):
        params = CommandParams(path=sample_python_file)
        result = handler.run(params)
        assert "flowchart" in result.code_snippets[0].label

    def test_architecture_diagram_type(
        self, handler: DiagramHandler, sample_python_file: str
    ):
        params = CommandParams(path=sample_python_file, query="architecture")
        result = handler.run(params)
        assert "architecture" in result.code_snippets[0].label
        assert "classDiagram" in result.code_snippets[0].code

    def test_sequence_diagram_type(
        self, handler: DiagramHandler, sample_python_file: str
    ):
        params = CommandParams(path=sample_python_file, query="sequence")
        result = handler.run(params)
        assert "sequence" in result.code_snippets[0].label
        assert "sequenceDiagram" in result.code_snippets[0].code

    def test_demo_mode_flag(
        self, handler: DiagramHandler, sample_python_file: str
    ):
        params = CommandParams(path=sample_python_file, mock=True)
        result = handler.run(params)
        assert result.is_demo_mode is True

    def test_content_contains_mermaid_code_block(
        self, handler: DiagramHandler, sample_python_file: str
    ):
        params = CommandParams(path=sample_python_file)
        result = handler.run(params)
        # Content should be the Mermaid diagram as a code block
        assert "```mermaid" in result.content


class TestDiagramHandlerErrorHandling:
    """Test error handling for invalid inputs."""

    def test_nonexistent_path_returns_no_files_message(
        self, handler: DiagramHandler
    ):
        params = CommandParams(path="/nonexistent/path/file.py")
        result = handler.run(params)
        assert "No Python files found" in result.content

    def test_non_python_file_returns_no_files_message(
        self, handler: DiagramHandler, tmp_path: Path
    ):
        txt_file = tmp_path / "readme.txt"
        txt_file.write_text("This is not Python", encoding="utf-8")
        params = CommandParams(path=str(txt_file))
        result = handler.run(params)
        assert "No Python files found" in result.content

    def test_unparseable_file_reports_error(
        self, handler: DiagramHandler, tmp_path: Path
    ):
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def broken(:\n    pass", encoding="utf-8")
        params = CommandParams(path=str(bad_file))
        result = handler.run(params)
        # Should report parse failure
        assert "could not be parsed" in result.content or "No Python files" in result.content


class TestDiagramHandlerDirectory:
    """Test directory scanning."""

    def test_scans_directory_for_python_files(
        self, handler: DiagramHandler, sample_directory: str
    ):
        params = CommandParams(path=sample_directory)
        result = handler.run(params)
        assert len(result.code_snippets) == 1
        assert result.code_snippets[0].language == "mermaid"

    def test_architecture_shows_inheritance(
        self, handler: DiagramHandler, sample_directory: str
    ):
        params = CommandParams(path=sample_directory, query="architecture")
        result = handler.run(params)
        diagram = result.code_snippets[0].code
        # Should contain classDiagram with inheritance arrows
        assert "classDiagram" in diagram


class TestDiagramHandlerOutputFile:
    """Test writing diagram to output file."""

    def test_writes_to_output_path(
        self, handler: DiagramHandler, sample_python_file: str, tmp_path: Path
    ):
        output_file = str(tmp_path / "output" / "diagram.mmd")
        params = CommandParams(path=sample_python_file, output=output_file)
        result = handler.run(params)

        # File should be written
        assert Path(output_file).exists()
        content = Path(output_file).read_text(encoding="utf-8")
        # Output uses MarkdownWriter.build_diagram_doc() which wraps in markdown
        assert "# Code Diagram" in content
        assert "```mermaid" in content
        assert result.code_snippets[0].code in content

    def test_output_path_mentioned_in_content(
        self, handler: DiagramHandler, sample_python_file: str, tmp_path: Path
    ):
        output_file = str(tmp_path / "diagram.mmd")
        params = CommandParams(path=sample_python_file, output=output_file)
        result = handler.run(params)
        assert output_file in result.content


class TestDiagramHandlerArchitecture:
    """Test architecture diagram generation specifics."""

    def test_detects_inheritance_relationships(
        self, handler: DiagramHandler, sample_python_file: str
    ):
        params = CommandParams(path=sample_python_file, query="architecture")
        result = handler.run(params)
        diagram = result.code_snippets[0].code
        # Should detect Dog inherits from Animal
        assert "classDiagram" in diagram
        # The inheritance arrow <|-- should be present
        assert "<|--" in diagram

    def test_all_classes_appear_in_diagram(
        self, handler: DiagramHandler, sample_python_file: str
    ):
        params = CommandParams(path=sample_python_file, query="architecture")
        result = handler.run(params)
        diagram = result.code_snippets[0].code
        # All three classes should appear
        assert "Animal" in diagram
        assert "Dog" in diagram
        assert "Cat" in diagram
