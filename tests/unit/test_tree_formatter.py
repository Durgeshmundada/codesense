"""Unit tests for TreeFormatter."""

import pytest

from codesense.output.tree_formatter import TreeFormatter


@pytest.fixture
def formatter():
    return TreeFormatter()


class TestTreeFormatterFormat:
    """Tests for TreeFormatter.format()."""

    def test_single_root_no_children(self, formatter):
        tree = {"name": "project", "is_dir": True, "children": []}
        result = formatter.format(tree)
        assert result == "project"

    def test_single_child(self, formatter):
        tree = {
            "name": "root",
            "is_dir": True,
            "children": [
                {"name": "file.py", "is_dir": False}
            ],
        }
        result = formatter.format(tree)
        assert "root" in result
        assert "└── file.py" in result

    def test_multiple_children(self, formatter):
        tree = {
            "name": "src",
            "is_dir": True,
            "children": [
                {"name": "main.py", "is_dir": False},
                {"name": "utils.py", "is_dir": False},
                {"name": "config.py", "is_dir": False},
            ],
        }
        result = formatter.format(tree)
        lines = result.split("\n")
        assert lines[0] == "src"
        assert "├── main.py" in lines[1]
        assert "├── utils.py" in lines[2]
        assert "└── config.py" in lines[3]

    def test_nested_children(self, formatter):
        tree = {
            "name": "project",
            "is_dir": True,
            "children": [
                {
                    "name": "src",
                    "is_dir": True,
                    "children": [
                        {"name": "app.py", "is_dir": False},
                    ],
                },
                {"name": "README.md", "is_dir": False},
            ],
        }
        result = formatter.format(tree)
        lines = result.split("\n")
        assert lines[0] == "project"
        assert "├── src" in lines[1]
        assert "│   └── app.py" in lines[2]
        assert "└── README.md" in lines[3]

    def test_annotations_on_files(self, formatter):
        tree = {
            "name": "root",
            "is_dir": True,
            "children": [
                {"name": "main.py", "is_dir": False},
            ],
        }
        annotations = {"root/main.py": "Entry point"}
        result = formatter.format(tree, annotations)
        assert "main.py" in result
        assert "Entry point" in result

    def test_annotation_on_root(self, formatter):
        tree = {
            "name": "project",
            "is_dir": True,
            "children": [],
        }
        annotations = {"project": "The main project"}
        result = formatter.format(tree, annotations)
        assert "project" in result
        assert "The main project" in result

    def test_annotations_on_nested_paths(self, formatter):
        tree = {
            "name": "root",
            "is_dir": True,
            "children": [
                {
                    "name": "src",
                    "is_dir": True,
                    "children": [
                        {"name": "utils.py", "is_dir": False},
                    ],
                },
            ],
        }
        annotations = {
            "root/src": "Source directory",
            "root/src/utils.py": "Utility functions",
        }
        result = formatter.format(tree, annotations)
        assert "src" in result
        assert "Source directory" in result
        assert "utils.py" in result
        assert "Utility functions" in result

    def test_no_annotations(self, formatter):
        tree = {
            "name": "root",
            "is_dir": True,
            "children": [
                {"name": "file.txt", "is_dir": False},
            ],
        }
        result = formatter.format(tree)
        assert "←" not in result

    def test_empty_children_list(self, formatter):
        tree = {"name": "empty", "is_dir": True, "children": []}
        result = formatter.format(tree)
        assert result == "empty"

    def test_deeply_nested(self, formatter):
        tree = {
            "name": "a",
            "is_dir": True,
            "children": [
                {
                    "name": "b",
                    "is_dir": True,
                    "children": [
                        {
                            "name": "c",
                            "is_dir": True,
                            "children": [
                                {"name": "d.txt", "is_dir": False},
                            ],
                        }
                    ],
                }
            ],
        }
        result = formatter.format(tree)
        assert "└── b" in result
        assert "    └── c" in result
        assert "        └── d.txt" in result


class TestTreeFormatterRichRender:
    """Tests for Rich tree rendering."""

    def test_render_rich_returns_string(self, formatter):
        tree = {
            "name": "project",
            "is_dir": True,
            "children": [
                {"name": "app.py", "is_dir": False},
            ],
        }
        result = formatter.render_rich(tree)
        assert isinstance(result, str)
        assert "project" in result
        assert "app.py" in result

    def test_render_rich_with_annotations(self, formatter):
        tree = {
            "name": "root",
            "is_dir": True,
            "children": [
                {"name": "main.py", "is_dir": False},
            ],
        }
        annotations = {"root/main.py": "Entry point"}
        result = formatter.render_rich(tree, annotations)
        assert "Entry point" in result
