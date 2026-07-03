"""Unit tests for ImportScanner."""

import os
import tempfile
from pathlib import Path

import pytest

from codesense.analysis.import_scanner import ImportScanner
from codesense.models.analysis import ImportGraph


@pytest.fixture
def tmp_project(tmp_path):
    """Create a temporary project structure for testing."""
    # Create project package
    pkg_dir = tmp_path / "myproject"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")

    # Create a module that imports internal and external deps
    (pkg_dir / "main.py").write_text(
        'import os\n'
        'import sys\n'
        'import requests\n'
        'from myproject.utils import helper\n'
        'from myproject import config\n'
        '\n'
        'API_KEY = os.getenv("API_KEY")\n'
        'DB_HOST = os.environ.get("DB_HOST")\n'
        'SECRET = os.environ["SECRET_TOKEN"]\n'
        '\n'
        'response = requests.get("https://api.example.com/data")\n'
        'requests.post("https://api.example.com/submit")\n'
    )

    # Create utils module
    (pkg_dir / "utils.py").write_text(
        'import json\n'
        'from pathlib import Path\n'
        '\n'
        'def helper():\n'
        '    return "help"\n'
    )

    # Create config module
    (pkg_dir / "config.py").write_text(
        'import os\n'
        '\n'
        'DEBUG = os.getenv("DEBUG")\n'
    )

    return tmp_path


class TestImportScanner:
    """Tests for ImportScanner."""

    def test_scan_single_file_internal_deps(self, tmp_project):
        """Internal imports are correctly categorized."""
        scanner = ImportScanner(project_root=str(tmp_project))
        result = scanner.scan(str(tmp_project / "myproject" / "main.py"))

        assert isinstance(result, ImportGraph)
        assert "myproject.utils" in result.internal_deps
        assert "myproject" in result.internal_deps or "myproject.config" in result.internal_deps

    def test_scan_single_file_external_deps(self, tmp_project):
        """External imports are correctly categorized."""
        scanner = ImportScanner(project_root=str(tmp_project))
        result = scanner.scan(str(tmp_project / "myproject" / "main.py"))

        assert "os" in result.external_deps
        assert "sys" in result.external_deps
        assert "requests" in result.external_deps

    def test_scan_single_file_env_vars(self, tmp_project):
        """Environment variable accesses are detected."""
        scanner = ImportScanner(project_root=str(tmp_project))
        result = scanner.scan(str(tmp_project / "myproject" / "main.py"))

        assert "API_KEY" in result.env_vars
        assert "DB_HOST" in result.env_vars
        assert "SECRET_TOKEN" in result.env_vars

    def test_scan_single_file_external_apis(self, tmp_project):
        """External API calls are detected."""
        scanner = ImportScanner(project_root=str(tmp_project))
        result = scanner.scan(str(tmp_project / "myproject" / "main.py"))

        # Should detect requests.get and requests.post calls
        api_strs = " ".join(result.external_apis)
        assert "requests.get" in api_strs
        assert "requests.post" in api_strs
        assert "api.example.com" in api_strs

    def test_scan_directory_aggregates_all_files(self, tmp_project):
        """Scanning a directory aggregates imports from all .py files."""
        scanner = ImportScanner(project_root=str(tmp_project))
        result = scanner.scan(str(tmp_project / "myproject"))

        # Should include env vars from config.py
        assert "DEBUG" in result.env_vars
        # Should include env vars from main.py
        assert "API_KEY" in result.env_vars

    def test_scan_nonexistent_path_returns_empty(self, tmp_project):
        """Non-existent paths return an empty ImportGraph."""
        scanner = ImportScanner(project_root=str(tmp_project))
        result = scanner.scan(str(tmp_project / "nonexistent.py"))

        assert result.internal_deps == []
        assert result.external_deps == []
        assert result.env_vars == []
        assert result.external_apis == []

    def test_scan_unparseable_file_gracefully_skipped(self, tmp_project):
        """Unparseable files are skipped without raising exceptions."""
        bad_file = tmp_project / "myproject" / "broken.py"
        bad_file.write_text("def broken(:\n    pass\n")

        scanner = ImportScanner(project_root=str(tmp_project))
        # Should not raise
        result = scanner.scan(str(tmp_project / "myproject"))

        # Still gets results from other valid files
        assert "API_KEY" in result.env_vars

    def test_scan_non_python_file_ignored(self, tmp_project):
        """Non-Python files are ignored."""
        txt_file = tmp_project / "myproject" / "readme.txt"
        txt_file.write_text("import fake_module\n")

        scanner = ImportScanner(project_root=str(tmp_project))
        result = scanner.scan(str(txt_file))

        assert result.internal_deps == []
        assert result.external_deps == []

    def test_scan_relative_imports_are_internal(self, tmp_project):
        """Relative imports are categorized as internal."""
        (tmp_project / "myproject" / "sub.py").write_text(
            'from . import utils\n'
            'from .config import DEBUG\n'
        )

        scanner = ImportScanner(project_root=str(tmp_project))
        result = scanner.scan(str(tmp_project / "myproject" / "sub.py"))

        # Relative imports should be in internal_deps
        assert len(result.internal_deps) > 0
        assert len(result.external_deps) == 0

    def test_scan_httpx_calls_detected(self, tmp_project):
        """httpx calls are detected as external APIs."""
        (tmp_project / "myproject" / "client.py").write_text(
            'import httpx\n'
            '\n'
            'resp = httpx.get("https://service.example.com/api")\n'
        )

        scanner = ImportScanner(project_root=str(tmp_project))
        result = scanner.scan(str(tmp_project / "myproject" / "client.py"))

        api_strs = " ".join(result.external_apis)
        assert "httpx.get" in api_strs
        assert "service.example.com" in api_strs

    def test_module_field_set_to_input_path(self, tmp_project):
        """ImportGraph.module is set to the input module_path."""
        scanner = ImportScanner(project_root=str(tmp_project))
        path = str(tmp_project / "myproject" / "main.py")
        result = scanner.scan(path)

        assert result.module == path

    def test_results_are_sorted(self, tmp_project):
        """All output lists are sorted."""
        scanner = ImportScanner(project_root=str(tmp_project))
        result = scanner.scan(str(tmp_project / "myproject"))

        assert result.internal_deps == sorted(result.internal_deps)
        assert result.external_deps == sorted(result.external_deps)
        assert result.env_vars == sorted(result.env_vars)
        assert result.external_apis == sorted(result.external_apis)


class TestImportScannerScanFile:
    """Tests for ImportScanner.scan_file method."""

    def test_scan_file_returns_import_graph(self, tmp_project):
        """scan_file returns an ImportGraph for a single file."""
        scanner = ImportScanner(project_root=str(tmp_project))
        result = scanner.scan_file(str(tmp_project / "myproject" / "main.py"))

        assert isinstance(result, ImportGraph)
        assert result.module == str(tmp_project / "myproject" / "main.py")

    def test_scan_file_detects_deps(self, tmp_project):
        """scan_file correctly categorizes imports from a single file."""
        scanner = ImportScanner(project_root=str(tmp_project))
        result = scanner.scan_file(str(tmp_project / "myproject" / "main.py"))

        assert "myproject.utils" in result.internal_deps
        assert "requests" in result.external_deps
        assert "API_KEY" in result.env_vars

    def test_scan_file_non_python_returns_empty(self, tmp_project):
        """scan_file returns empty graph for non-.py files."""
        txt_file = tmp_project / "myproject" / "data.txt"
        txt_file.write_text("import something\n")

        scanner = ImportScanner(project_root=str(tmp_project))
        result = scanner.scan_file(str(txt_file))

        assert result.internal_deps == []
        assert result.external_deps == []

    def test_scan_file_unparseable_logs_warning(self, tmp_project, caplog):
        """scan_file logs a warning for unparseable files."""
        bad_file = tmp_project / "myproject" / "bad.py"
        bad_file.write_text("def broken(:\n    pass\n")

        scanner = ImportScanner(project_root=str(tmp_project))
        import logging
        with caplog.at_level(logging.WARNING):
            result = scanner.scan_file(str(bad_file))

        assert result.internal_deps == []
        assert "Could not parse file" in caplog.text
