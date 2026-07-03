"""Output formatting models."""

from dataclasses import dataclass, field
from typing import Optional

from codesense.models.state import Conflict


@dataclass
class CodeSnippet:
    """A code snippet with language annotation for syntax highlighting."""

    code: str
    language: str
    label: Optional[str] = None


@dataclass
class TableData:
    """Structured tabular data for Rich table formatting."""

    headers: list[str]
    rows: list[list[str]]
    title: Optional[str] = None


@dataclass
class CommandOutput:
    """Base output model consumed by the Rich formatter."""

    title: str
    content: str  # markdown content
    code_snippets: list[CodeSnippet] = field(default_factory=list)
    tables: list[TableData] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    confidence: Optional[float] = None
    is_demo_mode: bool = False


@dataclass
class CommandParams:
    """Parsed CLI arguments passed to capability handlers."""

    path: Optional[str] = None
    query: Optional[str] = None
    mock: bool = False
    output: Optional[str] = None
    line_number: Optional[int] = None
    limit: Optional[int] = None
    function_name: Optional[str] = None
    line_range: Optional[str] = None  # e.g. "10-20"
