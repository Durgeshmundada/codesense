"""CodeSense CLI — Typer application entry point with all commands wired.

Provides 12 commands for codebase understanding:
explain, describe, tree, flow, diagram, trace, deps, related, risk, onboard, ask, ingest.

Each command:
- Validates that paths exist before proceeding (error if not found)
- Displays "[DEMO MODE]" indicator when --mock flag or CODESENSE_MOCK=true
- Routes to the appropriate capability handler
- Uses RichFormatter for output rendering

Dependency injection:
- Mock vs live sources selected via --mock flag or CODESENSE_MOCK env var
- KeyRotator + GeminiService created from GEMINI_KEY_* environment variables
- No conditional branching in reasoning pipeline for source selection
"""

import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

# Ensure UTF-8 output so emoji / box-drawing characters in formatted output
# don't crash with UnicodeEncodeError on Windows consoles (cp1252) when stdout
# is redirected to a file or piped. No-op on platforms that already use UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

app = typer.Typer(
    name="codesense",
    help="A CLI tool that answers WHY code exists using multi-agent reasoning.",
    no_args_is_help=True,
)

console = Console()


# ─── Utility helpers ──────────────────────────────────────────────────────────


def _is_mock_mode(mock_flag: bool) -> bool:
    """Check if mock/demo mode is active via flag or environment variable."""
    if mock_flag:
        return True
    return os.environ.get("CODESENSE_MOCK", "").lower() == "true"


def _validate_path(path: str) -> Path:
    """Validate that a file/directory path exists. Raise typer.Exit on failure."""
    p = Path(path)
    if not p.exists():
        console.print(f"[bold red]Error:[/bold red] Path not found: {path}")
        raise typer.Exit(code=1)
    return p


def _get_project_root() -> str:
    """Determine the project root directory."""
    return os.environ.get("CODESENSE_PROJECT_ROOT", os.getcwd())


def _create_gemini_service():
    """Create a GeminiService with KeyRotator from environment variables.

    Returns:
        GeminiService instance, or None if no API keys are configured.
    """
    from codesense.llm import GeminiService, KeyRotator

    # Collect API keys from GEMINI_KEY_1, GEMINI_KEY_2, ...
    keys: list[str] = []
    index = 1
    while True:
        key = os.environ.get(f"GEMINI_KEY_{index}")
        if key is None:
            break
        if key.strip():
            keys.append(key.strip())
        index += 1

    # Fallback: GEMINI_API_KEYS (comma-separated) or GEMINI_API_KEY (single)
    if not keys:
        keys_str = os.environ.get("GEMINI_API_KEYS", "")
        if keys_str:
            keys = [k.strip() for k in keys_str.split(",") if k.strip()]

    if not keys:
        single_key = os.environ.get("GEMINI_API_KEY", "")
        if single_key.strip():
            keys = [single_key.strip()]

    if not keys:
        return None

    rotator = KeyRotator(api_keys=keys)
    return GeminiService(key_rotator=rotator)


def _require_gemini_service():
    """Create GeminiService or exit with a helpful error if not configured."""
    service = _create_gemini_service()
    if service is None:
        console.print(
            "[bold red]Error:[/bold red] No Gemini API keys configured.\n"
            "Set GEMINI_KEY_1, GEMINI_KEY_2, ... or GEMINI_API_KEYS environment variable.\n"
            "Alternatively, use --mock flag for credential-free demo mode."
        )
        raise typer.Exit(code=1)
    return service


def _render_output(output) -> None:
    """Render a CommandOutput using RichFormatter."""
    from codesense.output.formatter import RichFormatter

    formatter = RichFormatter(console=console)
    formatter.format_output(output)


# ─── CLI Commands ─────────────────────────────────────────────────────────────


@app.command()
def explain(
    path: str = typer.Argument(..., help="Code path to explain"),
    function: Optional[str] = typer.Option(None, "--function", "-f", help="Specific function name"),
    line: Optional[int] = typer.Option(None, "--line", "-l", help="Specific line number"),
    mock: bool = typer.Option(False, "--mock", help="Use demo/mock data sources"),
) -> None:
    """Explain WHY specific code exists, with confidence score and source citations."""
    _validate_path(path)
    is_mock = _is_mock_mode(mock)

    from codesense.capabilities.explain import ExplainHandler
    from codesense.models.output import CommandParams

    # Create dependencies — ExplainHandler internally creates GeminiService,
    # VectorStore, and Embedder from environment if not provided.
    # For mock mode, the reasoning graph uses MockSource for data retrieval.
    gemini_service = _create_gemini_service()

    handler = ExplainHandler(gemini_service=gemini_service)

    params = CommandParams(
        path=path,
        function_name=function,
        line_number=line,
        mock=is_mock,
    )

    result = handler.run(params)
    _render_output(result)


@app.command()
def describe(
    path: str = typer.Argument(..., help="Code path to describe"),
    function: Optional[str] = typer.Option(None, "--function", "-f", help="Specific function name"),
    lines: Optional[str] = typer.Option(None, "--lines", help="Line range (e.g. 10-20)"),
    mock: bool = typer.Option(False, "--mock", help="Use demo/mock data sources"),
) -> None:
    """Describe WHAT specified code does at a high level (no credentials required)."""
    _validate_path(path)
    is_mock = _is_mock_mode(mock)

    from codesense.capabilities.describe import DescribeCapabilityHandler
    from codesense.models.output import CommandParams

    # Describe uses LLM for generation but doesn't require git/GitHub credentials.
    # It handles missing LLM keys gracefully by returning the code without description.
    gemini_service = _create_gemini_service()
    if gemini_service is None:
        # Describe can still work — it will show an error about LLM being unavailable
        # but still read and display the code
        from codesense.capabilities.describe import DescribeHandler

        handler_inner = DescribeHandler(gemini_service=None)
        params = CommandParams(
            path=path,
            function_name=function,
            line_range=lines,
            mock=is_mock,
        )
        result = handler_inner.run(params)
    else:
        handler = DescribeCapabilityHandler(gemini_service=gemini_service)
        result = handler.run(path, function=function, lines=lines, mock=is_mock)

    _render_output(result)


@app.command()
def tree(
    path: Optional[str] = typer.Argument(None, help="Root path for tree (defaults to current directory)"),
    depth: Optional[int] = typer.Option(None, "--depth", "-d", help="Maximum depth to display"),
    annotate: bool = typer.Option(False, "--annotate", "-a", help="Generate one-line descriptions per file"),
    mock: bool = typer.Option(False, "--mock", help="Use demo/mock data sources"),
) -> None:
    """Display the project structure with annotated one-line descriptions."""
    if path is not None:
        _validate_path(path)
    is_mock = _is_mock_mode(mock)

    from codesense.capabilities.tree import TreeHandler
    from codesense.models.output import CommandParams

    params = CommandParams(
        path=path,
        mock=is_mock,
        limit=depth,
    )

    handler = TreeHandler()
    result = handler.run(params)
    _render_output(result)


@app.command()
def flow(
    path: str = typer.Argument(..., help="Function or module path to trace flow through"),
    from_function: Optional[str] = typer.Option(None, "--from", help="Starting function for flow trace"),
    feature: Optional[str] = typer.Option(None, "--feature", help="Feature name to trace"),
    mock: bool = typer.Option(False, "--mock", help="Use demo/mock data sources"),
) -> None:
    """Display the numbered execution flow through specified code showing call sequences."""
    _validate_path(path)
    is_mock = _is_mock_mode(mock)

    from codesense.capabilities.flow import FlowHandler
    from codesense.models.output import CommandParams

    project_root = _get_project_root()

    # FlowHandler uses the entry_point path; --from and --feature can
    # modify the entry point format (file::function).
    entry_point = path
    if from_function:
        entry_point = f"{path}::{from_function}"

    params = CommandParams(
        path=entry_point,
        query=feature,
        mock=is_mock,
    )

    handler = FlowHandler(project_root=project_root)
    result = handler.run(params)
    _render_output(result)


@app.command()
def diagram(
    path: Optional[str] = typer.Argument(None, help="Code path for diagram generation"),
    feature: Optional[str] = typer.Option(None, "--feature", help="Feature name to diagram"),
    file: Optional[str] = typer.Option(None, "--file", help="Specific file to diagram"),
    type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Diagram type: flowchart, sequence, or architecture"
    ),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path for the diagram"),
    mock: bool = typer.Option(False, "--mock", help="Use demo/mock data sources"),
) -> None:
    """Generate a Mermaid-format structural diagram of code relationships."""
    if path is not None:
        _validate_path(path)
    if type is not None and type not in ("flowchart", "sequence", "architecture"):
        console.print(
            f"[bold red]Error:[/bold red] Invalid diagram type '{type}'. "
            "Must be one of: flowchart, sequence, architecture"
        )
        raise typer.Exit(code=1)
    is_mock = _is_mock_mode(mock)

    from codesense.capabilities.diagram import DiagramHandler
    from codesense.models.output import CommandParams

    project_root = _get_project_root()

    # Determine the target path: explicit path, --file, or project root
    target = path or file or "."

    params = CommandParams(
        path=target,
        query=type,  # diagram type passed via query field
        output=output,
        mock=is_mock,
    )

    handler = DiagramHandler(project_root=project_root)
    result = handler.run(params)
    _render_output(result)


@app.command()
def trace(
    path: str = typer.Argument(..., help="File path to trace history for"),
    line: int = typer.Option(0, "--line", "-l", help="Specific line number"),
    decision: Optional[str] = typer.Option(None, "--decision", help="Decision identifier to trace"),
    mock: bool = typer.Option(False, "--mock", help="Use demo/mock data sources"),
) -> None:
    """Display the timeline of commits, issues, and PRs that led to specified code."""
    _validate_path(path)
    is_mock = _is_mock_mode(mock)

    from codesense.capabilities.trace import TraceHandler
    from codesense.models.output import CommandParams

    params = CommandParams(
        path=path,
        line_number=line if line > 0 else None,
        mock=is_mock,
    )

    handler = TraceHandler(mock=is_mock)
    result = handler.run(params)
    _render_output(result)


@app.command()
def deps(
    path: Optional[str] = typer.Argument(None, help="Module path to analyze dependencies"),
    type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Dependency type: env, api, packages, or all"
    ),
    mock: bool = typer.Option(False, "--mock", help="Use demo/mock data sources"),
) -> None:
    """Display external packages, environment variables, APIs, and internal module dependencies."""
    if path is not None:
        _validate_path(path)
    if type is not None and type not in ("env", "api", "packages", "all"):
        console.print(
            f"[bold red]Error:[/bold red] Invalid dependency type '{type}'. "
            "Must be one of: env, api, packages, all"
        )
        raise typer.Exit(code=1)
    is_mock = _is_mock_mode(mock)

    from codesense.capabilities.deps import DepsHandler
    from codesense.models.output import CommandParams

    project_root = _get_project_root()

    params = CommandParams(
        path=path,
        mock=is_mock,
    )

    handler = DepsHandler(project_root=project_root)
    result = handler.run(params)
    _render_output(result)


@app.command()
def related(
    path: str = typer.Argument(..., help="Code path to find related files for"),
    mock: bool = typer.Option(False, "--mock", help="Use demo/mock data sources"),
) -> None:
    """Display dependents and dependencies with impact analysis."""
    _validate_path(path)
    is_mock = _is_mock_mode(mock)

    from codesense.capabilities.related import RelatedHandler
    from codesense.models.output import CommandParams

    project_root = _get_project_root()

    params = CommandParams(
        path=path,
        mock=is_mock,
    )

    handler = RelatedHandler(project_root=project_root, mock=is_mock)
    result = handler.run(params)
    _render_output(result)


@app.command()
def risk(
    path: str = typer.Argument(..., help="Code path to assess risk for"),
    function: Optional[str] = typer.Option(None, "--function", "-f", help="Specific function name"),
    mock: bool = typer.Option(False, "--mock", help="Use demo/mock data sources"),
) -> None:
    """Assess and display a risk score (0-10) with signal breakdown."""
    _validate_path(path)
    is_mock = _is_mock_mode(mock)

    from codesense.capabilities.risk import RiskHandler
    from codesense.models.output import CommandParams

    project_root = _get_project_root()

    params = CommandParams(
        path=path,
        function_name=function,
        mock=is_mock,
    )

    handler = RiskHandler(project_root=project_root)
    result = handler.run(params)
    _render_output(result)


@app.command()
def onboard(
    module: Optional[str] = typer.Option(None, "--module", "-m", help="Specific module to onboard"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path for the document"),
    mock: bool = typer.Option(False, "--mock", help="Use demo/mock data sources"),
) -> None:
    """Generate a complete onboarding markdown document for the project or a module."""
    if module is not None:
        _validate_path(module)
    is_mock = _is_mock_mode(mock)

    from codesense.capabilities.onboard import OnboardHandler
    from codesense.models.output import CommandParams

    project_root = _get_project_root()

    params = CommandParams(
        path=module,
        output=output,
        mock=is_mock,
    )

    handler = OnboardHandler(project_root=project_root)
    result = handler.run(params)
    _render_output(result)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Natural language question about the codebase"),
    mock: bool = typer.Option(False, "--mock", help="Use demo/mock data sources"),
) -> None:
    """Ask a natural language question — intent is classified and routed to the appropriate handler."""
    is_mock = _is_mock_mode(mock)

    from codesense.capabilities.ask import IntentClassifier
    from codesense.models.output import CommandParams

    gemini_service = _create_gemini_service()
    if gemini_service is None:
        console.print(
            "[bold red]Error:[/bold red] The 'ask' command requires a configured "
            "Gemini API key for intent classification.\n"
            "Set GEMINI_KEY_1 or GEMINI_API_KEYS environment variable."
        )
        raise typer.Exit(code=1)

    project_root = _get_project_root()
    classifier = IntentClassifier(gemini_service=gemini_service)
    classification = classifier.classify(question)

    # Route to the appropriate handler based on classified intent
    intent = classification.get("intent", "general")
    # classify() returns `params` as a CommandParams dataclass (not a dict).
    extracted_params = classification.get("params")

    # Build params from extracted values, falling back to safe defaults.
    params = CommandParams(
        path=getattr(extracted_params, "path", None),
        function_name=getattr(extracted_params, "function_name", None),
        line_number=getattr(extracted_params, "line_number", None),
        query=question,
        mock=is_mock,
    )

    # Route based on intent
    _route_intent(intent, params, project_root, gemini_service, is_mock)


def _route_intent(
    intent: str,
    params,
    project_root: str,
    gemini_service,
    is_mock: bool,
) -> None:
    """Route a classified intent to the appropriate capability handler.

    Args:
        intent: Classified intent name (explain, describe, tree, etc.).
        params: CommandParams with extracted parameters.
        project_root: Project root directory.
        gemini_service: Configured GeminiService instance.
        is_mock: Whether mock mode is active.
    """
    from codesense.models.output import CommandParams

    if intent == "explain":
        from codesense.capabilities.explain import ExplainHandler

        handler = ExplainHandler(gemini_service=gemini_service)
        result = handler.run(params)

    elif intent == "describe":
        from codesense.capabilities.describe import DescribeCapabilityHandler

        handler = DescribeCapabilityHandler(gemini_service=gemini_service)
        result = handler.run(
            params.path or ".",
            function=params.function_name,
            lines=params.line_range,
            mock=is_mock,
        )

    elif intent == "tree":
        from codesense.capabilities.tree import TreeHandler

        handler = TreeHandler()
        result = handler.run(params)

    elif intent == "flow":
        from codesense.capabilities.flow import FlowHandler

        handler = FlowHandler(project_root=project_root)
        result = handler.run(params)

    elif intent == "diagram":
        from codesense.capabilities.diagram import DiagramHandler

        handler = DiagramHandler(project_root=project_root)
        result = handler.run(params)

    elif intent == "trace":
        from codesense.capabilities.trace import TraceHandler

        handler = TraceHandler(mock=is_mock)
        result = handler.run(params)

    elif intent == "deps":
        from codesense.capabilities.deps import DepsHandler

        handler = DepsHandler(project_root=project_root)
        result = handler.run(params)

    elif intent == "related":
        from codesense.capabilities.related import RelatedHandler

        handler = RelatedHandler(project_root=project_root, mock=is_mock)
        result = handler.run(params)

    elif intent == "risk":
        from codesense.capabilities.risk import RiskHandler

        handler = RiskHandler(project_root=project_root)
        result = handler.run(params)

    elif intent == "onboard":
        from codesense.capabilities.onboard import OnboardHandler

        handler = OnboardHandler(project_root=project_root)
        result = handler.run(params)

    else:
        # General reasoning — route to explain with the full question
        from codesense.capabilities.explain import ExplainHandler

        handler = ExplainHandler(gemini_service=gemini_service)
        result = handler.run(params)

    _render_output(result)


@app.command()
def ingest(
    folder: str = typer.Argument(..., help="Folder path containing documents to ingest"),
    mock: bool = typer.Option(False, "--mock", help="Use demo/mock data sources"),
) -> None:
    """Ingest documents from a folder into Decision Memory for RAG retrieval."""
    _validate_path(folder)
    is_mock = _is_mock_mode(mock)

    if is_mock:
        from codesense.output.formatter import RichFormatter
        from codesense.models.output import CommandOutput

        result = CommandOutput(
            title="📥 Document Ingestion",
            content="[DEMO MODE] Ingestion skipped in demo mode.",
            is_demo_mode=True,
        )
        _render_output(result)
        return

    from codesense.memory.ingest import IngestPipeline
    from codesense.models.output import CommandOutput

    pipeline = IngestPipeline()
    results = pipeline.ingest_folder(folder)

    # Summarize results
    total = len(results)
    success_count = sum(1 for r in results if r.success)
    fail_count = total - success_count
    total_chunks = sum(r.chunks_created for r in results)

    content_parts = [
        f"**Processed:** {total} document(s)",
        f"**Succeeded:** {success_count}",
        f"**Failed:** {fail_count}",
        f"**Total chunks created:** {total_chunks}",
    ]

    if fail_count > 0:
        content_parts.append("\n**Errors:**")
        for r in results:
            if not r.success:
                content_parts.append(f"- `{r.document_id}`: {r.error}")

    result = CommandOutput(
        title="📥 Document Ingestion",
        content="\n".join(content_parts),
        is_demo_mode=is_mock,
    )
    _render_output(result)


if __name__ == "__main__":
    app()
