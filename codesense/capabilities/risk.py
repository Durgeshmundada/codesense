"""Risk assessment capability handler.

Computes a risk score (0-10) with breakdown of contributing signals:
- author_turnover: how many distinct authors, are early authors still active?
- staleness: how long since anyone touched this code
- dependency_count: how many other files depend on this
- test_coverage: does it have tests? (check for test file existence)
- hack_markers: count of "TODO", "HACK", "FIXME", "temporary", "workaround" in code/comments

Requirements: 5.9
"""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from codesense.models.analysis import RiskAssessment
from codesense.models.output import CommandOutput, CommandParams, TableData

logger = logging.getLogger(__name__)

# Markers that indicate code-level risk/technical debt
_HACK_MARKERS = ["TODO", "HACK", "FIXME", "temporary", "workaround"]

# Maximum individual signal contribution (each signal contributes up to 2.0)
_MAX_SIGNAL_SCORE = 2.0

# Number of signals used in computation
_NUM_SIGNALS = 5


def _clamp(value: float, min_val: float = 0.0, max_val: float = 10.0) -> float:
    """Clamp a value to the specified range."""
    return max(min_val, min(max_val, value))


def _compute_author_turnover(commits: list) -> float:
    """Compute author turnover signal (0.0 - 2.0).

    Higher score means more risk from author churn.
    Factors:
    - Many distinct authors = lower risk (knowledge spread) → inverted
    - Early authors no longer active = higher risk (knowledge loss)

    Actually, high turnover with early authors gone = high risk.
    """
    if not commits:
        return 1.0  # No data = moderate risk

    authors = [c.author for c in commits]
    unique_authors = set(authors)
    total_authors = len(unique_authors)

    if total_authors <= 1:
        # Single author: moderate risk (bus factor = 1)
        return 1.0

    # Check if early authors (first half of history) are still active (last quarter)
    midpoint = len(commits) // 2
    early_authors = set(c.author for c in commits[midpoint:])  # older commits
    recent_authors = set(c.author for c in commits[: max(1, len(commits) // 4)])

    # What fraction of early authors are no longer active?
    if early_authors:
        departed_ratio = len(early_authors - recent_authors) / len(early_authors)
    else:
        departed_ratio = 0.0

    # Score: high departed ratio + single recent author = high risk
    # Scale 0.0 to 2.0
    score = departed_ratio * _MAX_SIGNAL_SCORE
    return _clamp(score, 0.0, _MAX_SIGNAL_SCORE)


def _compute_staleness(commits: list) -> float:
    """Compute staleness signal (0.0 - 2.0).

    Higher score means code hasn't been touched in a long time.
    """
    if not commits:
        return _MAX_SIGNAL_SCORE  # No commits = maximum staleness

    # Find most recent commit timestamp
    most_recent = None
    for commit in commits:
        try:
            ts = datetime.fromisoformat(commit.timestamp.replace("Z", "+00:00"))
            if most_recent is None or ts > most_recent:
                most_recent = ts
        except (ValueError, AttributeError):
            continue

    if most_recent is None:
        return _MAX_SIGNAL_SCORE

    now = datetime.now(timezone.utc)
    days_since = (now - most_recent).days

    # Scoring: 0 days = 0.0, 365+ days = 2.0 (linear scale)
    score = min(days_since / 365.0, 1.0) * _MAX_SIGNAL_SCORE
    return _clamp(score, 0.0, _MAX_SIGNAL_SCORE)


def _compute_dependency_count(path: str, project_root: str) -> float:
    """Compute dependency count signal (0.0 - 2.0).

    Uses ImportScanner to count how many other files import this module.
    Higher dependency count = higher risk (more impact if changed).
    """
    try:
        from codesense.analysis.import_scanner import ImportScanner

        scanner = ImportScanner(project_root=project_root)

        # Scan the project for files that import this module
        target_path = Path(path).resolve()
        project_path = Path(project_root).resolve()

        # Get the module name from the file path
        try:
            relative = target_path.relative_to(project_path)
            # Convert path to module name (e.g., codesense/main.py -> codesense.main)
            module_parts = list(relative.with_suffix("").parts)
            module_name = ".".join(module_parts)
        except ValueError:
            module_name = target_path.stem

        # Count files that import this module by scanning all Python files
        dependents_count = 0
        for py_file in project_path.rglob("*.py"):
            if py_file == target_path:
                continue
            try:
                graph = scanner.scan_file(str(py_file))
                # Check if any internal deps reference our module
                for dep in graph.internal_deps:
                    if module_name in dep or target_path.stem in dep:
                        dependents_count += 1
                        break
            except Exception:
                continue

    except Exception as e:
        logger.debug("Could not compute dependency count: %s", e)
        return 1.0  # Default moderate risk on failure

    # Scoring: 0 dependents = 0.0, 10+ dependents = 2.0
    score = min(dependents_count / 10.0, 1.0) * _MAX_SIGNAL_SCORE
    return _clamp(score, 0.0, _MAX_SIGNAL_SCORE)


def _compute_test_coverage(path: str, project_root: str) -> float:
    """Compute test coverage signal (0.0 - 2.0).

    Checks if a corresponding test file exists for this code file.
    No tests = higher risk.
    """
    target = Path(path)
    project = Path(project_root)

    # Generate possible test file names
    stem = target.stem
    possible_test_names = [
        f"test_{stem}.py",
        f"{stem}_test.py",
        f"test_{stem}s.py",
    ]

    # Search in common test directories
    test_dirs = [
        project / "tests",
        project / "tests" / "unit",
        project / "tests" / "integration",
        project / "test",
        target.parent / "tests",
        target.parent,
    ]

    for test_dir in test_dirs:
        if not test_dir.exists():
            continue
        for test_name in possible_test_names:
            if (test_dir / test_name).exists():
                return 0.0  # Test exists = no risk from this signal

    # Also check recursively
    for test_file in project.rglob(f"test_{stem}.py"):
        return 0.0

    for test_file in project.rglob(f"{stem}_test.py"):
        return 0.0

    # No test file found = maximum risk for this signal
    return _MAX_SIGNAL_SCORE


def _compute_hack_markers(path: str) -> float:
    """Compute hack markers signal (0.0 - 2.0).

    Counts occurrences of TODO, HACK, FIXME, "temporary", "workaround"
    in the source code. More markers = higher risk.
    """
    try:
        content = Path(path).read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError):
        return 0.5  # Can't read file = low-moderate risk

    total_markers = 0
    for marker in _HACK_MARKERS:
        # Case-insensitive search
        total_markers += len(re.findall(re.escape(marker), content, re.IGNORECASE))

    # Scoring: 0 markers = 0.0, 5+ markers = 2.0
    score = min(total_markers / 5.0, 1.0) * _MAX_SIGNAL_SCORE
    return _clamp(score, 0.0, _MAX_SIGNAL_SCORE)


def compute_risk_score(
    path: str,
    function: Optional[str] = None,
    mock: bool = False,
    project_root: str = ".",
) -> RiskAssessment:
    """Compute the risk assessment for a code path.

    Gathers signals from git history, dependency analysis, test coverage,
    and code content to produce a composite risk score in [0.0, 10.0].

    Args:
        path: Path to the code file to assess.
        function: Optional specific function name (currently unused in scoring).
        mock: Whether to use mock data sources.
        project_root: Root directory of the project.

    Returns:
        RiskAssessment with score clamped to [0.0, 10.0] and signal breakdown.
    """
    # Get commit history for the file
    commits = _get_commits(path, mock=mock, project_root=project_root)

    # Compute individual signals
    signals: dict[str, float] = {}
    signals["author_turnover"] = _compute_author_turnover(commits)
    signals["staleness"] = _compute_staleness(commits)
    signals["dependency_count"] = _compute_dependency_count(path, project_root)
    signals["test_coverage"] = _compute_test_coverage(path, project_root)
    signals["hack_markers"] = _compute_hack_markers(path)

    # Composite score: sum of all signals (each 0-2, total 0-10)
    raw_score = sum(signals.values())

    # Clamp to [0.0, 10.0] for safety
    final_score = _clamp(raw_score, 0.0, 10.0)

    return RiskAssessment(
        path=path,
        score=final_score,
        signals=signals,
    )


def _get_commits(path: str, mock: bool = False, project_root: str = ".") -> list:
    """Retrieve commit history for the given path using MCP tools.

    Args:
        path: Code file path.
        mock: Whether to use mock data source.
        project_root: Root directory of the project.

    Returns:
        List of CommitRecord objects.
    """
    try:
        if mock:
            from codesense.sources.mock_source import MockSource

            source = MockSource()
            return source.get_commits(path, limit=50)
        else:
            from codesense.mcp_server.git_source import GitSource

            source = GitSource(repo_path=project_root)
            return source.get_commits(path, limit=50)
    except Exception as e:
        logger.warning("Could not retrieve commits for %s: %s", path, e)
        return []


class RiskHandler:
    """Capability handler for the 'risk' command.

    Implements the CapabilityHandler protocol. Computes a composite risk
    score (0-10) based on multiple signals and formats the result.

    Args:
        project_root: Root directory of the project (defaults to ".").
    """

    def __init__(self, project_root: str = ".") -> None:
        self._project_root = project_root

    def run(self, params: CommandParams) -> CommandOutput:
        """Execute the risk assessment capability.

        Args:
            params: Parsed CLI arguments. Must include `path`.

        Returns:
            CommandOutput with title "⚠️ Risk Assessment", content with
            score breakdown, and a table of signal contributions.
        """
        path = params.path or "."
        function = params.function_name
        is_demo = params.mock

        assessment = compute_risk_score(
            path=path,
            function=function,
            mock=is_demo,
            project_root=self._project_root,
        )

        # Build content markdown
        content = self._format_assessment(assessment)

        # Build signal breakdown table
        table = TableData(
            headers=["Signal", "Score", "Max", "Description"],
            rows=self._build_signal_rows(assessment.signals),
            title="Signal Breakdown",
        )

        return CommandOutput(
            title="⚠️ Risk Assessment",
            content=content,
            confidence=assessment.score / 10.0,
            tables=[table],
            is_demo_mode=is_demo,
        )

    def _format_assessment(self, assessment: RiskAssessment) -> str:
        """Format the risk assessment as markdown content."""
        score = assessment.score
        level = self._risk_level(score)

        content = f"**Risk Score: {score:.1f} / 10.0** ({level})\n\n"
        content += f"**File:** `{assessment.path}`\n\n"

        # Summary interpretation
        if score <= 3.0:
            content += "This code has low risk. It is well-maintained and tested.\n"
        elif score <= 6.0:
            content += (
                "This code has moderate risk. Some attention may be needed.\n"
            )
        else:
            content += (
                "This code has high risk. Consider prioritizing maintenance, "
                "adding tests, or reducing complexity.\n"
            )

        return content

    def _risk_level(self, score: float) -> str:
        """Convert numeric score to human-readable risk level."""
        if score <= 2.0:
            return "Low"
        elif score <= 4.0:
            return "Low-Medium"
        elif score <= 6.0:
            return "Medium"
        elif score <= 8.0:
            return "Medium-High"
        else:
            return "High"

    def _build_signal_rows(self, signals: dict[str, float]) -> list[list[str]]:
        """Build table rows from signal scores."""
        descriptions = {
            "author_turnover": "Knowledge concentration and author departure risk",
            "staleness": "Time since last modification",
            "dependency_count": "Number of files depending on this code",
            "test_coverage": "Existence of test files for this code",
            "hack_markers": "Count of TODO/HACK/FIXME/temporary/workaround markers",
        }

        rows = []
        for signal_name, score in sorted(signals.items()):
            desc = descriptions.get(signal_name, "")
            rows.append([signal_name, f"{score:.1f}", "2.0", desc])

        return rows
