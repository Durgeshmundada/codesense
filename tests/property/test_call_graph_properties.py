"""Property-based tests for call graph depth limit enforcement.

Tests Property 25 from the design document using Hypothesis.

Validates: Requirements 12.3, 12.6
"""

import os
import tempfile

from hypothesis import given, settings
from hypothesis import strategies as st

from codesense.analysis.call_graph import CallGraphBuilder
from codesense.models.analysis import CallGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_call_chain_file(directory: str, chain_depth: int) -> str:
    """Create a Python file with a linear chain of function calls.

    Generates functions f_0() -> f_1() -> ... -> f_{chain_depth-1}()
    where each function calls the next, forming a chain of the given depth.

    Args:
        directory: Directory to create the file in.
        chain_depth: Number of functions in the chain (depth of calls).

    Returns:
        Path to the created file.
    """
    lines = []
    for i in range(chain_depth):
        if i < chain_depth - 1:
            lines.append(f"def f_{i}():")
            lines.append(f"    f_{i + 1}()")
            lines.append("")
        else:
            # Leaf function — no calls
            lines.append(f"def f_{i}():")
            lines.append("    pass")
            lines.append("")

    file_path = os.path.join(directory, "chain.py")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return file_path


# ---------------------------------------------------------------------------
# Property 25: Call graph depth limit
# ---------------------------------------------------------------------------


# Feature: codesense, Property 25: Call graph depth limit
@settings(max_examples=50, deadline=None)
@given(
    max_depth=st.integers(min_value=1, max_value=15),
    chain_depth=st.integers(min_value=1, max_value=20),
)
def test_call_graph_depth_never_exceeds_max_depth(
    max_depth: int, chain_depth: int
) -> None:
    """For any max_depth value (1 ≤ max_depth ≤ 15), the CallGraph.depth field
    never exceeds max_depth.

    **Validates: Requirements 12.3**

    Creates a temporary Python file with a call chain of varying depth and
    verifies that the CallGraphBuilder respects the max_depth limit.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = create_call_chain_file(tmp_dir, chain_depth)
        rel_path = os.path.relpath(file_path, tmp_dir)

        builder = CallGraphBuilder(project_root=tmp_dir)
        entry_point = f"{rel_path}::f_0"
        result: CallGraph = builder.build(entry_point, max_depth=max_depth)

        # Property: depth never exceeds max_depth
        assert result.depth <= max_depth, (
            f"CallGraph.depth ({result.depth}) exceeded max_depth ({max_depth}) "
            f"with chain_depth={chain_depth}"
        )


# Feature: codesense, Property 25: Call graph depth limit
@settings(max_examples=50, deadline=None)
@given(
    max_depth=st.integers(min_value=1, max_value=10),
)
def test_truncation_indicated_when_graph_deeper(max_depth: int) -> None:
    """When a file has call chains deeper than max_depth,
    CallGraph.max_depth_reached is True.

    **Validates: Requirements 12.6**

    Creates a temporary Python file with a call chain strictly deeper than
    max_depth and verifies that the CallGraph indicates truncation occurred.
    """
    # Ensure chain is deeper than max_depth so truncation must occur
    chain_depth = max_depth + 5

    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = create_call_chain_file(tmp_dir, chain_depth)
        rel_path = os.path.relpath(file_path, tmp_dir)

        builder = CallGraphBuilder(project_root=tmp_dir)
        entry_point = f"{rel_path}::f_0"
        result: CallGraph = builder.build(entry_point, max_depth=max_depth)

        # Property: truncation is indicated when actual graph is deeper
        assert result.max_depth_reached is True, (
            f"Expected max_depth_reached=True when chain_depth ({chain_depth}) > "
            f"max_depth ({max_depth}), but got max_depth_reached=False. "
            f"Actual depth reported: {result.depth}"
        )
