"""Markdown writer for generating onboarding documents and diagram files."""

import os
from pathlib import Path


class MarkdownWriter:
    """Writes and builds markdown documents for onboarding and diagrams."""

    # Standard sections for onboarding documents, in display order
    ONBOARDING_SECTIONS = [
        "Overview",
        "Project Structure",
        "Execution Flow",
        "Design Decisions",
        "Setup",
        "Safe to Change",
    ]

    def render(self, sections: list[dict]) -> str:
        """Render a list of section dicts into a formatted markdown string.

        Args:
            sections: List of section dicts with keys:
                - "heading" (str): The section heading text
                - "content" (str): The section body content
                - "level" (int): Heading level (1-6)

        Returns:
            Formatted markdown string with proper heading levels and spacing.
        """
        parts: list[str] = []

        for section in sections:
            heading = section.get("heading", "")
            content = section.get("content", "")
            level = section.get("level", 1)

            # Clamp level between 1 and 6
            level = max(1, min(6, level))

            heading_prefix = "#" * level
            parts.append(f"{heading_prefix} {heading}")
            if content:
                parts.append("")
                parts.append(content)
            parts.append("")

        return "\n".join(parts).rstrip() + "\n"

    def write_file(self, content: str, output_path: str) -> None:
        """Write markdown content to a file.

        Creates parent directories if they don't exist.

        Args:
            content: The markdown content string to write.
            output_path: The file path to write to.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def render_onboarding(self, project_info: dict) -> str:
        """Generate a full onboarding document from project information.

        Produces sections: Purpose, Structure, Execution Flow, Design Decisions,
        Setup, and Safe Areas.

        Args:
            project_info: Dictionary with optional keys:
                - "name" (str): Project name
                - "purpose" (str): Project purpose/description
                - "structure" (str): Project structure overview
                - "execution_flow" (str): How the code executes
                - "design_decisions" (list[str]): Key design decisions
                - "setup" (str): Setup instructions
                - "safe_areas" (list[str]): Areas safe to modify

        Returns:
            Complete onboarding markdown document string.
        """
        project_name = project_info.get("name", "Project")

        sections: list[dict] = [
            {
                "heading": f"{project_name} — Onboarding Guide",
                "content": "",
                "level": 1,
            },
        ]

        # Purpose section
        purpose = project_info.get("purpose", "No purpose description provided.")
        sections.append({
            "heading": "Purpose",
            "content": purpose,
            "level": 2,
        })

        # Structure section
        structure = project_info.get("structure", "No structure information available.")
        sections.append({
            "heading": "Structure",
            "content": structure,
            "level": 2,
        })

        # Execution Flow section
        execution_flow = project_info.get(
            "execution_flow", "No execution flow information available."
        )
        sections.append({
            "heading": "Execution Flow",
            "content": execution_flow,
            "level": 2,
        })

        # Design Decisions section
        design_decisions = project_info.get("design_decisions", [])
        if design_decisions:
            decisions_content = "\n".join(
                f"- {decision}" for decision in design_decisions
            )
        else:
            decisions_content = "No design decisions documented."
        sections.append({
            "heading": "Design Decisions",
            "content": decisions_content,
            "level": 2,
        })

        # Setup section
        setup = project_info.get("setup", "No setup instructions provided.")
        sections.append({
            "heading": "Setup",
            "content": setup,
            "level": 2,
        })

        # Safe Areas section
        safe_areas = project_info.get("safe_areas", [])
        if safe_areas:
            safe_content = "\n".join(f"- {area}" for area in safe_areas)
        else:
            safe_content = "No safe-to-change areas identified."
        sections.append({
            "heading": "Safe Areas to Modify",
            "content": safe_content,
            "level": 2,
        })

        return self.render(sections)

    def write(self, content: str, output_path: str) -> None:
        """Write markdown content to a file (alias for write_file).

        Args:
            content: The markdown string to write.
            output_path: File path where the markdown will be saved.

        Raises:
            OSError: If the file cannot be written.
        """
        self.write_file(content, output_path)

    def build_onboarding_doc(self, sections: dict[str, str]) -> str:
        """Build a complete onboarding markdown document from section content.

        Takes a dict of section_name → content and produces a markdown document
        with headers for each section. Sections are ordered according to the
        standard onboarding layout:
        1. Overview
        2. Project Structure
        3. Execution Flow
        4. Design Decisions
        5. Setup
        6. Safe to Change

        Any sections not in the standard list are appended at the end.

        Args:
            sections: Mapping of section name to markdown content for that section.

        Returns:
            Complete markdown document string with H2 headers for each section.
        """
        parts: list[str] = []
        parts.append("# Onboarding Guide\n")

        # Add standard sections in order
        added: set[str] = set()
        for section_name in self.ONBOARDING_SECTIONS:
            if section_name in sections:
                content = sections[section_name].strip()
                parts.append(f"## {section_name}\n")
                parts.append(f"{content}\n")
                added.add(section_name)

        # Add any extra sections not in the standard list
        for section_name, content in sections.items():
            if section_name not in added:
                content = content.strip()
                parts.append(f"## {section_name}\n")
                parts.append(f"{content}\n")

        return "\n".join(parts)

    def build_diagram_doc(self, title: str, mermaid_content: str) -> str:
        """Wrap Mermaid content in a proper markdown file with title and code fence.

        Args:
            title: The title for the diagram document.
            mermaid_content: Raw Mermaid diagram content (without code fences).

        Returns:
            Complete markdown document with title and fenced Mermaid block.
        """
        return f"# {title}\n\n```mermaid\n{mermaid_content.strip()}\n```\n"
