"""Generate PDF from the CodeSense project report markdown.

Usage: py docs/generate_pdf.py

This creates docs/CodeSense_Project_Report.html which you can open
in any browser and print to PDF (Ctrl+P → Save as PDF).
"""

import re
from pathlib import Path


def markdown_to_html(md_content: str) -> str:
    """Simple markdown to HTML converter for the report."""
    html = md_content

    # Escape HTML entities (but not our markdown)
    # Skip this to keep things simple - markdown chars won't conflict

    # Headers
    html = re.sub(r'^#{3}\s+(.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^#{2}\s+(.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^#{1}\s+(.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

    # Horizontal rules
    html = re.sub(r'^---+$', '<hr>', html, flags=re.MULTILINE)

    # Bold
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)

    # Inline code
    html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)

    # Code blocks
    html = re.sub(
        r'```(\w*)\n(.*?)```',
        r'<pre><code class="\1">\2</code></pre>',
        html,
        flags=re.DOTALL
    )

    # Tables
    lines = html.split('\n')
    in_table = False
    new_lines = []
    for line in lines:
        if '|' in line and line.strip().startswith('|'):
            if not in_table:
                new_lines.append('<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%;">')
                in_table = True
            # Skip separator rows
            if re.match(r'^\|[\s\-:|]+\|$', line.strip()):
                continue
            cells = [c.strip() for c in line.strip().split('|')[1:-1]]
            row = '<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>'
            new_lines.append(row)
        else:
            if in_table:
                new_lines.append('</table>')
                in_table = False
            new_lines.append(line)
    if in_table:
        new_lines.append('</table>')
    html = '\n'.join(new_lines)

    # Lists
    html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'(<li>.*?</li>\n?)+', lambda m: '<ul>' + m.group() + '</ul>', html)

    # Paragraphs (lines not already tagged)
    paragraphs = []
    for line in html.split('\n'):
        stripped = line.strip()
        if stripped and not stripped.startswith('<') and not stripped.startswith('|'):
            paragraphs.append(f'<p>{line}</p>')
        else:
            paragraphs.append(line)
    html = '\n'.join(paragraphs)

    return html


def main():
    report_path = Path(__file__).parent / "CodeSense_Project_Report.md"
    output_path = Path(__file__).parent / "CodeSense_Project_Report.html"

    md_content = report_path.read_text(encoding="utf-8")
    body_html = markdown_to_html(md_content)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CodeSense — Project Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
            line-height: 1.6;
            color: #333;
        }}
        h1 {{
            color: #1a5276;
            border-bottom: 3px solid #2980b9;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #2c3e50;
            margin-top: 30px;
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5px;
        }}
        h3 {{
            color: #34495e;
        }}
        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 0.9em;
        }}
        pre {{
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-size: 0.85em;
        }}
        pre code {{
            background: none;
            color: inherit;
            padding: 0;
        }}
        table {{
            margin: 15px 0;
            font-size: 0.9em;
        }}
        table td:first-child {{
            font-weight: bold;
            white-space: nowrap;
        }}
        li {{
            margin: 5px 0;
        }}
        hr {{
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 30px 0;
        }}
        strong {{
            color: #2c3e50;
        }}
        @media print {{
            body {{
                max-width: none;
                padding: 20px;
            }}
            pre {{
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
        }}
    </style>
</head>
<body>
{body_html}
</body>
</html>"""

    output_path.write_text(full_html, encoding="utf-8")
    print(f"HTML report generated: {output_path}")
    print(f"Open in browser and press Ctrl+P → 'Save as PDF'")


if __name__ == "__main__":
    main()
