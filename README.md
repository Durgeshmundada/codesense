<div align="center">

# 🧭 CodeSense

### Understand *why* your code exists — not just *what* it does.

**CodeSense** is a multi-agent codebase intelligence tool that fuses static analysis,
version-control history, GitHub discussions, and architecture documents with an LLM
reasoning loop to explain the intent behind code — with grounded, source-cited answers,
confidence scores, and honest surfacing of contradictions.

<br/>

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Agent-LangGraph-1C3C3C)](https://github.com/langchain-ai/langgraph)
[![Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![Streamlit](https://img.shields.io/badge/Web-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-pytest%20%2B%20Hypothesis-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Command-Line Interface](#command-line-interface)
  - [Web Interface](#web-interface)
- [Commands Reference](#commands-reference)
- [Demo Mode](#demo-mode)
- [Testing](#testing)
- [Technology Stack](#technology-stack)
- [License](#license)

---

## Overview

When developers join a new team or open an unfamiliar codebase, reading the code only
reveals *what* it does mechanically. The harder questions — *why was this written this
way, is it safe to change, what breaks if I touch it* — usually require weeks of code
archaeology: digging through commits, issues, pull requests, and tribal knowledge.

CodeSense automates that archaeology. It reads git history, GitHub issues and PR
discussions, and architecture decision records, then uses a Google Gemini reasoning
pipeline to produce clear, evidence-backed explanations in seconds.

## Key Features

- **Intent-first explanations** — answers *why* code exists with a confidence score and
  cited sources.
- **Twelve focused capabilities** — from structure trees and execution flows to risk
  scoring and onboarding guides (see [Commands Reference](#commands-reference)).
- **Multi-agent reasoning loop** — a five-node LangGraph pipeline that explores, forms
  hypotheses, verifies them, checks for contradictions, and synthesizes an answer.
- **Honest about conflicts** — when sources disagree, CodeSense presents both sides
  rather than silently picking a winner.
- **Decision Memory (RAG)** — ingests architecture docs into a ChromaDB vector store for
  retrieval-augmented answers.
- **Two front-ends** — a Rich-powered CLI and an interactive Streamlit web dashboard.
- **Demo mode** — explore every capability with built-in sample data, no API keys
  required.
- **Resilient API usage** — automatic rotation across multiple Gemini keys with
  rate-limit backoff.

## How It Works

CodeSense reasons about code in five stages, looping when confidence is low or
contradictions appear (bounded to a maximum of three iterations so it never runs
indefinitely):

1. **Explore** — gather evidence from git commits, GitHub issues/PRs, and decision docs.
2. **Hypothesize** — the LLM proposes one to five candidate explanations.
3. **Verify** — each hypothesis is checked against supporting and contradicting evidence,
   producing a confidence score.
4. **Check contradictions** — conflicting claims across sources are surfaced explicitly.
5. **Synthesize** — a grounded, plain-English answer is assembled with citations,
   confidence, and any unresolved conflicts.

## Architecture

```
                ┌─────────────────────────────────────────────┐
   User ──────▶ │  Front-ends:  CLI (Typer/Rich) │ Web (Streamlit) │
                └───────────────────────┬─────────────────────┘
                                        │
                                        ▼
                           ┌────────────────────────┐
                           │   Capabilities layer    │  explain, describe, tree,
                           │   (12 command handlers)  │  flow, diagram, trace, deps,
                           └────────────┬─────────────┘  related, risk, onboard, ask,
                                        │                ingest
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
          ┌──────────────┐   ┌──────────────────┐  ┌────────────────┐
          │  Agent loop   │   │  Static analysis  │  │ Decision Memory │
          │  (LangGraph)  │   │  (AST, imports,   │  │ (ChromaDB RAG)  │
          │  5-node graph │   │   call graph)     │  │                 │
          └──────┬────────┘   └──────────────────┘  └────────────────┘
                 │
                 ▼
        ┌──────────────────┐        ┌───────────────────────────┐
        │  LLM services     │        │  Data sources (MCP)        │
        │  (Gemini + key    │        │  git history, GitHub API,  │
        │   rotation)       │        │  mock/demo source          │
        └──────────────────┘        └───────────────────────────┘
```

## Project Structure

```
codesence/
├── codesense/                 # Main application package
│   ├── agent/                 # LangGraph reasoning loop (graph, nodes, router)
│   ├── analysis/              # Static analysis: AST walker, call graph, imports
│   ├── capabilities/          # The 12 command handlers
│   ├── llm/                   # Gemini service + multi-key rotation manager
│   ├── mcp_server/            # FastMCP server + git data source
│   ├── memory/                # Decision Memory: chunker, embedder, vector store, ingest
│   ├── models/                # Pydantic/dataclass models (state, output, analysis, ...)
│   ├── output/                # Formatters: Rich, Markdown, Mermaid, tree
│   ├── sources/               # GitHub and mock data sources
│   ├── stubs/                 # Type stubs
│   ├── interfaces.py          # Shared protocol/interface definitions
│   ├── main.py                # Typer CLI entry point (all commands)
│   └── __main__.py            # `python -m codesense` entry point
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   ├── property/              # Property-based (Hypothesis) tests
│   └── fixtures/              # Shared test fixtures
├── docs/                      # Project report and PDF generator
├── .streamlit/                # Streamlit UI configuration
├── streamlit_app.py           # Web dashboard entry point
├── pyproject.toml             # Package metadata, dependencies, entry points
├── requirements.txt           # Pinned dependency versions
├── .env.template              # Environment variable template
└── README.md
```

## Requirements

- **Python 3.11+**
- A **Google Gemini API key** for live reasoning (free tier works). Get one from
  [Google AI Studio](https://makersuite.google.com/app/apikey). *Not required for demo mode.*
- A **GitHub personal access token** for issue/PR retrieval on private or rate-limited
  repositories (optional).

## Installation

Clone the repository and install the package in editable mode:

```bash
py -m pip install -e .
```

To include the web UI and development dependencies:

```bash
py -m pip install -e ".[web,dev]"
```

Alternatively, install pinned versions directly:

```bash
py -m pip install -r requirements.txt
```

## Configuration

Copy the environment template and fill in your values:

```bash
copy .env.template .env
```

| Variable            | Required | Description                                                        |
|---------------------|----------|--------------------------------------------------------------------|
| `GEMINI_KEY_1`      | Yes*     | Primary Gemini API key. Add `GEMINI_KEY_2`, `GEMINI_KEY_3`, … for automatic rotation. |
| `GITHUB_TOKEN`      | No       | GitHub personal access token for issue/PR retrieval.              |
| `CODESENSE_MOCK`    | No       | Set to `true` to force demo mode (no credentials needed).         |
| `BACKOFF_MINUTES`   | No       | Minutes to wait before retrying a rate-limited key (default `1`). |

\* Not required when running in [demo mode](#demo-mode).

## Usage

### Command-Line Interface

Run any command via the installed `codesense` script or the module form:

```bash
codesense <command> [arguments] [options]
# or
py -m codesense <command> [arguments] [options]
```

Examples:

```bash
# Explain why a file exists (full reasoning loop)
codesense explain codesense/agent/graph.py

# Describe what a function does
codesense describe codesense/main.py --function explain

# Show the annotated project tree
codesense tree . --depth 3

# Assess how risky a file is to change
codesense risk codesense/llm/key_manager.py

# Ask a natural-language question
codesense ask "Is it safe to delete the retry logic in the LLM service?"
```

### Web Interface

Launch the Streamlit dashboard, which exposes all capabilities interactively:

```bash
py -m streamlit run streamlit_app.py
```

The web UI supports loading a repository from a GitHub URL or a local folder, entering
Gemini keys at runtime, toggling demo mode, and rendering Mermaid diagrams natively.

## Commands Reference

| Command    | Answers                          | Description                                                            |
|------------|----------------------------------|------------------------------------------------------------------------|
| `explain`  | Why does this code exist?        | Runs the full reasoning loop with confidence and source citations.     |
| `describe` | What does this code do?          | Plain-English summary from code alone (no git/GitHub needed).          |
| `tree`     | How is the project organized?    | Annotated directory structure with per-file descriptions.              |
| `flow`     | How does execution proceed?      | Numbered call sequence from an entry point, with a sequence diagram.   |
| `diagram`  | What are the relationships?      | Mermaid flowchart, sequence, or architecture diagram.                  |
| `trace`    | What is the history?             | Timeline of commits, issues, and PRs that shaped a file.               |
| `deps`     | What does this depend on?        | External packages, environment variables, APIs, and internal imports. |
| `related`  | What is connected to this?       | Dependents and dependencies with impact analysis.                      |
| `risk`     | Is it safe to change?            | Risk score (0–10) with a breakdown of contributing signals.            |
| `onboard`  | How do I get started?            | Generates a full onboarding document for the project or a module.      |
| `ask`      | *(anything, in plain English)*   | Classifies intent and routes to the appropriate capability.            |
| `ingest`   | —                                | Imports documents into Decision Memory for RAG retrieval.              |

Every command accepts `--mock` to run against built-in demo data.

## Demo Mode

Demo mode lets you explore CodeSense without any API keys or credentials:

```bash
codesense tree . --mock
codesense risk codesense/main.py --mock
codesense deps codesense/ --mock
codesense related codesense/main.py --mock
codesense onboard --mock
```

You can also enable it globally by setting `CODESENSE_MOCK=true`, or via the Demo Mode
toggle in the web UI.

## Testing

The test suite uses `pytest` with `pytest-asyncio` and property-based tests via
`hypothesis`:

```bash
py -m pytest
```

Tests are organized into `unit/`, `integration/`, and `property/` suites, with shared
fixtures in `tests/fixtures/`.

## Technology Stack

| Concern            | Technology                                  |
|--------------------|---------------------------------------------|
| Reasoning agent    | LangGraph `StateGraph` (5-node loop)        |
| LLM                | Google Gemini via LangChain                 |
| Data retrieval     | FastMCP server, GitPython, PyGithub         |
| Decision Memory    | ChromaDB + Sentence-Transformers embeddings |
| CLI                | Typer + Rich                                |
| Web UI             | Streamlit + pandas                          |
| Diagrams           | Mermaid                                     |
| Testing            | pytest, pytest-asyncio, Hypothesis          |

## License

Released under the [MIT License](LICENSE). You are free to use, modify, and distribute
this software with attribution.

---

<div align="center">
<sub>Built with LangGraph, Google Gemini, and a healthy respect for code archaeology.</sub>
</div>
