# CodeSense — Project Report

## A Simple Guide to Understanding CodeSense

---

## 1. What is CodeSense?

CodeSense is a **Python command-line tool** (CLI) that helps developers understand code.

Think of it like a **smart assistant for your codebase**. When you join a new team or open someone else's code, you always ask:

- "Why was this code written?"
- "What does this file do?"
- "Is it safe to change this?"
- "How does this project work?"

**CodeSense answers all these questions automatically.**

It looks at git history, GitHub issues, pull request discussions, and architecture documents — then gives you a plain-English answer with confidence scores and sources.

---

## 2. What Problem Does It Solve?

### The Problem

When developers join a new project, they face a **knowledge crisis**:

- They can read the code and understand WHAT it does mechanically
- But they have NO idea WHY it was written that way
- They don't know if some "ugly hack" is actually keeping the system alive
- They spend 3-6 months just doing "archaeology" — reading old commits, asking colleagues

### The Solution

CodeSense does this archaeology **automatically** in seconds:

- It reads git commit history
- It reads GitHub issues and PR discussions
- It reads architecture decision documents
- It uses AI (Google Gemini) to reason about all this evidence
- It gives you a clear answer: "This code exists because..."

---

## 3. What Can CodeSense Do? (Features)

CodeSense has **12 commands** that answer different questions:

| Command | What It Does | Example |
|---------|-------------|---------|
| `explain` | Tells you WHY code exists | "Why does auth.py exist?" |
| `describe` | Tells you WHAT code does | "What does this function do?" |
| `tree` | Shows project structure | Like a file explorer with descriptions |
| `flow` | Shows how code runs step by step | "main.py calls auth.py calls db.py" |
| `diagram` | Draws visual diagrams | Creates Mermaid charts |
| `trace` | Shows history timeline | "This was added on Jan 15 because of bug #42" |
| `deps` | Shows dependencies | "This needs: requests, redis, 3 env vars" |
| `related` | Shows connected files | "If you change this, these 4 files break" |
| `risk` | Rates how dangerous code is | "Risk: 7/10 — no tests, original author left" |
| `onboard` | Creates a starter guide | Full document for new team members |
| `ask` | Natural language questions | "Is it safe to delete the retry logic?" |
| `ingest` | Imports documents | Feeds architecture docs into memory |

---

## 4. How Does It Work? (Simple Explanation)

CodeSense works in **5 steps** (like a detective investigating a case):

### Step 1: EXPLORE (Gather Clues)
- Reads git commit messages
- Reads GitHub issues and PR comments
- Searches architecture documents
- Collects all "evidence" about the code

### Step 2: HYPOTHESIZE (Form a Theory)
- AI looks at all the evidence
- Creates 1-5 possible explanations
- Example: "This code probably exists because of a race condition bug"

### Step 3: VERIFY (Check the Theory)
- Searches for more supporting or contradicting evidence
- Gives a confidence score (0% to 100%)

### Step 4: CHECK CONTRADICTIONS (Find Conflicts)
- Sometimes sources disagree!
- CodeSense honestly shows both sides
- It does NOT pick a winner — you decide

### Step 5: SYNTHESIZE (Give Final Answer)
- Combines everything into a clear English explanation
- Shows confidence score
- Lists all sources used
- Shows any unresolved conflicts

**If confidence is low → goes back to Step 1 to gather more evidence**
**If contradictions found → goes back to Step 2 to re-think**
**Always stops after 3 loops maximum (never runs forever)**

---

## 5. Technology Used

| What | Technology | Why |
|------|-----------|-----|
| AI Brain | Google Gemini (via LangChain) | Free tier, good at reasoning |
| Agent Loop | LangGraph StateGraph | Manages the 5-step thinking process |
| Data Tools | FastMCP Server | Retrieves git/GitHub data |
| Memory | ChromaDB + HuggingFace | Remembers architecture decisions |
| Git Access | GitPython | Reads commit history |
| GitHub Access | PyGithub | Reads issues and PR comments |
| CLI | Typer + Rich | Beautiful terminal output |
| Diagrams | Mermaid | Text-based diagrams that render on GitHub |
| Testing | Hypothesis (PBT) | Property-based testing for correctness |

---

## 6. How to Install and Use

### Step 1: Install

```bash
py -m pip install -e .
```

### Step 2: Set Up API Keys

Copy `.env.template` to `.env` and add your Gemini API key:

```
GEMINI_KEY_1=your_gemini_api_key_here
```

Get a free key from: https://makersuite.google.com/app/apikey

### Step 3: Try It (Demo Mode — No Keys Needed!)

```bash
py -m codesense tree . --mock
py -m codesense risk codesense/main.py --mock
py -m codesense deps codesense/ --mock
py -m codesense related codesense/main.py --mock
py -m codesense onboard --mock
```

The `--mock` flag uses built-in demo data so you can try everything without API keys.

### Step 4: Use for Real (Needs Gemini Key)

```bash
py -m codesense explain src/auth.py
py -m codesense ask "why does the retry logic exist?"
py -m codesense diagram codesense/ --type architecture
```

---

## 7. Project Structure (Folder Map)

```
codesense/
├── main.py              ← CLI entry point (all 12 commands)
├── agent/               ← The AI reasoning brain
│   ├── graph.py         ← LangGraph 5-step loop
│   ├── nodes.py         ← Each thinking step
│   ├── router.py        ← Decides: loop back or finish?
│   └── check_contradictions.py
├── capabilities/        ← One file per command
│   ├── explain.py       ← "Why does this exist?"
│   ├── describe.py      ← "What does this do?"
│   ├── tree.py          ← Project structure
│   ├── flow.py          ← Execution tracing
│   ├── diagram.py       ← Mermaid diagrams
│   ├── trace.py         ← History timeline
│   ├── deps.py          ← Dependencies
│   ├── related.py       ← Impact analysis
│   ├── risk.py          ← Risk scoring
│   ├── onboard.py       ← Onboarding guide
│   └── ask.py           ← Natural language routing
├── mcp_server/          ← Data retrieval tools
│   ├── server.py        ← FastMCP with 4 tools
│   └── git_source.py    ← Git history access
├── sources/             ← Data sources
│   ├── mock_source.py   ← Demo data (no keys needed)
│   └── github_source.py ← Live GitHub API
├── memory/              ← Decision Memory (RAG)
│   ├── chunker.py       ← Splits docs by decision
│   ├── embedder.py      ← HuggingFace embeddings
│   ├── vector_store.py  ← ChromaDB storage
│   └── ingest.py        ← Document ingestion
├── llm/                 ← AI service
│   ├── key_manager.py   ← Multi-key rotation
│   └── gemini_service.py← Gemini API wrapper
├── analysis/            ← Code analysis
│   ├── ast_walker.py    ← Python AST parsing
│   ├── call_graph.py    ← Execution path tracing
│   └── import_scanner.py← Dependency scanning
├── output/              ← Output formatting
│   ├── formatter.py     ← Rich terminal output
│   ├── mermaid_formatter.py ← Diagram generation
│   ├── tree_formatter.py    ← Tree display
│   └── markdown_writer.py   ← Document writing
└── models/              ← Data structures
    ├── state.py         ← Agent state
    ├── mcp.py           ← Git/GitHub records
    ├── memory.py        ← Decision units
    ├── analysis.py      ← Code analysis models
    ├── output.py        ← CLI output models
    └── llm.py           ← Key rotation models
```

---

## 8. Key Design Decisions

1. **Conflicts have NO winner** — When sources disagree, CodeSense shows both sides equally. It never picks a winner or hides contradictions.

2. **Decision-unit chunking** — Architecture documents are split by "decision" (not by word count). Each chunk has a complete rationale.

3. **Multi-key rotation** — Uses multiple free Gemini API keys with automatic switching when one hits rate limits.

4. **Mock sources** — Everything works without credentials using built-in demo data. Great for evaluation.

5. **Always terminates** — The reasoning loop has a hard limit of 3 cycles. It can never run forever.

---

## 9. Who Is This For?

- **New developers** joining a team — understand the codebase in hours, not months
- **Senior developers** reviewing unfamiliar code — know what's safe to change
- **Team leads** — generate onboarding docs automatically
- **Anyone** who opens code and asks "but WHY?"

---

## 10. Example Output

```
🔍 CodeSense — Codebase Archaeology Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY THIS EXISTS:
The process_retry function was introduced in commit a4f2b1c (Nov 2022)
to handle the race condition discovered during the Diwali sale traffic
spike. The original fix (PR #341, reverted) broke payment idempotency.
This version is a known compromise.

CONFIDENCE: 83%

SOURCES:
  • git:a4f2b1c — "fix: add retry logic for payment race condition"
  • PR #341 — "Revert: breaks idempotency for concurrent requests"
  • Issue #892 — "Technical debt: retry count hardcoded, revisit Q1"

UNRESOLVED CONFLICTS: None
```

---

## 11. Future Plans

- VS Code extension (see code explanations inline)
- Slack bot (`/why src/auth.py` in Slack)
- Support for more languages (JavaScript, Java, Go)
- Team knowledge graph

---

## 12. Summary

| Question | Answer |
|----------|--------|
| What is it? | A Python CLI tool for understanding code |
| What does it do? | Answers "WHY does this code exist?" |
| How does it work? | AI + git history + GitHub + RAG memory |
| Who needs it? | Any developer joining a new codebase |
| What makes it special? | Grounded answers with citations, never hallucinated |
| How to try it? | `py -m codesense tree . --mock` |

---

*Built with: Python 3.11+, LangGraph, LangChain, Google Gemini, FastMCP, ChromaDB, GitPython, PyGithub, Typer, Rich*

*Project by: B.Tech IT Final Year Student*
*Courses applied: LangGraph, MCP, LangChain (CampusX playlists)*
