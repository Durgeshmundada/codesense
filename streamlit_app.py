"""CodeSense — Streamlit Web UI

A beautiful web interface for CodeSense that provides all 12 capabilities
through an interactive dashboard.

Run with: py -m streamlit run streamlit_app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st

# Ensure codesense package is importable
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="CodeSense — Codebase Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Sidebar ──────────────────────────────────────────────────────────────────


st.sidebar.image("https://img.icons8.com/fluency/96/source-code.png", width=80)
st.sidebar.title("🔍 CodeSense")
st.sidebar.markdown("*Codebase Intelligence Tool*")
st.sidebar.markdown("---")

# Settings
st.sidebar.subheader("⚙️ Settings")
mock_mode = st.sidebar.checkbox("🎭 Demo Mode (no API keys needed)", value=False)

# ─── Repository Source ────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("📂 Repository Source")

repo_source = st.sidebar.radio(
    "Choose source:",
    ["🌐 GitHub URL", "💻 Local Folder"],
    index=0,
)

clone_dir = Path.home() / ".codesense_repos"
project_root = str(Path.cwd())

if repo_source == "🌐 GitHub URL":
    repo_url = st.sidebar.text_input("GitHub URL:", value="", placeholder="https://github.com/user/repo")
    
    github_token = st.sidebar.text_input("🔑 GitHub Token (required):", type="password", placeholder="ghp_...")
    if github_token:
        os.environ["GITHUB_TOKEN"] = github_token

    clone_method = st.sidebar.radio(
        "How to load files:",
        ["📥 Clone repo (recommended)", "☁️ Download via API (no git needed)"],
        index=0,
    )

    if st.sidebar.button("🚀 Load Repository", type="primary"):
        if repo_url and "github.com" in repo_url:
            parts = repo_url.rstrip("/").split("github.com/")[-1].split("/")
            if len(parts) >= 2:
                repo_name = parts[1].replace(".git", "")
                owner_repo = f"{parts[0]}/{repo_name}"
                local_path = clone_dir / repo_name
                os.environ["GITHUB_REPO"] = owner_repo

                if local_path.exists():
                    st.sidebar.success(f"✅ Already loaded!")
                elif clone_method == "📥 Clone repo (recommended)":
                    with st.sidebar:
                        with st.spinner(f"Cloning {owner_repo}..."):
                            clone_dir.mkdir(parents=True, exist_ok=True)
                            import subprocess, base64
                            clone_url = repo_url
                            # Keep the token OUT of the URL (which would persist in
                            # the git remote and could leak via stderr/process list).
                            # Pass it as a one-shot Authorization header instead.
                            git_cmd = ["git"]
                            if github_token and "github.com" in clone_url:
                                basic = base64.b64encode(
                                    f"x-access-token:{github_token}".encode()
                                ).decode()
                                git_cmd += ["-c", f"http.extraHeader=Authorization: Basic {basic}"]
                            git_cmd += ["clone", "--depth", "50", clone_url, str(local_path)]
                            result = subprocess.run(
                                git_cmd, capture_output=True, text=True, timeout=120
                            )
                            if result.returncode == 0:
                                st.success(f"✅ Cloned!")
                            else:
                                # Scrub any credentials before surfacing the error.
                                err = result.stderr[:200]
                                if github_token:
                                    err = err.replace(github_token, "***")
                                st.error(f"❌ Failed: {err}")
                else:
                    # Download via GitHub API (no git needed)
                    if not github_token:
                        st.sidebar.error("GitHub Token required for API download")
                    else:
                        with st.sidebar:
                            with st.spinner(f"Downloading {owner_repo} via API..."):
                                try:
                                    import zipfile, io, requests as req
                                    clone_dir.mkdir(parents=True, exist_ok=True)
                                    headers = {"Authorization": f"token {github_token}"}
                                    zip_url = f"https://api.github.com/repos/{owner_repo}/zipball"
                                    resp = req.get(zip_url, headers=headers, stream=True, timeout=60)
                                    if resp.status_code == 200:
                                        z = zipfile.ZipFile(io.BytesIO(resp.content))
                                        # Extract to local_path
                                        local_path.mkdir(parents=True, exist_ok=True)
                                        # Zip has a root folder like "user-repo-sha/"
                                        root_prefix = z.namelist()[0].split("/")[0] + "/"
                                        for member in z.namelist():
                                            if member.endswith("/"):
                                                continue
                                            rel_path = member[len(root_prefix):]
                                            if not rel_path:
                                                continue
                                            target = local_path / rel_path
                                            target.parent.mkdir(parents=True, exist_ok=True)
                                            target.write_bytes(z.read(member))
                                        st.success(f"✅ Downloaded!")
                                    else:
                                        st.error(f"❌ API error {resp.status_code}: {resp.text[:100]}")
                                except Exception as e:
                                    st.error(f"❌ Download failed: {e}")
                st.rerun()
        else:
            st.sidebar.error("Enter a valid GitHub URL")

    # Auto-detect cloned/downloaded repo
    if repo_url and "github.com" in repo_url:
        parts = repo_url.rstrip("/").split("github.com/")[-1].split("/")
        if len(parts) >= 2:
            repo_name = parts[1].replace(".git", "")
            local_path = clone_dir / repo_name
            os.environ["GITHUB_REPO"] = f"{parts[0]}/{repo_name}"
            if local_path.exists():
                project_root = str(local_path)
                os.environ["CODESENSE_REPO_PATH"] = project_root
                st.sidebar.success(f"📁 Active: **{repo_name}**")

else:
    # Local folder
    project_root = st.sidebar.text_input("📁 Local folder path:", value=str(Path.cwd()))
    if project_root and Path(project_root).is_dir():
        os.environ["CODESENSE_REPO_PATH"] = project_root
        st.sidebar.success(f"📁 Active: **{Path(project_root).name}**")
    else:
        st.sidebar.warning("Enter a valid folder path")

# ─── Gemini API Keys (Required — supports multiple for rotation) ─────────────
st.sidebar.markdown("---")
st.sidebar.subheader("🔑 Gemini API Keys (Required)")
st.sidebar.caption("Add multiple keys — if one hits rate limit, the next is used automatically")

num_keys = st.sidebar.number_input("Number of API keys:", min_value=1, max_value=10, value=1)

gemini_keys = []
for i in range(num_keys):
    key = st.sidebar.text_input(f"Key {i+1}:", type="password", placeholder=f"AIza... (key #{i+1})", key=f"gemini_key_{i}")
    if key and key.strip():
        gemini_keys.append(key.strip())
        os.environ[f"GEMINI_KEY_{i+1}"] = key.strip()

if gemini_keys:
    st.sidebar.success(f"✅ {len(gemini_keys)} key(s) loaded — auto-rotation enabled")
elif not mock_mode:
    st.sidebar.warning("⚠️ Add your Gemini key(s) or enable Demo Mode")

if mock_mode:
    os.environ["CODESENSE_MOCK"] = "true"
else:
    os.environ.pop("CODESENSE_MOCK", None)

st.sidebar.markdown("---")
st.sidebar.markdown("**Commands:**")
command = st.sidebar.radio(
    "Choose a capability:",
    [
        "🏠 Home",
        "🔍 Explain (WHY)",
        "📝 Describe (WHAT)",
        "🌳 Tree (Structure)",
        "🔄 Flow (Execution)",
        "📊 Diagram (Visual)",
        "📜 Trace (Timeline)",
        "📦 Dependencies",
        "🔗 Related Files",
        "⚠️ Risk Assessment",
        "📚 Onboard Guide",
        "💬 Ask (Natural Language)",
        "📥 Ingest Documents",
    ],
    index=0,
)


# ─── Helper Functions ─────────────────────────────────────────────────────────


def get_gemini_service():
    """Create GeminiService if keys are available."""
    try:
        from codesense.llm import GeminiService, KeyRotator

        keys = []
        i = 1
        while True:
            key = os.environ.get(f"GEMINI_KEY_{i}")
            if not key:
                break
            keys.append(key)
            i += 1

        if not keys:
            return None
        return GeminiService(key_rotator=KeyRotator(api_keys=keys))
    except Exception:
        return None


def render_command_output(output):
    """Render a CommandOutput object in Streamlit."""
    if output.is_demo_mode:
        st.info("🎭 **DEMO MODE** — Using built-in sample data")

    if output.confidence is not None:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(output.title)
        with col2:
            confidence_pct = output.confidence * 100
            if confidence_pct > 70:
                st.success(f"Confidence: {confidence_pct:.0f}%")
            elif confidence_pct >= 40:
                st.warning(f"Confidence: {confidence_pct:.0f}%")
            else:
                st.error(f"Confidence: {confidence_pct:.0f}%")
    else:
        st.subheader(output.title)

    # Check if content looks like a tree/code structure (has box-drawing chars or indentation)
    content = output.content
    if any(ch in content for ch in ("├──", "└──", "│", "┌", "┐", "┘", "└")):
        # Render as preformatted code block to preserve structure
        st.code(content, language="")
    elif content.startswith("```"):
        # Already fenced code — render as markdown
        st.markdown(content)
    else:
        st.markdown(content)

    # Code snippets
    for snippet in output.code_snippets:
        if snippet.label:
            st.caption(snippet.label)
        st.code(snippet.code, language=snippet.language or "text")

    # Tables
    for table in output.tables:
        if table.title:
            st.caption(table.title)
        import pandas as pd
        df = pd.DataFrame(table.rows, columns=table.headers)
        st.dataframe(df, use_container_width=True)

    # Conflicts
    if output.conflicts:
        st.markdown("---")
        st.subheader("⚠️ Unresolved Conflicts")
        for i, conflict in enumerate(output.conflicts, 1):
            with st.expander(f"Conflict {i}: {conflict.description}", expanded=True):
                cols = st.columns(len(conflict.sources))
                for col, source in zip(cols, conflict.sources):
                    with col:
                        st.markdown(f"**Source: `{source.source_id}`**")
                        st.markdown(f"> {source.claim}")


# ─── Pages ────────────────────────────────────────────────────────────────────


if command == "🏠 Home":
    st.title("🔍 CodeSense")
    st.markdown("### *Codebase Intelligence — Answer WHY Code Exists*")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Commands", "12")
        st.markdown("Explain, Describe, Tree, Flow, Diagram, Trace, Deps, Related, Risk, Onboard, Ask, Ingest")
    with col2:
        st.metric("AI Engine", "Gemini")
        st.markdown("5-node reasoning loop with evidence gathering and verification")
    with col3:
        st.metric("Data Sources", "4")
        st.markdown("Git history, GitHub issues, PR comments, Architecture docs")

    st.markdown("---")
    st.markdown("""
    ## How It Works

    1. **🔎 Explore** — Gathers evidence from git, GitHub, and decision docs
    2. **💡 Hypothesize** — AI generates candidate explanations
    3. **✅ Verify** — Checks evidence support for each hypothesis
    4. **⚖️ Check Contradictions** — Surfaces conflicting information honestly
    5. **📋 Synthesize** — Produces a grounded answer with citations

    ## Quick Start

    1. Enable **Demo Mode** in the sidebar (no API keys needed!)
    2. Pick a command from the sidebar
    3. Enter a file path and click Run

    > 💡 **Tip:** Start with **Tree** to see your project structure, then use **Explain** on interesting files.
    """)

    st.markdown("---")
    st.caption("Built with Python, LangGraph, LangChain, Google Gemini, FastMCP, ChromaDB")


elif command == "🔍 Explain (WHY)":
    st.title("🔍 Explain — Why Does This Code Exist?")
    st.markdown("Runs the full AI reasoning loop to explain the rationale behind code.")

    col1, col2 = st.columns([3, 1])
    with col1:
        path = st.text_input("File path:", placeholder="e.g., src/auth.py")
    with col2:
        function = st.text_input("Function (optional):", placeholder="e.g., process_retry")

    if st.button("🔍 Explain", type="primary"):
        if not path:
            st.error("Please enter a file path.")
        else:
            with st.spinner("Reasoning... (gathering evidence, forming hypotheses, verifying)"):
                try:
                    from codesense.capabilities.explain import ExplainHandler
                    from codesense.models.output import CommandParams

                    gemini = get_gemini_service()
                    capability = ExplainHandler(gemini_service=gemini)
                    params = CommandParams(path=path, mock=mock_mode)
                    result = capability.run(params)
                    render_command_output(result)
                except Exception as e:
                    st.error(f"Error: {e}")


elif command == "📝 Describe (WHAT)":
    st.title("📝 Describe — What Does This Code Do?")
    st.markdown("Provides a plain-English description without needing git history.")

    path = st.text_input("File path:", placeholder="e.g., codesense/main.py")
    col1, col2 = st.columns(2)
    with col1:
        function = st.text_input("Function (optional):", placeholder="e.g., explain")
    with col2:
        lines = st.text_input("Line range (optional):", placeholder="e.g., 10-20")

    if st.button("📝 Describe", type="primary"):
        if not path:
            st.error("Please enter a file path.")
        elif not Path(path).exists():
            st.error(f"File not found: {path}")
        else:
            with st.spinner("Reading and describing code..."):
                try:
                    from codesense.capabilities.describe import DescribeCapabilityHandler

                    gemini = get_gemini_service()
                    handler = DescribeCapabilityHandler(gemini_service=gemini)
                    result = handler.run(path, function=function or None, lines=lines or None, mock=mock_mode)
                    render_command_output(result)
                except Exception as e:
                    st.error(f"Error: {e}")


elif command == "🌳 Tree (Structure)":
    st.title("🌳 Project Tree")
    st.markdown("Shows your project structure with optional descriptions.")

    col1, col2 = st.columns([3, 1])
    with col1:
        tree_path = st.text_input("Path (leave empty for current directory):", value=project_root)
    with col2:
        depth = st.number_input("Max depth:", min_value=1, max_value=10, value=3)

    if st.button("🌳 Show Tree", type="primary"):
        with st.spinner("Walking directory..."):
            try:
                from codesense.output.tree_formatter import TreeFormatter

                formatter = TreeFormatter()
                tree = formatter.build_tree(tree_path or ".", depth=depth, gitignore=True)
                tree_str = formatter.format(tree)

                st.markdown("### 🌳 Project Structure")
                st.code(tree_str, language="")
            except Exception as e:
                st.error(f"Error: {e}")


elif command == "🔄 Flow (Execution)":
    st.title("🔄 Execution Flow")
    st.markdown("Traces the call path from an entry point and generates a sequence diagram.")

    path = st.text_input("Entry point file:", placeholder="e.g., codesense/main.py")
    function = st.text_input("Starting function (optional):", placeholder="e.g., explain")

    if st.button("🔄 Trace Flow", type="primary"):
        if not path:
            st.error("Please enter a file path.")
        else:
            with st.spinner("Tracing execution path..."):
                try:
                    from codesense.capabilities.flow import run as flow_run

                    entry = f"{path}::{function}" if function else path
                    result = flow_run(from_path=entry, project_root=project_root)
                    render_command_output(result)
                except Exception as e:
                    st.error(f"Error: {e}")


elif command == "📊 Diagram (Visual)":
    st.title("📊 Architecture Diagram")
    st.markdown("Generates Mermaid diagrams showing code relationships.")

    path = st.text_input("Path to analyze:", placeholder="e.g., codesense/agent/")
    diagram_type = st.selectbox("Diagram type:", ["flowchart", "sequence", "architecture"])

    if st.button("📊 Generate Diagram", type="primary"):
        if not path:
            st.error("Please enter a path.")
        else:
            with st.spinner("Analyzing code structure..."):
                try:
                    from codesense.capabilities.diagram import DiagramHandler
                    from codesense.models.output import CommandParams

                    handler = DiagramHandler(project_root=project_root)
                    params = CommandParams(path=path, query=diagram_type, mock=mock_mode)
                    result = handler.run(params)

                    # Render the Mermaid diagram natively in Streamlit
                    if result.code_snippets:
                        mermaid_code = result.code_snippets[0].code
                        st.markdown(result.content)
                        st.markdown(f"```mermaid\n{mermaid_code}\n```")
                        with st.expander("Raw Mermaid code (copy for GitHub/Notion)"):
                            st.code(mermaid_code, language="mermaid")
                    else:
                        render_command_output(result)
                except Exception as e:
                    st.error(f"Error: {e}")


elif command == "📜 Trace (Timeline)":
    st.title("📜 Decision Timeline")
    st.markdown("Shows the chronological history of commits, issues, and PRs for a file.")

    path = st.text_input("File path:", placeholder="e.g., src/payments/gateway.py")
    line = st.number_input("Line number (optional, 0 = all):", min_value=0, value=0)

    if st.button("📜 Trace History", type="primary"):
        if not path:
            st.error("Please enter a file path.")
        else:
            with st.spinner("Gathering historical data..."):
                try:
                    from codesense.capabilities.trace import TraceHandler
                    from codesense.models.output import CommandParams

                    handler = TraceHandler()
                    params = CommandParams(path=path, line_number=line if line > 0 else None, mock=mock_mode)
                    result = handler.run(params)
                    render_command_output(result)
                except Exception as e:
                    st.error(f"Error: {e}")


elif command == "📦 Dependencies":
    st.title("📦 Dependency Analysis")
    st.markdown("Shows external packages, environment variables, APIs, and internal imports.")

    path = st.text_input("Module path:", placeholder="e.g., codesense/")
    dep_type = st.selectbox("Filter by type:", ["all", "packages", "env", "api"])

    if st.button("📦 Scan Dependencies", type="primary"):
        with st.spinner("Scanning imports and dependencies..."):
            try:
                from codesense.capabilities.deps import DepsHandler
                from codesense.models.output import CommandParams

                handler = DepsHandler(project_root=project_root)
                params = CommandParams(path=path or None, mock=mock_mode, output=dep_type)
                result = handler.run(params)
                render_command_output(result)
            except Exception as e:
                st.error(f"Error: {e}")


elif command == "🔗 Related Files":
    st.title("🔗 Related Files & Impact Analysis")
    st.markdown("Shows what depends on a file and what the file depends on.")

    path = st.text_input("File path:", placeholder="e.g., codesense/llm/key_manager.py")

    if st.button("🔗 Find Related", type="primary"):
        if not path:
            st.error("Please enter a file path.")
        else:
            with st.spinner("Scanning import relationships..."):
                try:
                    from codesense.capabilities.related import RelatedHandler
                    from codesense.models.output import CommandParams

                    handler = RelatedHandler(project_root=project_root)
                    params = CommandParams(path=path, mock=mock_mode)
                    result = handler.run(params)
                    render_command_output(result)
                except Exception as e:
                    st.error(f"Error: {e}")


elif command == "⚠️ Risk Assessment":
    st.title("⚠️ Risk Assessment")
    st.markdown("Rates how dangerous it is to modify code (0-10 scale).")

    path = st.text_input("File path:", placeholder="e.g., codesense/agent/graph.py")

    if st.button("⚠️ Assess Risk", type="primary"):
        if not path:
            st.error("Please enter a file path.")
        else:
            with st.spinner("Computing risk signals..."):
                try:
                    from codesense.capabilities.risk import RiskHandler
                    from codesense.models.output import CommandParams

                    handler = RiskHandler(project_root=project_root)
                    params = CommandParams(path=path, mock=mock_mode)
                    result = handler.run(params)
                    render_command_output(result)
                except Exception as e:
                    st.error(f"Error: {e}")


elif command == "📚 Onboard Guide":
    st.title("📚 Onboarding Guide Generator")
    st.markdown("Generates a complete onboarding document for new developers.")

    module = st.text_input("Scope to module (optional):", placeholder="e.g., codesense/agent/")

    if st.button("📚 Generate Guide", type="primary"):
        with st.spinner("Generating onboarding document..."):
            try:
                from codesense.capabilities.onboard import OnboardHandler
                from codesense.models.output import CommandParams

                handler = OnboardHandler(project_root=project_root)
                params = CommandParams(path=module or None, mock=mock_mode)
                result = handler.run(params)
                render_command_output(result)

                # Download button
                st.download_button(
                    "📥 Download as Markdown",
                    data=result.content,
                    file_name="ONBOARDING.md",
                    mime="text/markdown",
                )
            except Exception as e:
                st.error(f"Error: {e}")


elif command == "💬 Ask (Natural Language)":
    st.title("💬 Ask Anything")
    st.markdown("Ask a question in plain English — CodeSense answers it using the full reasoning pipeline.")

    question = st.text_area(
        "Your question:",
        placeholder="e.g., Why does the retry logic exist in payments?\nWhat files are related to auth?\nIs it safe to delete the cache module?\nExplain this project",
        height=100,
    )

    if st.button("💬 Ask", type="primary"):
        if not question.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("Understanding your question and generating answer..."):
                try:
                    from codesense.capabilities.ask import IntentClassifier
                    from codesense.models.output import CommandParams

                    gemini = get_gemini_service()

                    # Step 1: Classify intent
                    classifier = IntentClassifier(gemini_service=gemini)
                    classification = classifier.classify(question)
                    intent = classification.get("intent", "explain")
                    confidence = classification.get("confidence", 0.0)
                    extracted_params = classification.get("params")

                    st.caption(f"🎯 Intent: `{intent}` (confidence: {confidence:.0%})")

                    # Step 2: Actually execute the handler based on intent
                    if intent == "explain" and gemini:
                        # For general "explain this project" questions, read files and ask LLM directly
                        target_path = extracted_params.path if extracted_params and extracted_params.path else None
                        
                        if target_path and Path(target_path).is_file():
                            # Specific file — use describe (reads file content)
                            from codesense.capabilities.describe import DescribeCapabilityHandler
                            handler = DescribeCapabilityHandler(gemini_service=gemini)
                            result = handler.run(str(target_path), mock=mock_mode)
                            render_command_output(result)
                        else:
                            # General project question — gather file list and key files, ask LLM
                            from codesense.models.output import CommandOutput
                            
                            # Read key project files for context
                            root = Path(project_root)
                            context_parts = []
                            
                            # Get file listing
                            py_files = sorted(root.rglob("*.py"))[:30]
                            if py_files:
                                file_list = "\n".join(str(f.relative_to(root)) for f in py_files)
                                context_parts.append(f"Project files:\n{file_list}")
                            
                            # Read README if exists
                            for readme_name in ["README.md", "readme.md", "README.rst"]:
                                readme = root / readme_name
                                if readme.is_file():
                                    content = readme.read_text(encoding="utf-8", errors="replace")[:2000]
                                    context_parts.append(f"README:\n{content}")
                                    break
                            
                            # Read main entry files
                            for entry_name in ["main.py", "app.py", "__main__.py", "manage.py"]:
                                for entry_file in root.rglob(entry_name):
                                    content = entry_file.read_text(encoding="utf-8", errors="replace")[:1500]
                                    rel = entry_file.relative_to(root)
                                    context_parts.append(f"File {rel}:\n{content}")
                                    break
                            
                            # Read pyproject.toml or requirements.txt
                            for config_name in ["pyproject.toml", "requirements.txt", "setup.py", "package.json"]:
                                config = root / config_name
                                if config.is_file():
                                    content = config.read_text(encoding="utf-8", errors="replace")[:1000]
                                    context_parts.append(f"{config_name}:\n{content}")
                                    break
                            
                            if context_parts:
                                full_context = "\n\n---\n\n".join(context_parts)
                                prompt = (
                                    f"Based on the following project files and context, answer this question: {question}\n\n"
                                    f"Project root: {project_root}\n\n"
                                    f"{full_context}\n\n"
                                    f"Provide a clear, detailed answer explaining this project."
                                )
                                # Sanitize for ASCII
                                prompt = prompt.encode("ascii", errors="replace").decode("ascii")
                                
                                try:
                                    answer = gemini.generate(prompt)
                                    result = CommandOutput(
                                        title="🔍 CodeSense — Project Analysis",
                                        content=answer,
                                        confidence=0.85,
                                        is_demo_mode=mock_mode,
                                    )
                                    render_command_output(result)
                                except Exception as e:
                                    st.error(f"LLM Error: {e}")
                            else:
                                st.warning("No project files found at the specified path.")

                    elif intent == "describe" and gemini:
                        from codesense.capabilities.describe import DescribeCapabilityHandler
                        file_path = extracted_params.path if extracted_params else None
                        if file_path and Path(file_path).exists():
                            handler = DescribeCapabilityHandler(gemini_service=gemini)
                            result = handler.run(file_path, mock=mock_mode)
                            render_command_output(result)
                        else:
                            # Use explain as fallback for general "what does this project do" questions
                            from codesense.capabilities.explain import ExplainHandler
                            params = CommandParams(path=project_root, query=question, mock=mock_mode)
                            capability = ExplainHandler(gemini_service=gemini)
                            result = capability.run(params)
                            render_command_output(result)

                    elif intent == "tree":
                        from codesense.output.tree_formatter import TreeFormatter
                        formatter = TreeFormatter()
                        tree = formatter.build_tree(project_root, depth=3, gitignore=True)
                        tree_str = formatter.format(tree)
                        st.code(tree_str, language="")

                    elif intent == "flow":
                        from codesense.capabilities.flow import run as flow_run
                        entry = extracted_params.path if extracted_params and extracted_params.path else project_root
                        result = flow_run(from_path=entry, project_root=project_root)
                        render_command_output(result)

                    elif intent == "diagram":
                        from codesense.capabilities.diagram import DiagramHandler
                        from codesense.models.output import CommandParams as CP
                        handler = DiagramHandler(project_root=project_root)
                        params = CP(path=project_root, query="architecture", mock=mock_mode)
                        result = handler.run(params)
                        render_command_output(result)

                    elif intent == "deps":
                        from codesense.capabilities.deps import DepsHandler
                        from codesense.models.output import CommandParams as CP
                        handler = DepsHandler(project_root=project_root)
                        params = CP(path=extracted_params.path if extracted_params else project_root, mock=mock_mode)
                        result = handler.run(params)
                        render_command_output(result)

                    elif intent == "related":
                        from codesense.capabilities.related import RelatedHandler
                        from codesense.models.output import CommandParams as CP
                        file_path = extracted_params.path if extracted_params and extracted_params.path else None
                        if file_path:
                            handler = RelatedHandler(project_root=project_root)
                            params = CP(path=file_path, mock=mock_mode)
                            result = handler.run(params)
                            render_command_output(result)
                        else:
                            st.warning("Please mention a specific file path in your question.")

                    elif intent == "risk":
                        from codesense.capabilities.risk import RiskHandler
                        from codesense.models.output import CommandParams as CP
                        file_path = extracted_params.path if extracted_params and extracted_params.path else None
                        if file_path:
                            handler = RiskHandler(project_root=project_root)
                            params = CP(path=file_path, mock=mock_mode)
                            result = handler.run(params)
                            render_command_output(result)
                        else:
                            st.warning("Please mention a specific file path to assess risk.")

                    elif intent == "trace":
                        from codesense.capabilities.trace import TraceHandler
                        from codesense.models.output import CommandParams as CP
                        handler = TraceHandler()
                        params = CP(path=extracted_params.path if extracted_params else "", mock=mock_mode)
                        result = handler.run(params)
                        render_command_output(result)

                    elif intent == "onboard":
                        from codesense.capabilities.onboard import OnboardHandler
                        from codesense.models.output import CommandParams as CP
                        handler = OnboardHandler(project_root=project_root)
                        params = CP(mock=mock_mode)
                        result = handler.run(params)
                        render_command_output(result)

                    else:
                        # Fallback: run explain with the question
                        if gemini:
                            from codesense.capabilities.explain import ExplainHandler
                            params = CommandParams(path=project_root, query=question, mock=mock_mode)
                            capability = ExplainHandler(gemini_service=gemini)
                            result = capability.run(params)
                            render_command_output(result)
                        else:
                            st.error("Gemini API key required for this question.")

                except Exception as e:
                    st.error(f"Error: {e}")


elif command == "📥 Ingest Documents":
    st.title("📥 Document Ingestion")
    st.markdown("Feed architecture decision records and design docs into CodeSense's memory.")

    folder = st.text_input("Folder path with .md documents:", placeholder="e.g., docs/adr/")

    if st.button("📥 Ingest", type="primary"):
        if not folder:
            st.error("Please enter a folder path.")
        elif not Path(folder).is_dir():
            st.error(f"Folder not found: {folder}")
        else:
            with st.spinner("Ingesting documents into Decision Memory..."):
                try:
                    from codesense.memory.ingest import IngestPipeline

                    pipeline = IngestPipeline()
                    results = pipeline.ingest_folder(folder)

                    success = sum(1 for r in results if r.success)
                    failed = len(results) - success
                    chunks = sum(r.chunks_created for r in results)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Documents", len(results))
                    col2.metric("Succeeded", success)
                    col3.metric("Chunks Created", chunks)

                    if failed > 0:
                        st.warning(f"{failed} document(s) failed:")
                        for r in results:
                            if not r.success:
                                st.error(f"❌ {r.document_id}: {r.error}")

                    for r in results:
                        if r.success:
                            st.success(f"✅ {r.document_id} — {r.chunks_created} chunks")
                except Exception as e:
                    st.error(f"Error: {e}")
