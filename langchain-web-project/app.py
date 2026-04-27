"""
ARIA — Streamlit UI for the Linux desktop AI agent.
Run with: streamlit run app.py
"""
import streamlit as st
import time
from pathlib import Path

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="ARIA · Linux Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg:       #0d0f14;
    --bg2:      #13161e;
    --bg3:      #1a1e2a;
    --accent:   #7fffb0;
    --accent2:  #4fc3f7;
    --warn:     #ffb347;
    --danger:   #ff6b6b;
    --text:     #e8eaf0;
    --muted:    #6b7280;
    --border:   #2a2f3d;
    --radius:   10px;
}

/* Global */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent) !important;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

/* Title bar */
.aria-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 0 10px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
}
.aria-logo {
    font-family: 'Space Mono', monospace;
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -0.03em;
    line-height: 1;
}
.aria-sub {
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace;
}

/* Chat messages */
.msg-user {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent2);
    border-radius: var(--radius);
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.92rem;
}
.msg-assistant {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius);
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.92rem;
    white-space: pre-wrap;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.7;
}
.msg-label {
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace;
    margin-bottom: 6px;
}
.msg-label.user  { color: var(--accent2); }
.msg-label.agent { color: var(--accent);  }

/* Tool badge */
.tool-badge {
    display: inline-block;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    color: var(--warn);
    margin: 2px 3px 4px 0;
}

/* Status dot */
.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent);
    display: inline-block;
    margin-right: 6px;
    box-shadow: 0 0 6px var(--accent);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.4; }
}

/* Inputs */
.stTextInput input, .stTextArea textarea {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: var(--radius) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.88rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(127,255,176,0.15) !important;
}

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    border-radius: var(--radius) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: var(--bg) !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 12px 16px;
}
[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: 'Space Mono', monospace !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--muted) !important;
    letter-spacing: 0.08em;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg2) !important;
    border-bottom: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em;
    color: var(--muted) !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* Upload area */
[data-testid="stFileUploadDropzone"] {
    background: var(--bg3) !important;
    border: 1px dashed var(--border) !important;
    border-radius: var(--radius) !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ──────────────────────────────────────────────────────
if "agent" not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_error" not in st.session_state:
    st.session_state.agent_error = None
if "verbose" not in st.session_state:
    st.session_state.verbose = False


# ── Lazy agent init ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_agent(verbose: bool = False):
    from agent.core import build_agent
    return build_agent(verbose=verbose)


def load_agent():
    try:
        st.session_state.agent = get_agent(st.session_state.verbose)
        st.session_state.agent_error = None
    except Exception as e:
        st.session_state.agent_error = str(e)


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ◈ ARIA")
    st.markdown("**Autonomous Reasoning & Interaction Agent**")
    st.markdown("---")

    # Status
    if st.session_state.agent:
        st.markdown('<span class="status-dot"></span> **Agent Online**', unsafe_allow_html=True)
    elif st.session_state.agent_error:
        st.error(f"⚠️ {st.session_state.agent_error}")
    else:
        st.markdown("**Agent not started**")

    if st.button("⚡ Initialize Agent", use_container_width=True):
        with st.spinner("Booting ARIA..."):
            load_agent()
        st.rerun()

    st.markdown("---")

    # ── RAG Section ────────────────────────────────────────────────────
    st.markdown("### ◈ KNOWLEDGE BASE")

    uploaded = st.file_uploader(
        "Upload document",
        type=["pdf", "docx", "txt", "md", "csv", "log"],
        label_visibility="collapsed",
    )
    if uploaded and st.button("📥 Ingest Document", use_container_width=True):
        from agent.rag import ingest_file_bytes, get_kb_stats
        with st.spinner(f"Ingesting {uploaded.name}..."):
            result = ingest_file_bytes(uploaded.name, uploaded.read())
        st.success(result)

    if st.button("📚 Show KB Stats", use_container_width=True):
        from agent.rag import get_kb_stats
        stats = get_kb_stats()
        st.metric("Documents", stats["total_documents"])
        st.metric("Chunks", stats["total_chunks"])
        if stats["documents"]:
            st.markdown("**Files:**")
            for d in stats["documents"]:
                st.markdown(f"- `{d}`")

    st.markdown("---")

    # ── Settings ──────────────────────────────────────────────────────
    st.markdown("### ◈ SETTINGS")
    verbose_new = st.toggle("Verbose agent logs", value=st.session_state.verbose)
    if verbose_new != st.session_state.verbose:
        st.session_state.verbose = verbose_new
        st.cache_resource.clear()
        st.session_state.agent = None
        st.rerun()

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        # Clear memory on the live agent instance
        if st.session_state.agent:
            try:
                st.session_state.agent.memory.clear()
            except Exception:
                pass
        # Also bust the cache so a fresh agent is created next time
        st.cache_resource.clear()
        st.session_state.agent = None
        st.rerun()

    st.markdown("---")

    # ── Tool browser ──────────────────────────────────────────────────
    st.markdown("### ◈ TOOLS")
    try:
        from agent.core import get_tool_manifest
        manifest = get_tool_manifest()
        cats = {}
        for t in manifest:
            cats.setdefault(t["category"], []).append(t)
        for cat, tools in cats.items():
            with st.expander(f"{cat} ({len(tools)})", expanded=False):
                for t in tools:
                    st.markdown(f'<span class="tool-badge">{t["name"]}</span>', unsafe_allow_html=True)
    except Exception:
        st.markdown("*Initialize agent to see tools*")


# ── Main area ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="aria-header">
    <div>
        <div class="aria-logo">ARIA</div>
        <div class="aria-sub">Linux Desktop Intelligence Agent</div>
    </div>
</div>
""", unsafe_allow_html=True)

tab_chat, tab_tools, tab_about = st.tabs(["💬  CHAT", "🔧  TOOLS", "ℹ️  ABOUT"])

# ── CHAT TAB ────────────────────────────────────────────────────────────────
with tab_chat:
    # Render history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align:center; padding: 60px 0; color: var(--muted);">
                <div style="font-family:'Space Mono',monospace; font-size:2rem; color:var(--accent); opacity:0.3;">◈</div>
                <div style="font-family:'Space Mono',monospace; font-size:0.75rem; letter-spacing:0.15em; margin-top:12px;">
                    ARIA READY · INITIALIZE AGENT · THEN ASK ANYTHING
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="msg-user">
                        <div class="msg-label user">YOU</div>
                        {msg['content']}
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="msg-assistant">
                        <div class="msg-label agent">ARIA</div>
                        {msg['content']}
                    </div>""", unsafe_allow_html=True)

    # Input row
    col_input, col_send = st.columns([6, 1])
    with col_input:
        user_input = st.text_input(
            "Message",
            placeholder="Ask ARIA anything about your system, files, or documents…",
            label_visibility="collapsed",
            key="user_input",
        )
    with col_send:
        send = st.button("Send ▶", use_container_width=True)

    # Quick actions
    st.markdown("**Quick actions:**")
    qcols = st.columns(4)
    quick = [
        ("💻 System Info",    "Get full system information including CPU, RAM, and disk usage"),
        ("📂 List Desktop",   "List all files on my Desktop"),
        ("📚 List KB Docs",   "List all documents in the knowledge base"),
        ("⏰ Current Time",   "What is the current date and time?"),
    ]
    for i, (label, prompt) in enumerate(quick):
        if qcols[i].button(label, use_container_width=True):
            user_input = prompt
            send = True

    # Process message
    if send and user_input.strip():
        if not st.session_state.agent:
            st.warning("Please initialize the agent first (sidebar → ⚡ Initialize Agent)")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.spinner("ARIA is thinking…"):
                try:
                    result = st.session_state.agent.invoke({"input": user_input})
                    answer = result.get("output", str(result))
                except Exception as e:
                    answer = f"❌ Agent error: {e}"
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()


# ── TOOLS TAB ───────────────────────────────────────────────────────────────
with tab_tools:
    st.markdown("### Available Tools")
    try:
        from agent.core import get_tool_manifest
        manifest = get_tool_manifest()
        cats = {}
        for t in manifest:
            cats.setdefault(t["category"], []).append(t)

        for cat, tools in cats.items():
            st.markdown(f"#### {cat}")
            cols = st.columns(2)
            for i, t in enumerate(tools):
                with cols[i % 2]:
                    with st.expander(f"`{t['name']}`"):
                        st.markdown(t["description"])
    except Exception as e:
        st.info(f"Initialize the agent to browse tools. ({e})")


# ── ABOUT TAB ───────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
### ARIA — Autonomous Reasoning & Interaction Agent

ARIA is a local AI-powered Linux desktop agent built with:

| Component | Technology |
|---|---|
| LLM | Google Gemini 2.0 Flash |
| Agent Framework | LangChain ReAct |
| RAG Vector Store | ChromaDB (local) |
| Embeddings | `all-MiniLM-L6-v2` (local, CPU) |
| UI | Streamlit |

#### Capabilities

**System Tools** — file management, process control, shell commands, system info, screenshots, volume control

**Extended Tools** — clipboard, git operations, zip/archive, desktop notifications, cron jobs, network diagnostics, env vars

**RAG / Knowledge** — ingest PDF / DOCX / TXT documents, semantic search, document management

#### Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create .env with your API key
echo "GOOGLE_API_KEY=your_key_here" > .env

# 3. Launch
streamlit run app.py
```

#### Usage Tips

- Use the sidebar to upload and ingest documents into the knowledge base
- Ask "What does my report.pdf say about X?" to query ingested docs  
- All data is stored **locally** — no files leave your machine
- The agent has memory: it remembers the last 10 exchanges in a session
    """)