# ARIA — Autonomous Reasoning & Interaction Agent

> A local AI-powered Linux desktop agent with a Streamlit UI, RAG document intelligence, and full system control — built with LangChain ReAct + Google Gemini.

---

## What is ARIA?

ARIA is a conversational AI agent that runs entirely on your local machine. You can ask it to manage files, control processes, query documents you've uploaded, run shell commands, manage git repos, and much more — all through a clean chat interface.

It uses the **ReAct (Reasoning + Acting)** pattern: instead of just generating text, the agent thinks step by step, picks the right tool, runs it, reads the result, and reasons again until it has a confident answer.

---

## Features

### 🖥️ System Tools
- Create, delete, copy, move, read, and write files and folders
- List directory contents with file sizes
- Run arbitrary shell commands
- Open applications and files with their default app
- Get system info (CPU, RAM, disk, uptime)
- List and kill running processes
- Take screenshots, set volume, check internet connectivity
- Get current date and time

### 🔧 Extended Tools
- Read and write to the system clipboard
- Send desktop notifications
- Zip, unzip, and create tar archives
- Git operations: status, log, commit, clone
- Inspect and list environment variables
- Check disk usage per directory
- Network info and ping hosts
- List and add cron jobs

### 📚 RAG / Knowledge Base
- Ingest **PDF**, **DOCX**, **TXT**, **MD**, **CSV** documents
- Semantic search over your documents using local embeddings
- Powered by **ChromaDB** (fully local, no API calls)
- Embeddings via `all-MiniLM-L6-v2` (runs on CPU)
- Ask questions like *"What does my report say about X?"* and get answers grounded in your actual files

### 💬 Streamlit UI
- Dark terminal-aesthetic chat interface
- Sidebar document upload and ingestion
- Knowledge base stats dashboard
- Browsable tool catalog
- Quick-action buttons
- Session memory (last 10 exchanges)

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash |
| Agent Framework | LangChain ReAct |
| Vector Store | ChromaDB (local) |
| Embeddings | `all-MiniLM-L6-v2` (local, CPU) |
| Document Loaders | LangChain Community (PDF, DOCX, TXT) |
| UI | Streamlit |
| System Tools | Python `psutil`, `subprocess`, `shutil` |

---

## Project Structure

```
linux-agent/
├── app.py                  # Streamlit UI
├── agent/
│   ├── __init__.py
│   ├── core.py             # LangChain ReAct agent setup
│   ├── tools_system.py     # File, process, shell tools
│   ├── tools_extended.py   # Git, clipboard, zip, cron, network tools
│   └── rag.py              # RAG pipeline (ChromaDB + embeddings)
├── chroma_db/              # Auto-created local vector store
├── .env                    # Your API key goes here
└── requirements.txt
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/linux-agent.git
cd linux-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API key

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

Get a free key at [aistudio.google.com](https://aistudio.google.com).

### 4. Run

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

### Chat
1. Click **⚡ Initialize Agent** in the sidebar
2. Type any question or command in the chat input
3. ARIA will reason through it and use the appropriate tools

### Document Q&A
1. Upload a file using the sidebar uploader (PDF, DOCX, TXT, MD, CSV)
2. Click **📥 Ingest Document**
3. Ask questions about it in the chat — e.g. *"What is my name according to hello.txt?"*

### Example prompts
```
List all files on my Desktop
What is my CPU usage right now?
Create a folder called Projects on my Desktop
What does my resume.pdf say about my experience?
Show me the git log for ~/myproject
Zip the folder ~/Documents/reports and save it to Desktop
```

---

## How ReAct Works

ARIA uses the ReAct (Reasoning + Acting) pattern:

```
Question → Thought → Action → Observation → Thought → ... → Final Answer
```

For every question, the agent:
1. **Thinks** about what it needs to do
2. **Picks a tool** (e.g. `query_knowledge_base`, `run_shell_command`)
3. **Runs it** and reads the result
4. **Thinks again** based on the observation
5. Repeats until it has a confident **Final Answer**

This means ARIA grounds its answers in real data from your system and documents — not hallucinated responses.

---

## Requirements

- Python 3.10+
- Linux (Ubuntu recommended)
- Internet connection (for Gemini API calls only)
- Optional system packages for full functionality:
  ```bash
  sudo apt install scrot xclip libnotify-bin
  ```

---

## License

MIT