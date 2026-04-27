"""
RAG Pipeline using ChromaDB + sentence-transformers.
Supports PDF, DOCX, and plain text ingestion.
"""
import os
from pathlib import Path
from typing import List, Optional

from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_DIR = str(Path(__file__).parent.parent / "chroma_db")
COLLECTION  = "linux_agent_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"  # fast, local, no API key needed

_embeddings: Optional[HuggingFaceEmbeddings] = None
_vectorstore: Optional[Chroma] = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def _get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=COLLECTION,
            embedding_function=_get_embeddings(),
            persist_directory=CHROMA_DIR,
        )
    return _vectorstore


def _reset_vectorstore():
    """Force reload (called after ingestion)."""
    global _vectorstore
    _vectorstore = None


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_document(path: str):
    """Load a document by extension and return list of LangChain Documents."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(path).load()
    elif ext in (".docx", ".doc"):
        return Docx2txtLoader(path).load()
    elif ext in (".txt", ".md", ".rst", ".csv", ".log"):
        return TextLoader(path, encoding="utf-8").load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ── Public API (also exposed as agent tools) ──────────────────────────────────

@tool
def ingest_document(file_path: str) -> str:
    """
    Ingest a document (PDF, DOCX, TXT, MD) into the RAG knowledge base.
    The document will be chunked, embedded, and stored in ChromaDB.
    """
    try:
        path = os.path.expanduser(file_path.strip())
        if not os.path.exists(path):
            return f"❌ File not found: {path}"

        docs = _load_document(path)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        chunks = splitter.split_documents(docs)

        # Tag each chunk with the source filename
        source_name = Path(path).name
        for chunk in chunks:
            chunk.metadata["source"] = source_name

        vs = _get_vectorstore()
        vs.add_documents(chunks)
        _reset_vectorstore()

        return (
            f"✅ Ingested '{source_name}'\n"
            f"   Pages/sections: {len(docs)}\n"
            f"   Chunks stored:  {len(chunks)}"
        )
    except Exception as e:
        return f"❌ Ingest error: {e}"


@tool
def query_knowledge_base(question: str, top_k: int = 5) -> str:
    """
    Search the RAG knowledge base for documents relevant to a question.
    Returns the most relevant chunks with their source file names.
    Use this when the user asks about documents they've uploaded.
    """
    try:
        vs = _get_vectorstore()
        results = vs.similarity_search_with_relevance_scores(question, k=top_k)

        if not results:
            return "📭 No relevant documents found. Have you ingested any documents?"

        output = [f"📚 Found {len(results)} relevant chunk(s) for: '{question}'\n"]
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown")
            page   = doc.metadata.get("page", "")
            page_str = f" (p.{page+1})" if page != "" else ""
            output.append(
                f"── [{i}] {source}{page_str}  relevance={score:.2f}\n"
                f"{doc.page_content.strip()}\n"
            )
        return "\n".join(output)
    except Exception as e:
        return f"❌ Query error: {e}"


@tool
def list_ingested_documents() -> str:
    """List all documents currently stored in the RAG knowledge base."""
    try:
        vs = _get_vectorstore()
        col = vs._collection
        results = col.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])

        if not metadatas:
            return "📭 Knowledge base is empty. Ingest some documents first."

        sources = sorted({m.get("source", "unknown") for m in metadatas})
        lines = [f"📚 {len(sources)} document(s) in knowledge base:\n"]
        for s in sources:
            count = sum(1 for m in metadatas if m.get("source") == s)
            lines.append(f"  • {s}  ({count} chunks)")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Error: {e}"


@tool
def delete_document_from_kb(source_name: str) -> str:
    """
    Remove a specific document from the RAG knowledge base by its filename.
    Example: delete_document_from_kb('report.pdf')
    """
    try:
        vs = _get_vectorstore()
        col = vs._collection
        results = col.get(include=["metadatas"])
        ids_to_delete = [
            results["ids"][i]
            for i, m in enumerate(results["metadatas"])
            if m.get("source") == source_name
        ]
        if not ids_to_delete:
            return f"⚠️ No chunks found for '{source_name}'"
        col.delete(ids=ids_to_delete)
        _reset_vectorstore()
        return f"✅ Deleted {len(ids_to_delete)} chunks for '{source_name}'"
    except Exception as e:
        return f"❌ Error: {e}"


@tool
def clear_knowledge_base() -> str:
    """Delete ALL documents from the RAG knowledge base. Use with caution."""
    try:
        vs = _get_vectorstore()
        vs._collection.delete(where={"source": {"$ne": ""}})
        _reset_vectorstore()
        return "✅ Knowledge base cleared."
    except Exception as e:
        return f"❌ Error: {e}"


# ── Programmatic helpers (used by Streamlit, not exposed as agent tools) ──────

def ingest_file_bytes(filename: str, content: bytes) -> str:
    """Ingest a file from raw bytes (for Streamlit uploader)."""
    import tempfile
    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    result = ingest_document.invoke({"file_path": tmp_path})  # type: ignore
    os.unlink(tmp_path)
    # Patch the source name to the original filename
    result = result.replace(Path(tmp_path).name, filename)
    return result


def get_kb_stats() -> dict:
    """Return stats dict for Streamlit dashboard."""
    try:
        vs = _get_vectorstore()
        col = vs._collection
        results = col.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])
        sources = {m.get("source", "unknown") for m in metadatas}
        return {
            "total_chunks": len(metadatas),
            "total_documents": len(sources),
            "documents": sorted(sources),
        }
    except Exception:
        return {"total_chunks": 0, "total_documents": 0, "documents": []}


RAG_TOOLS = [
    ingest_document,
    query_knowledge_base,
    list_ingested_documents,
    delete_document_from_kb,
    clear_knowledge_base,
]