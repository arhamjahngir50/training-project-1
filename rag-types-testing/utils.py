"""
RAG Shared Utilities
Chunking, embedding, BM25, FAISS index — no LLM required.
"""

from __future__ import annotations
import re, time, math
from typing import List, Tuple, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss

# ── Embedding model (loaded once) ─────────────────────────────────────────────
_EMBED_MODEL: Optional[SentenceTransformer] = None

def get_embed_model() -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL

def embed(texts: List[str]) -> np.ndarray:
    """Return L2-normalised embeddings, shape (N, D)."""
    model = get_embed_model()
    vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    return (vecs / norms).astype("float32")

# ── Sentence-level chunking ────────────────────────────────────────────────────
def sentence_split(text: str) -> List[str]:
    """Simple sentence splitter (no NLTK needed)."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def chunk_text(text: str, chunk_size: int = 3, overlap: int = 1) -> List[str]:
    """
    Split text into overlapping sentence windows.
    chunk_size = number of sentences per chunk.
    overlap    = sentences shared between consecutive chunks.
    """
    sentences = sentence_split(text)
    if len(sentences) <= chunk_size:
        return [text]
    chunks, step = [], max(1, chunk_size - overlap)
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

# ── BM25 sparse retriever ─────────────────────────────────────────────────────
def simple_tokenise(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())

class BM25Index:
    def __init__(self, chunks: List[str]):
        tokenised = [simple_tokenise(c) for c in chunks]
        self.bm25 = BM25Okapi(tokenised)
        self.chunks = chunks

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        tokens = simple_tokenise(query)
        scores = self.bm25.get_scores(tokens)
        ranked = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in ranked if scores[i] > 0]

# ── FAISS dense retriever ─────────────────────────────────────────────────────
class FAISSIndex:
    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        vecs = embed(chunks)
        dim = vecs.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product on normalised = cosine
        self.index.add(vecs)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        q_vec = embed([query])
        scores, indices = self.index.search(q_vec, top_k)
        return [
            (self.chunks[idx], float(scores[0][rank]))
            for rank, idx in enumerate(indices[0])
            if idx >= 0
        ]

# ── Cross-encoder reranker (uses cosine sim as proxy — no model needed) ────────
def cross_encoder_rerank(query: str, candidates: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Production cross-encoders are fine-tuned models. Here we use a richer
    similarity: embedding cosine + token overlap (TF-IDF-like) for a
    meaningful proxy that behaves differently from plain vector search.
    """
    if not candidates:
        return []
    q_tokens = set(simple_tokenise(query))
    q_vec = embed([query])[0]
    c_vecs = embed(candidates)
    scored = []
    for i, (c_text, c_vec) in enumerate(zip(candidates, c_vecs)):
        cosine = float(np.dot(q_vec, c_vec))
        c_tokens = set(simple_tokenise(c_text))
        overlap = len(q_tokens & c_tokens) / (len(q_tokens | c_tokens) + 1e-10)
        score = 0.7 * cosine + 0.3 * overlap
        scored.append((c_text, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────
def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]], k: int = 60
) -> List[Tuple[str, float]]:
    """
    Merge multiple ranked result lists using RRF.
    k=60 is the standard constant that dampens the influence of top ranks.
    """
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (doc, _) in enumerate(ranked):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# ── Context compression ────────────────────────────────────────────────────────
def compress_context(query: str, chunks: List[str], threshold: float = 0.30) -> List[str]:
    """
    Keep only sentences from retrieved chunks that are similar enough to the query.
    Reduces token count and noise before generation.
    """
    q_vec = embed([query])[0]
    kept = []
    for chunk in chunks:
        sentences = sentence_split(chunk)
        if not sentences:
            continue
        s_vecs = embed(sentences)
        relevant = [
            s for s, v in zip(sentences, s_vecs)
            if float(np.dot(q_vec, v)) >= threshold
        ]
        if relevant:
            kept.append(" ".join(relevant))
    return kept if kept else chunks  # fallback: return original

# ── Deduplication ─────────────────────────────────────────────────────────────
def deduplicate(chunks: List[str], similarity_threshold: float = 0.92) -> List[str]:
    """Remove near-duplicate chunks using cosine similarity."""
    if len(chunks) <= 1:
        return chunks
    vecs = embed(chunks)
    kept_indices = [0]
    for i in range(1, len(chunks)):
        is_dup = False
        for j in kept_indices:
            sim = float(np.dot(vecs[i], vecs[j]))
            if sim >= similarity_threshold:
                is_dup = True
                break
        if not is_dup:
            kept_indices.append(i)
    return [chunks[i] for i in kept_indices]

# ── Evaluation helpers ─────────────────────────────────────────────────────────
def context_precision(retrieved: List[str], relevant_doc_id: str, corpus_map: Dict) -> float:
    """Fraction of retrieved chunks that come from the expected source document."""
    if not retrieved:
        return 0.0
    relevant_text = corpus_map.get(relevant_doc_id, "")
    relevant_sents = set(sentence_split(relevant_text))
    hits = sum(
        1 for chunk in retrieved
        if any(s in chunk for s in relevant_sents)
    )
    return hits / len(retrieved)

def mean_reciprocal_rank(retrieved: List[str], relevant_doc_id: str, corpus_map: Dict) -> float:
    """MRR: 1/rank of first relevant chunk."""
    relevant_text = corpus_map.get(relevant_doc_id, "")
    relevant_sents = set(sentence_split(relevant_text))
    for rank, chunk in enumerate(retrieved, 1):
        if any(s in chunk for s in relevant_sents):
            return 1.0 / rank
    return 0.0

def semantic_similarity(query: str, retrieved: List[str]) -> float:
    """Average cosine similarity between query and retrieved chunks."""
    if not retrieved:
        return 0.0
    q_vec = embed([query])[0]
    r_vecs = embed(retrieved)
    return float(np.mean([np.dot(q_vec, v) for v in r_vecs]))

# ── Timer context manager ─────────────────────────────────────────────────────
class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000