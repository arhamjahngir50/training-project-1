"""
Production RAG Implementations (No LLM Required)
Each RAG type is a standalone class with a .retrieve(query, top_k) method
returning RetrievalResult objects with full diagnostic metadata.
"""

from __future__ import annotations
import re, math, time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import networkx as nx

from utils import (
    chunk_text, embed, BM25Index, FAISSIndex, cross_encoder_rerank,
    reciprocal_rank_fusion, compress_context, deduplicate, simple_tokenise,
    sentence_split, Timer, semantic_similarity,
)
from corpus import DOCUMENTS, ALL_TEXTS


# ─────────────────────────────────────────────────────────────────────────────
# Shared result container
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RetrievalResult:
    rag_type: str
    query: str
    transformed_query: Optional[str]          # what actually hit the index
    retrieved_chunks: List[str]               # final chunks handed to "LLM"
    scores: List[float]
    latency_ms: float
    pipeline_steps: List[str]                 # human-readable audit trail
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Computed metrics ──────────────────────────────────────────────────
    def avg_score(self) -> float:
        return float(np.mean(self.scores)) if self.scores else 0.0

    def total_tokens(self) -> int:
        return sum(len(c.split()) for c in self.retrieved_chunks)

    def unique_domains(self, domain_map: Dict[str, str]) -> List[str]:
        """Domains represented in retrieved chunks."""
        domains = set()
        for chunk in self.retrieved_chunks:
            for doc_id, dom in domain_map.items():
                if chunk[:80] in "".join(t for _, _, t in ALL_TEXTS):
                    domains.add(dom)
        return sorted(domains)


# ─────────────────────────────────────────────────────────────────────────────
# Base class — builds the corpus index once
# ─────────────────────────────────────────────────────────────────────────────
class BaseRAG:
    RAG_TYPE = "Base"

    def __init__(self):
        # Build corpus: one chunk per sentence-window per document
        self.raw_chunks: List[str] = []
        self.chunk_source: List[str] = []   # doc_id for each chunk
        for doc_id, title, text in ALL_TEXTS:
            for chunk in chunk_text(text, chunk_size=3, overlap=1):
                self.raw_chunks.append(chunk)
                self.chunk_source.append(doc_id)

        self.faiss_index = FAISSIndex(self.raw_chunks)
        self.bm25_index  = BM25Index(self.raw_chunks)

        # Quick doc-text map for eval
        self.doc_text_map = {doc_id: text for doc_id, _, text in ALL_TEXTS}

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# 1. NAIVE RAG
# ─────────────────────────────────────────────────────────────────────────────
class NaiveRAG(BaseRAG):
    """
    Baseline: embed query → ANN search → return top-k chunks.
    No pre/post processing. Faithfully replicates the simplest possible pipeline.
    """
    RAG_TYPE = "Naive RAG"

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        steps = []
        with Timer() as t:
            steps.append(f"① Embed query: '{query[:60]}'")
            results = self.faiss_index.search(query, top_k)
            steps.append(f"② ANN search → {len(results)} chunks returned")

        chunks = [r[0] for r in results]
        scores = [r[1] for r in results]

        return RetrievalResult(
            rag_type=self.RAG_TYPE,
            query=query,
            transformed_query=None,
            retrieved_chunks=chunks,
            scores=scores,
            latency_ms=t.elapsed_ms,
            pipeline_steps=steps,
            metadata={"index": "FAISS cosine"},
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. ADVANCED RAG
# ─────────────────────────────────────────────────────────────────────────────
class AdvancedRAG(BaseRAG):
    """
    Pre-retrieval: query expansion (synonym injection + length enrichment)
    Post-retrieval: cross-encoder reranking, context compression, deduplication.
    """
    RAG_TYPE = "Advanced RAG"

    # Domain synonym expansions (replaces an LLM rewriter for the no-key case)
    EXPANSIONS: Dict[str, str] = {
        "diabetes":    "type 2 diabetes treatment metformin HbA1c blood glucose",
        "hypertension":"high blood pressure ACE inhibitor DASH diet management",
        "inflation":   "inflation monetary policy fiscal Keynesian Austrian economics",
        "crisis":      "financial crisis 2008 subprime mortgage recession causes",
        "climate":     "climate change greenhouse CO2 global warming IPCC",
        "kubernetes":  "kubernetes k8s container orchestration deployment pods",
        "transformer": "transformer attention BERT GPT neural network architecture",
        "crispr":      "CRISPR gene editing Cas9 genome modification clinical",
        "rag":         "retrieval augmented generation embedding vector search",
        "refund":      "refund policy return shipping order e-commerce",
        "pricing":     "pricing strategy value-based penetration cost-plus elasticity",
    }

    def _expand_query(self, query: str) -> str:
        q_lower = query.lower()
        expansions = []
        for kw, expansion in self.EXPANSIONS.items():
            if kw in q_lower:
                expansions.append(expansion)
        if expansions:
            return query + " " + " ".join(expansions)
        return query

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        steps = []
        with Timer() as t:
            # ── Pre-retrieval ─────────────────────────────────────────────
            expanded = self._expand_query(query)
            steps.append(f"① Query expansion: '{query[:50]}' → '{expanded[:80]}'")

            # Retrieve more candidates than needed (for reranking)
            candidates_dense = self.faiss_index.search(expanded, top_k * 3)
            steps.append(f"② Dense ANN search → {len(candidates_dense)} candidates")

            # ── Post-retrieval ────────────────────────────────────────────
            candidate_texts = [c[0] for c in candidates_dense]
            reranked = cross_encoder_rerank(query, candidate_texts, top_k)
            steps.append(f"③ Cross-encoder rerank → top {len(reranked)}")

            chunks_before = [r[0] for r in reranked]
            compressed = compress_context(query, chunks_before, threshold=0.28)
            steps.append(f"④ Context compression: {len(chunks_before)} → {len(compressed)} chunks")

            deduped = deduplicate(compressed)
            steps.append(f"⑤ Deduplication: {len(compressed)} → {len(deduped)} chunks")

        final = deduped[:top_k]
        final_scores = [r[1] for r in reranked[:len(final)]]

        return RetrievalResult(
            rag_type=self.RAG_TYPE,
            query=query,
            transformed_query=expanded,
            retrieved_chunks=final,
            scores=final_scores,
            latency_ms=t.elapsed_ms,
            pipeline_steps=steps,
            metadata={
                "expansion_applied": expanded != query,
                "candidates_before_rerank": len(candidates_dense),
                "after_compression": len(compressed),
                "after_dedup": len(deduped),
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. HYBRID RAG (BM25 + Dense + RRF)
# ─────────────────────────────────────────────────────────────────────────────
class HybridRAG(BaseRAG):
    """
    Combines sparse BM25 keyword search with dense vector search,
    fused via Reciprocal Rank Fusion (RRF). Best of both worlds.
    """
    RAG_TYPE = "Hybrid RAG (BM25 + Dense + RRF)"

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        steps = []
        with Timer() as t:
            bm25_results  = self.bm25_index.search(query, top_k * 2)
            steps.append(f"① BM25 sparse search → {len(bm25_results)} results")

            dense_results = self.faiss_index.search(query, top_k * 2)
            steps.append(f"② Dense vector search → {len(dense_results)} results")

            fused = reciprocal_rank_fusion([bm25_results, dense_results])
            steps.append(f"③ RRF fusion → {len(fused)} merged, ranked results")

            final = fused[:top_k]
            steps.append(f"④ Top-{top_k} selected")

        chunks = [f[0] for f in final]
        scores = [f[1] for f in final]

        return RetrievalResult(
            rag_type=self.RAG_TYPE,
            query=query,
            transformed_query=None,
            retrieved_chunks=chunks,
            scores=scores,
            latency_ms=t.elapsed_ms,
            pipeline_steps=steps,
            metadata={
                "bm25_hits": len(bm25_results),
                "dense_hits": len(dense_results),
                "rrf_unique": len(fused),
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. SELF-RAG (Retrieval Decision Gate)
# ─────────────────────────────────────────────────────────────────────────────
class SelfRAG(BaseRAG):
    """
    Emulates Self-RAG's four reflection tokens WITHOUT an LLM:
    [Retrieve]   — should we even retrieve? Heuristic classifier.
    [Relevant]   — is each retrieved chunk actually relevant?
    [Supported]  — is the answer grounded? (proxy: mean cosine to query)
    [Useful]     — overall: pass or re-retrieve with relaxed threshold.
    """
    RAG_TYPE = "Self-RAG"

    # Patterns that indicate no retrieval needed (factual, closed-world)
    NO_RETRIEVE_PATTERNS = [
        r"\bwhat\s+is\s+\d+\s*[\+\-\*/]\s*\d+\b",  # arithmetic
        r"\bhow\s+many\s+days\s+in\b",              # calendar facts
        r"\bwhat\s+year\s+(was|is|did)\b.*\b(born|died|founded)\b",
        r"\bdefine\s+\w+\b",
        r"\bwhat\s+does\s+\w+\s+stand\s+for\b",
        r"\bwhat\s+is\s+the\s+capital\s+of\b",
    ]

    RELEVANCE_THRESHOLD = 0.30
    SUPPORTED_THRESHOLD = 0.28

    def _needs_retrieval(self, query: str) -> Tuple[bool, str]:
        q = query.lower()
        for pat in self.NO_RETRIEVE_PATTERNS:
            if re.search(pat, q):
                return False, f"[No-Retrieve] matched pattern: {pat}"
        return True, "[Retrieve] query needs external knowledge"

    def _score_relevance(self, query: str, chunks: List[str]) -> List[Tuple[str, float, str]]:
        """Return (chunk, score, token) for each chunk."""
        if not chunks:
            return []
        q_vec = embed([query])[0]
        c_vecs = embed(chunks)
        results = []
        for chunk, c_vec in zip(chunks, c_vecs):
            sim = float(np.dot(q_vec, c_vec))
            token = "[Relevant]" if sim >= self.RELEVANCE_THRESHOLD else "[Irrelevant]"
            results.append((chunk, sim, token))
        return results

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        steps = []
        with Timer() as t:
            # ── Reflect: retrieve? ─────────────────────────────────────────
            needs, token = self._needs_retrieval(query)
            steps.append(f"① {token}")

            if not needs:
                elapsed_ms = (time.perf_counter() - t.start) * 1000
                return RetrievalResult(
                    rag_type=self.RAG_TYPE,
                    query=query,
                    transformed_query=None,
                    retrieved_chunks=[],
                    scores=[],
                    latency_ms=elapsed_ms,
                    pipeline_steps=steps,
                    metadata={"retrieve_token": "[No-Retrieve]", "reason": token},
                )

            # ── Retrieve initial candidates ────────────────────────────────
            candidates = self.faiss_index.search(query, top_k * 2)
            steps.append(f"② ANN search → {len(candidates)} candidates")

            # ── Reflect: relevant? ─────────────────────────────────────────
            scored = self._score_relevance(query, [c[0] for c in candidates])
            relevant = [(c, s, tk) for c, s, tk in scored if tk == "[Relevant]"]
            irrelevant_count = len(scored) - len(relevant)
            steps.append(
                f"③ Relevance gate → {len(relevant)} [Relevant], {irrelevant_count} [Irrelevant] filtered"
            )

            # If too few relevant, relax threshold and retry
            if len(relevant) < max(1, top_k // 2):
                relaxed = [(c, s, "[Relevant-relaxed]") for c, s, _ in scored if s >= self.RELEVANCE_THRESHOLD * 0.7]
                relevant = relaxed
                steps.append(f"④ Threshold relaxed → {len(relevant)} chunks accepted")
            else:
                steps.append(f"④ Threshold OK, no relaxation needed")

            final_chunks = [r[0] for r in relevant[:top_k]]
            final_scores = [r[1] for r in relevant[:top_k]]

            # ── Reflect: supported? ────────────────────────────────────────
            avg_sim = float(np.mean(final_scores)) if final_scores else 0.0
            supported_token = "[Supported]" if avg_sim >= self.SUPPORTED_THRESHOLD else "[Partially-Supported]"
            steps.append(f"⑤ {supported_token} — avg relevance {avg_sim:.3f}")

            # ── Reflect: useful? ───────────────────────────────────────────
            useful_token = "[Useful]" if final_chunks else "[Not-Useful]"
            steps.append(f"⑥ {useful_token}")

        return RetrievalResult(
            rag_type=self.RAG_TYPE,
            query=query,
            transformed_query=None,
            retrieved_chunks=final_chunks,
            scores=final_scores,
            latency_ms=t.elapsed_ms,
            pipeline_steps=steps,
            metadata={
                "retrieve_token": "[Retrieve]",
                "relevant_count": len(relevant),
                "supported_token": supported_token,
                "avg_similarity": avg_sim,
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. CORRECTIVE RAG (CRAG)
# ─────────────────────────────────────────────────────────────────────────────
class CorrectiveRAG(BaseRAG):
    """
    External quality evaluator scores retrieved docs.
    If confidence is low, triggers fallback (web search stub → BM25 widened search)
    and applies knowledge refinement (sentence-level compression).
    """
    RAG_TYPE = "Corrective RAG (CRAG)"

    CONFIDENCE_THRESHOLD = 0.35

    def _evaluator_score(self, query: str, chunks: List[str]) -> float:
        """
        Lightweight evaluator proxy.
        Production: fine-tuned T5-small classifier.
        Here: cosine sim + query token coverage.
        """
        if not chunks:
            return 0.0
        q_tokens = set(simple_tokenise(query))
        q_vec = embed([query])[0]
        c_vecs = embed(chunks)
        sims = [float(np.dot(q_vec, v)) for v in c_vecs]
        # Token coverage: what fraction of query keywords appear in any chunk
        all_chunk_tokens = set(simple_tokenise(" ".join(chunks)))
        coverage = len(q_tokens & all_chunk_tokens) / (len(q_tokens) + 1e-10)
        return 0.6 * float(np.mean(sims)) + 0.4 * coverage

    def _knowledge_refinement(self, query: str, chunks: List[str]) -> List[str]:
        """Strip irrelevant sentences from chunks."""
        return compress_context(query, chunks, threshold=0.25)

    def _fallback_search(self, query: str, top_k: int) -> Tuple[List[str], str]:
        """
        Simulates web search fallback: use BM25 with a widened term set
        and include low-ANN-rank results normally excluded.
        """
        bm25 = self.bm25_index.search(query, top_k * 3)
        dense = self.faiss_index.search(query, top_k * 3)
        fused = reciprocal_rank_fusion([bm25, dense])
        return [f[0] for f in fused[:top_k]], "BM25+Dense fallback (web search stub)"

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        steps = []
        with Timer() as t:
            # Initial retrieval
            initial = self.faiss_index.search(query, top_k)
            steps.append(f"① Initial dense retrieval → {len(initial)} chunks")

            chunks = [r[0] for r in initial]
            scores = [r[1] for r in initial]

            # Evaluator gate
            confidence = self._evaluator_score(query, chunks)
            steps.append(f"② Evaluator score: {confidence:.3f} (threshold {self.CONFIDENCE_THRESHOLD})")

            fallback_used = False
            if confidence < self.CONFIDENCE_THRESHOLD:
                steps.append("③ LOW CONFIDENCE → triggering fallback search")
                chunks, fallback_label = self._fallback_search(query, top_k)
                scores = [0.0] * len(chunks)  # fallback scores not comparable
                steps.append(f"③  Fallback: {fallback_label} → {len(chunks)} chunks")
                fallback_used = True
                # Re-score after fallback
                confidence2 = self._evaluator_score(query, chunks)
                steps.append(f"③  Post-fallback evaluator: {confidence2:.3f}")
            else:
                steps.append("③ SUFFICIENT CONFIDENCE → using initial results")

            # Knowledge refinement (always applied)
            refined = self._knowledge_refinement(query, chunks)
            steps.append(f"④ Knowledge refinement: {len(chunks)} → {len(refined)} chunks")

        final = refined[:top_k]
        return RetrievalResult(
            rag_type=self.RAG_TYPE,
            query=query,
            transformed_query=None,
            retrieved_chunks=final,
            scores=scores[:len(final)],
            latency_ms=t.elapsed_ms,
            pipeline_steps=steps,
            metadata={
                "evaluator_confidence": confidence,
                "threshold": self.CONFIDENCE_THRESHOLD,
                "fallback_triggered": fallback_used,
                "after_refinement": len(final),
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. ADAPTIVE RAG
# ─────────────────────────────────────────────────────────────────────────────
class AdaptiveRAG:
    """
    Classifies query complexity → routes to the appropriate sub-strategy:
      SIMPLE   → no retrieval (closed-world fact)
      MODERATE → single-pass dense search (Naive-like)
      COMPLEX  → multi-step hybrid with reranking (Advanced-like)
    """
    RAG_TYPE = "Adaptive RAG"

    def __init__(self):
        self.naive    = NaiveRAG()
        self.advanced = AdvancedRAG()
        self.hybrid   = HybridRAG()
        # Share the index
        self.faiss_index = self.naive.faiss_index
        self.bm25_index  = self.naive.bm25_index
        self.raw_chunks  = self.naive.raw_chunks
        self.doc_text_map = self.naive.doc_text_map

    SIMPLE_PATTERNS = [
        r"\bwhat\s+(year|date|time)\b",
        r"\bwho\s+(invented|created|founded|born|died)\b",
        r"\bwhat\s+is\s+\d+",
        r"\bdefine\s+\w+\b",
        r"\bhow\s+many\b.*\b(days|months|years)\b",
    ]
    COMPLEX_SIGNALS = [
        "compare", "vs", "versus", "difference between", "contrast",
        "analyse", "analyze", "explain how", "pros and cons",
        "advantages and disadvantages", "across", "multiple", "both",
    ]

    def _classify(self, query: str) -> Tuple[str, str]:
        q = query.lower()
        for pat in self.SIMPLE_PATTERNS:
            if re.search(pat, q):
                return "SIMPLE", f"matched simple pattern: {pat}"
        for signal in self.COMPLEX_SIGNALS:
            if signal in q:
                return "COMPLEX", f"contains complexity signal: '{signal}'"
        word_count = len(query.split())
        if word_count > 12:
            return "COMPLEX", f"long query ({word_count} words)"
        return "MODERATE", "standard single-topic lookup"

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        steps = []
        with Timer() as t:
            complexity, reason = self._classify(query)
            steps.append(f"① Query classified as {complexity}: {reason}")

            if complexity == "SIMPLE":
                steps.append("② Route → NO RETRIEVAL (closed-world fact)")
                result = RetrievalResult(
                    rag_type=self.RAG_TYPE,
                    query=query,
                    transformed_query=None,
                    retrieved_chunks=[],
                    scores=[],
                    latency_ms=0.0,
                    pipeline_steps=steps,
                    metadata={"complexity": "SIMPLE", "route": "no_retrieval"},
                )
            elif complexity == "MODERATE":
                steps.append("② Route → NAIVE single-pass dense search")
                sub = self.naive.retrieve(query, top_k)
                result = RetrievalResult(
                    rag_type=self.RAG_TYPE,
                    query=query,
                    transformed_query=sub.transformed_query,
                    retrieved_chunks=sub.retrieved_chunks,
                    scores=sub.scores,
                    latency_ms=0.0,
                    pipeline_steps=steps + [f"   [{s}]" for s in sub.pipeline_steps],
                    metadata={"complexity": "MODERATE", "route": "naive_dense"},
                )
            else:  # COMPLEX
                steps.append("② Route → HYBRID+ADVANCED (multi-step)")
                sub = self.hybrid.retrieve(query, top_k)
                result = RetrievalResult(
                    rag_type=self.RAG_TYPE,
                    query=query,
                    transformed_query=sub.transformed_query,
                    retrieved_chunks=sub.retrieved_chunks,
                    scores=sub.scores,
                    latency_ms=0.0,
                    pipeline_steps=steps + [f"   [{s}]" for s in sub.pipeline_steps],
                    metadata={"complexity": "COMPLEX", "route": "hybrid_advanced"},
                )
        result.latency_ms = t.elapsed_ms
        return result


# ─────────────────────────────────────────────────────────────────────────────
# 7. GRAPH RAG
# ─────────────────────────────────────────────────────────────────────────────
class GraphRAG:
    """
    Builds a knowledge graph from the corpus:
      - Nodes: entities (domain, document, key noun-phrases)
      - Edges: co-occurrence within sentences + semantic similarity above threshold

    At query time: identify seed nodes → traverse neighbours → collect
    associated text chunks → rerank by relevance.

    Demonstrates multi-hop relational retrieval.
    """
    RAG_TYPE = "Graph RAG"

    EDGE_SIM_THRESHOLD = 0.60

    def __init__(self):
        self.G = nx.DiGraph()
        self.node_text: Dict[str, str]  = {}  # node_id → representative text
        self.doc_text_map = {doc_id: text for doc_id, _, text in ALL_TEXTS}
        self._build_graph()
        # Also keep a flat FAISS index for seed-node lookup
        self.faiss_index = FAISSIndex(
            [text for _, _, text in ALL_TEXTS]
        )
        self.doc_ids = [doc_id for doc_id, _, _ in ALL_TEXTS]

    def _extract_entities(self, text: str) -> List[str]:
        """
        Lightweight entity extraction: capitalised noun phrases + domain keywords.
        Production: spaCy NER. Here: regex over title-case runs + known keywords.
        """
        entities = []
        # Title-case phrases (often named entities)
        for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text):
            phrase = m.group(1)
            if len(phrase) > 3:
                entities.append(phrase.lower())
        # Key domain terms
        keywords = [
            "diabetes", "metformin", "insulin", "hypertension", "covid",
            "inflation", "keynesian", "austrian", "subprime", "mortgage",
            "transformer", "attention", "bert", "kubernetes", "container",
            "crispr", "genome", "cas9", "climate", "co2", "greenhouse",
            "rag", "retrieval", "embedding", "vector", "refund", "pricing",
        ]
        t_lower = text.lower()
        for kw in keywords:
            if kw in t_lower:
                entities.append(kw)
        return list(set(entities))

    def _build_graph(self):
        # Add document nodes
        doc_entities: Dict[str, List[str]] = {}
        for doc_id, title, text in ALL_TEXTS:
            self.G.add_node(doc_id, type="document", title=title, domain=next(
                d["domain"] for d in DOCUMENTS if d["id"] == doc_id
            ))
            self.node_text[doc_id] = text
            entities = self._extract_entities(text)
            doc_entities[doc_id] = entities

            # Add entity nodes and doc→entity edges
            for ent in entities:
                ent_id = f"ent:{ent}"
                if not self.G.has_node(ent_id):
                    self.G.add_node(ent_id, type="entity", label=ent)
                    self.node_text[ent_id] = ent
                self.G.add_edge(doc_id, ent_id, weight=1.0, rel="mentions")

        # Cross-document edges: if two docs share entities → connect
        doc_list = list(doc_entities.keys())
        for i in range(len(doc_list)):
            for j in range(i + 1, len(doc_list)):
                shared = set(doc_entities[doc_list[i]]) & set(doc_entities[doc_list[j]])
                if len(shared) >= 2:
                    weight = len(shared) / 10.0
                    self.G.add_edge(doc_list[i], doc_list[j], weight=weight, rel="co-mentions", shared=list(shared)[:5])
                    self.G.add_edge(doc_list[j], doc_list[i], weight=weight, rel="co-mentions", shared=list(shared)[:5])

        # Semantic similarity edges between document nodes
        doc_texts  = [self.doc_text_map[d] for d in doc_list]
        doc_vecs   = embed(doc_texts)
        for i in range(len(doc_list)):
            for j in range(i + 1, len(doc_list)):
                sim = float(np.dot(doc_vecs[i], doc_vecs[j]))
                if sim >= self.EDGE_SIM_THRESHOLD:
                    self.G.add_edge(doc_list[i], doc_list[j], weight=sim, rel="semantic")
                    self.G.add_edge(doc_list[j], doc_list[i], weight=sim, rel="semantic")

    def _find_seed_nodes(self, query: str, top_n: int = 3) -> List[str]:
        """Map query to top-N document nodes via dense search."""
        results = self.faiss_index.search(query, top_n)
        # results are over full doc texts; map back to doc_ids by index
        q_vec = embed([query])[0]
        doc_texts = [self.doc_text_map[d] for d in self.doc_ids]
        vecs = embed(doc_texts)
        sims = [(self.doc_ids[i], float(np.dot(q_vec, vecs[i]))) for i in range(len(self.doc_ids))]
        sims.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in sims[:top_n]]

    def _traverse(self, seed_nodes: List[str], hops: int = 2) -> List[str]:
        """BFS from seed nodes up to `hops` edges, collect document nodes."""
        visited_docs = set(seed_nodes)
        frontier = set(seed_nodes)
        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                for neighbour in self.G.successors(node):
                    if self.G.nodes[neighbour].get("type") == "document" and neighbour not in visited_docs:
                        visited_docs.add(neighbour)
                        next_frontier.add(neighbour)
            frontier = next_frontier
        return list(visited_docs)

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        steps = []
        with Timer() as t:
            # Seed nodes
            seeds = self._find_seed_nodes(query, top_n=2)
            steps.append(f"① Seed nodes: {seeds}")

            # Graph traversal (2 hops)
            traversed = self._traverse(seeds, hops=2)
            steps.append(f"② Traversal (2-hop) → {len(traversed)} document nodes reached")

            # Collect and rank texts from traversed nodes
            candidate_texts = []
            candidate_ids   = []
            for node in traversed:
                text = self.node_text.get(node, "")
                if text and len(text) > 50:
                    candidate_texts.append(text)
                    candidate_ids.append(node)

            steps.append(f"③ Collected {len(candidate_texts)} candidate document texts")

            # Chunk and rerank
            all_chunks = []
            for text in candidate_texts:
                all_chunks.extend(chunk_text(text, chunk_size=3, overlap=1))

            all_chunks = list(set(all_chunks))
            reranked = cross_encoder_rerank(query, all_chunks, top_k)
            steps.append(f"④ Reranked {len(all_chunks)} chunks → top {len(reranked)}")

            # Graph metadata: what edges were traversed?
            edge_types = {}
            for u, v, data in self.G.edges(data=True):
                rel = data.get("rel", "unknown")
                edge_types[rel] = edge_types.get(rel, 0) + 1
            steps.append(f"⑤ Graph stats: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges | edge types: {edge_types}")

        chunks = [r[0] for r in reranked]
        scores = [r[1] for r in reranked]

        return RetrievalResult(
            rag_type=self.RAG_TYPE,
            query=query,
            transformed_query=None,
            retrieved_chunks=chunks,
            scores=scores,
            latency_ms=t.elapsed_ms,
            pipeline_steps=steps,
            metadata={
                "seed_nodes": seeds,
                "traversed_docs": traversed,
                "graph_nodes": self.G.number_of_nodes(),
                "graph_edges": self.G.number_of_edges(),
                "edge_types": edge_types,
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# RAG Registry
# ─────────────────────────────────────────────────────────────────────────────
def build_all_rags() -> Dict[str, Any]:
    """Instantiate all RAG types. Shared models loaded once."""
    return {
        "1_naive":     NaiveRAG(),
        "2_advanced":  AdvancedRAG(),
        "3_hybrid":    HybridRAG(),
        "4_self":      SelfRAG(),
        "5_corrective": CorrectiveRAG(),
        "6_adaptive":  AdaptiveRAG(),
        "7_graph":     GraphRAG(),
    }