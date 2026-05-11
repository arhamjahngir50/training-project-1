# RAG Benchmark Suite — Production Implementations (No LLM Required)

A comprehensive test harness for 7 retrieval-augmented generation (RAG) implementations, each with distinct retrieval strategies and no LLM dependency.

---

## Overview

This project benchmarks seven RAG types across multiple domains (medical, finance, technology, history, science, business). Each implementation:
- Uses **sentence-transformers embeddings** for semantic search
- Operates **entirely without LLM** (no API keys needed)
- Includes **diagnostic metadata** (pipeline steps, scores, latency)
- Returns structured `RetrievalResult` objects for evaluation

### File Structure
```
rag-types-testing/
├── main.py           # Entry point, CLI runner
├── rags.py           # All 7 RAG implementations + RetrievalResult
├── utils.py          # Embedding, indexing, reranking, evaluation metrics
├── corpus.py         # Multi-domain knowledge base
└── README.md         # This file
```

---

## RAG Implementations & Logic

### 1. **Naive RAG**
**Class:** `NaiveRAG`

**Logic:**
- Embed the query using sentence-transformers
- Search FAISS index with cosine similarity
- Return top-k chunks as-is

**Pipeline:**
```
① Embed query
② ANN search → return top-k chunks
```

**Pros:** Fast, simple baseline  
**Cons:** No relevance filtering, no reranking, no quality checks  
**Use case:** Baseline for comparison

---

### 2. **Advanced RAG**
**Class:** `AdvancedRAG`

**Logic:**
1. **Pre-retrieval:** Query expansion using domain-specific synonyms
2. **Retrieval:** Dense ANN search with expanded query
3. **Post-retrieval:** Cross-encoder reranking, context compression, deduplication

**Pipeline:**
```
① Query expansion (synonym injection)
② Dense ANN search → 3x candidates
③ Cross-encoder rerank → top-k
④ Context compression (sentence-level filtering)
⑤ Deduplication (remove near-duplicates)
```

**Pros:** Better relevance through expansion + reranking + compression  
**Cons:** Slower; requires domain synonyms dict  
**Use case:** High-quality single-topic queries

---

### 3. **Hybrid RAG** (BM25 + Dense + RRF)
**Class:** `HybridRAG`

**Logic:**
1. Run **BM25 keyword search** (sparse, lexical)
2. Run **dense vector search** (semantic, embedding-based)
3. Fuse results using **Reciprocal Rank Fusion (RRF)**

**Pipeline:**
```
① BM25 sparse search → 2x candidates
② Dense vector search → 2x candidates
③ RRF fusion (reciprocal rank combination)
④ Top-k selected
```

**RRF Formula:**
```
score(d) = sum over all rankings: 1 / (k + rank(d))
```

**Pros:** Combines lexical + semantic; robust to both keyword and meaning-based queries  
**Cons:** Higher latency (two indexes)  
**Use case:** Mixed-vocabulary queries, cross-domain retrieval

---

### 4. **Self-RAG** (Reflection Tokens)
**Class:** `SelfRAG`

**Logic:**
Emulates Self-RAG's four reflection tokens without an LLM:

1. **[Retrieve]** — Should we even retrieve?
   - Regex patterns detect closed-world facts (math, definitions, capital cities)
   - If matched → skip retrieval entirely
   - Otherwise → proceed to ANN search

2. **[Relevant]** — Is each chunk actually relevant?
   - Embed query and candidates
   - Compute cosine similarity
   - Label `[Relevant]` if similarity >= 0.30, else `[Irrelevant]`
   - Filter irrelevant chunks

3. **Threshold Relaxation**
   - If too few relevant chunks remain, lower threshold to 0.7 × 0.30
   - Accept `[Relevant-relaxed]` chunks

4. **[Supported]** — Is the answer grounded?
   - Compute average similarity of final chunks
   - If avg_sim >= 0.28 → `[Supported]`, else `[Partially-Supported]`

5. **[Useful]** — Overall pass/fail?
   - If chunks exist → `[Useful]`, else `[Not-Useful]`

**Pipeline:**
```
① [Retrieve] decision (heuristic check)
② ANN search → candidates
③ Relevance gate → [Relevant] / [Irrelevant]
④ Threshold OK or relaxed
⑤ [Supported] / [Partially-Supported] (avg similarity check)
⑥ [Useful] / [Not-Useful]
```

**Pros:** Adaptive no-retrieval gate; explicit quality signals  
**Cons:** Regex-based patterns may miss some queries; heuristic-heavy  
**Use case:** Mixed queries where some need no retrieval

---

### 5. **Corrective RAG (CRAG)**
**Class:** `CorrectiveRAG`

**Logic:**
1. **Initial retrieval** → dense ANN search
2. **Evaluator gate** → score confidence using:
   - Cosine similarity (60% weight)
   - Query token coverage in chunks (40% weight)
3. **Fallback trigger** → if confidence < 0.35:
   - Run BM25 + Dense fusion (fallback search)
   - Treat result as "web search stub"
4. **Knowledge refinement** → always compress chunks (sentence-level)

**Pipeline:**
```
① Initial dense retrieval
② Evaluator score (hybrid: sim + coverage)
③ LOW CONFIDENCE → fallback (BM25+Dense fusion)
   OR SUFFICIENT CONFIDENCE → use initial
④ Knowledge refinement (context compression)
```

**Pros:** Graceful fallback for low-confidence cases; quality scoring  
**Cons:** Added latency from evaluator + potential fallback  
**Use case:** Production systems needing confidence-aware retrieval

---

### 6. **Adaptive RAG**
**Class:** `AdaptiveRAG`

**Logic:**
Query complexity classifier → route to appropriate sub-strategy:

1. **SIMPLE queries** (regex patterns)
   - "What is 2+2?", "Define X", "What year was Y born?"
   - Action: **No retrieval** (return empty)

2. **MODERATE queries**
   - Standard single-topic lookups
   - Action: Route to **NaiveRAG** (single-pass dense search)

3. **COMPLEX queries**
   - "Compare X vs Y", "Pros and cons of Z", "Analyze how..."
   - Long queries (>12 words)
   - Action: Route to **HybridRAG** (multi-step BM25 + dense + RRF)

**Pipeline:**
```
① Classify query complexity
② Route to appropriate strategy:
   SIMPLE   → no retrieval
   MODERATE → NaiveRAG
   COMPLEX  → HybridRAG
```

**Pros:** Efficient routing; avoids unnecessary processing  
**Cons:** Classification heuristics may mislabel  
**Use case:** Production systems with varied query types

---

### 7. **Graph RAG**
**Class:** `GraphRAG`

**Logic:**
Builds a knowledge graph and performs multi-hop retrieval:

1. **Graph Construction:**
   - **Nodes:** Document nodes + entity nodes (capitalised phrases + domain keywords)
   - **Edges:** 
     - Document → Entity (mentions)
     - Document ↔ Document (co-mention: share ≥2 entities)
     - Document ↔ Document (semantic similarity ≥0.60)

2. **Seed Node Identification:**
   - Map query to top-2 document nodes via dense search

3. **Graph Traversal:**
   - 2-hop BFS from seed nodes
   - Collect all document nodes reachable

4. **Reranking:**
   - Extract chunks from traversed documents
   - Cross-encoder rerank → top-k

**Pipeline:**
```
① Seed nodes (top-2 documents)
② Traversal (2-hop BFS)
③ Collect candidate document texts
④ Chunk & rerank (cross-encoder)
⑤ Graph stats (nodes, edges, edge types)
```

**Pros:** Multi-hop reasoning; captures document relationships  
**Cons:** Slow (graph construction + traversal + reranking); memory overhead  
**Use case:** Complex multi-document reasoning

---

## Installation

```bash
cd rag-types-testing
pip install -r requirements.txt
```

**Requirements:**
- `sentence-transformers` (embeddings)
- `faiss-cpu` (dense index)
- `rank-bm25` (sparse index)
- `networkx` (for Graph RAG)
- `numpy`

---

## Running the Code

### Full Benchmark (Summary Mode)
```bash
python main.py
```
Shows one-line stats per RAG type across all test queries.

### Full Benchmark (Verbose Mode)
```bash
python main.py --verbose
```
Shows full chunk content for each RAG.

### Single RAG Deep-Dive
```bash
python main.py --rag naive
python main.py --rag advanced
python main.py --rag hybrid
python main.py --rag self
python main.py --rag corrective
python main.py --rag adaptive
python main.py --rag graph
```
Runs a single RAG on all test queries with full diagnostic output.

### Custom Query
```bash
python main.py --query "How does CRISPR gene editing work?"
```
Runs a single custom query against all RAG types.

### Filter by Domain
```bash
python main.py --domain medical
python main.py --domain finance
python main.py --domain technology
python main.py --domain history
python main.py --domain science
python main.py --domain business
python main.py --domain cross
python main.py --domain edge
```
Run only queries from a specific domain.

### Evaluation Metrics Only
```bash
python main.py --eval
```
Prints context precision, MRR, and semantic similarity scores.

---

## Output Format

### Pipeline Steps
Each result includes an audit trail:
```
Pipeline steps:
    ① [Retrieve] query needs external knowledge
    ② ANN search → 10 candidates
    ③ Relevance gate → 5 [Relevant], 5 [Irrelevant] filtered
    ④ Threshold OK, no relaxation needed
    ⑤ [Supported] — avg relevance 0.405
    ⑥ [Useful]
```

### Retrieved Chunks
```
Retrieved chunks (truncated):
    [1] score=0.5168  Base editing and prime editing offer single-nucleotide precision …
    [2] score=0.4814  CRISPR-Cas9 is a bacterial immune system adapted for precision …
    [3] score=0.3806  Cas9 protein binds the gRNA, scans the genome …
```

### Comparison Table
```
────────────────────────────────────────────────────────────────────────────
RAG Type                      Latency    Avg Score    Chunks      Tokens
············································································
1_naive                         120 ms       0.4521        5          ~89
2_advanced                      350 ms       0.5112        5          ~78
3_hybrid                        200 ms       0.4834        5          ~92
4_self                           98 ms       0.3890        3          ~65
5_corrective                    280 ms       0.5034        5          ~81
6_adaptive                      110 ms       0.4678        5          ~88
7_graph                        1250 ms       0.5301        5          ~95
────────────────────────────────────────────────────────────────────────────
```

---

## Evaluation Metrics

The `--eval` flag computes three metrics per RAG:

1. **Context Precision (CP)**
   - Fraction of retrieved chunks from the ground-truth document
   - Range: 0–1 (higher is better)

2. **Mean Reciprocal Rank (MRR)**
   - 1 / rank of first relevant chunk
   - Range: 0–1 (higher is better)

3. **Semantic Similarity (SS)**
   - Average cosine similarity between query and retrieved chunks
   - Range: -1–1 (higher is better)

---

## Test Corpus

The corpus (`corpus.py`) includes:
- **Medical:** Type 2 diabetes, hypertension, COVID-19
- **Finance:** 2008 crisis, Keynesian vs Austrian economics, VC funding
- **Technology:** Transformers, Kubernetes, RAG
- **History:** Pacific War, Industrial Revolution
- **Science:** CRISPR, climate change
- **Business:** Refund policies, pricing strategies

---

## Key Parameters

| Parameter | Default | Location |
|-----------|---------|----------|
| `top_k` | 5 | All RAG classes |
| `chunk_size` | 3 (sentences) | `utils.chunk_text()` |
| `chunk_overlap` | 1 | `utils.chunk_text()` |
| Self-RAG `RELEVANCE_THRESHOLD` | 0.30 | `SelfRAG` |
| Self-RAG `SUPPORTED_THRESHOLD` | 0.28 | `SelfRAG` |
| Corrective-RAG `CONFIDENCE_THRESHOLD` | 0.35 | `CorrectiveRAG` |
| Graph-RAG `EDGE_SIM_THRESHOLD` | 0.60 | `GraphRAG` |
| Graph-RAG `hops` | 2 | `GraphRAG._traverse()` |

---

## Performance Tips

1. **First run is slow** (~10–30s): Embeddings model + FAISS index initialization
2. **Bottleneck:** Cross-encoder reranking (most expensive step)
3. **Fastest:** Naive RAG, Adaptive RAG (simple paths)
4. **Slowest:** Graph RAG (graph traversal + reranking)

---

## Example Usage Flow

```bash
# 1. Run full benchmark
python main.py

# 2. Deep-dive on Self-RAG to understand decisions
python main.py --rag self

# 3. Test custom medical query against all RAGs
python main.py --query "What are the side effects of metformin?"

# 4. Run evaluation metrics
python main.py --eval

# 5. Filter to only finance queries
python main.py --domain finance --verbose
```

---

## Extending the Code

### Add a New RAG Type
1. Create a class inheriting from `BaseRAG`
2. Implement `retrieve(query: str, top_k: int) -> RetrievalResult`
3. Register in `build_all_rags()` dict
4. Add CLI option in `main.py`

### Add Custom Queries
Edit `TEST_QUERIES` in `main.py`:
```python
TEST_QUERIES = [
    ("medical", "Your custom query here?"),
    # ...
]
```

### Add Evaluation Metrics
Add functions to `utils.py` and call from `run_evaluation()` in `main.py`.

---

## References

- Self-RAG: [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)
- Corrective RAG: [arXiv:2401.15884](https://arxiv.org/abs/2401.15884)
- Adaptive RAG concepts
- Graph RAG: Multi-hop retrieval patterns
- Reciprocal Rank Fusion: [Cormack et al. 2009](https://plg.uwaterloo.ca/~gvcormac/cormacksigirforum09.pdf)

---

**Last Updated:** May 2026  
**Status:** All 7 RAG types tested and benchmarked
