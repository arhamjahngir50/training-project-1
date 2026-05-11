"""
RAG Test Runner — No LLM Required
Runs a structured benchmark across all 7 RAG implementations.
Usage:
    python main.py                  # run full benchmark
    python main.py --rag naive      # run single RAG type
    python main.py --query "your question here"  # custom query
"""

from __future__ import annotations
import sys, argparse, textwrap, time
from typing import List, Dict, Any

# ── Pretty printing helpers ───────────────────────────────────────────────────
BOLD  = "\033[1m"
GREEN = "\033[92m"
CYAN  = "\033[96m"
YELLOW= "\033[93m"
RED   = "\033[91m"
DIM   = "\033[2m"
RESET = "\033[0m"

def hr(char="─", width=80): print(char * width)
def header(text): print(f"\n{BOLD}{CYAN}{text}{RESET}")
def subheader(text): print(f"{BOLD}{YELLOW}{text}{RESET}")

# ── Test queries: one per domain + cross-domain + edge cases ─────────────────
TEST_QUERIES = [
    # Single-domain lookups
    ("medical",    "What is the first-line treatment for type 2 diabetes?"),
    ("medical",    "How is hypertension managed in elderly patients?"),
    ("medical",    "What causes severe COVID-19 and how is it treated?"),
    ("finance",    "What caused the 2008 financial crisis?"),
    ("finance",    "How do Keynesian and Austrian economists view inflation differently?"),
    ("finance",    "How does venture capital funding work for startups?"),
    ("technology", "How does the Transformer architecture work?"),
    ("technology", "What are the core components of Kubernetes?"),
    ("technology", "What is retrieval augmented generation and how does it work?"),
    ("history",    "What was the turning point in the Pacific War?"),
    ("history",    "How did the Industrial Revolution change the economy?"),
    ("science",    "How does CRISPR-Cas9 gene editing work?"),
    ("science",    "What mechanisms drive climate change?"),
    ("business",   "What is the refund policy for international orders?"),
    ("business",   "What pricing strategies are used in e-commerce?"),

    # Cross-domain / complex
    ("cross",      "Compare the economic causes and effects of the 2008 crisis versus the Industrial Revolution"),
    ("cross",      "How do RAG systems relate to transformer architectures?"),
    ("cross",      "What are the pros and cons of CRISPR versus traditional medical treatments like metformin?"),

    # Edge cases for Self-RAG / Adaptive no-retrieval gate
    ("edge",       "What is 42 + 58?"),
    ("edge",       "Define the term RAG"),
    ("edge",       "What is the capital of France?"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Single-result printer
# ─────────────────────────────────────────────────────────────────────────────
def print_result(result, show_chunks: bool = True, max_chunk_words: int = 40):
    print(f"\n  {BOLD}RAG type   :{RESET} {result.rag_type}")
    print(f"  {BOLD}Query      :{RESET} {result.query}")
    if result.transformed_query and result.transformed_query != result.query:
        tq = result.transformed_query[:120] + ("…" if len(result.transformed_query) > 120 else "")
        print(f"  {BOLD}Transformed:{RESET} {DIM}{tq}{RESET}")
    print(f"  {BOLD}Latency    :{RESET} {result.latency_ms:.1f} ms")
    print(f"  {BOLD}Chunks     :{RESET} {len(result.retrieved_chunks)}  |  "
          f"{BOLD}Avg score:{RESET} {result.avg_score():.4f}  |  "
          f"{BOLD}Tokens:{RESET} ~{result.total_tokens()}")

    print(f"\n  {BOLD}Pipeline steps:{RESET}")
    for step in result.pipeline_steps:
        print(f"    {DIM}{step}{RESET}")

    if show_chunks and result.retrieved_chunks:
        print(f"\n  {BOLD}Retrieved chunks (truncated):{RESET}")
        for i, (chunk, score) in enumerate(zip(result.retrieved_chunks, result.scores), 1):
            words = chunk.split()
            preview = " ".join(words[:max_chunk_words])
            if len(words) > max_chunk_words:
                preview += " …"
            print(f"    {GREEN}[{i}] score={score:.4f}{RESET}  {preview}")
    elif not result.retrieved_chunks:
        print(f"  {YELLOW}⚠  No chunks retrieved (no-retrieval path taken){RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Comparison table across all RAGs for a single query
# ─────────────────────────────────────────────────────────────────────────────
def print_comparison_table(query: str, results: Dict[str, Any]):
    header(f"COMPARISON TABLE — '{query[:70]}'")
    hr()
    col = "{:<28} {:>8} {:>10} {:>8} {:>10}"
    print(col.format("RAG Type", "Latency", "Avg Score", "Chunks", "Tokens"))
    hr("·")
    for key in sorted(results):
        r = results[key]
        name = r.rag_type[:27]
        print(col.format(
            name,
            f"{r.latency_ms:.0f} ms",
            f"{r.avg_score():.4f}",
            str(len(r.retrieved_chunks)),
            f"~{r.total_tokens()}",
        ))
    hr()


# ─────────────────────────────────────────────────────────────────────────────
# Full benchmark
# ─────────────────────────────────────────────────────────────────────────────
def run_benchmark(rags: Dict, queries: List, verbose: bool = False):
    summary_rows = []

    for domain, query in queries:
        header(f"[{domain.upper()}]  {query}")
        hr()
        row = {"domain": domain, "query": query}
        query_results = {}

        for rag_key in sorted(rags):
            rag = rags[rag_key]
            try:
                result = rag.retrieve(query, top_k=5)
                query_results[rag_key] = result
                row[rag_key] = {
                    "latency_ms": result.latency_ms,
                    "avg_score":  result.avg_score(),
                    "n_chunks":   len(result.retrieved_chunks),
                    "tokens":     result.total_tokens(),
                }
                if verbose:
                    print_result(result, show_chunks=True)
                else:
                    # Brief one-liner
                    status = f"{len(result.retrieved_chunks)} chunks, score={result.avg_score():.4f}, {result.latency_ms:.0f}ms"
                    print(f"  {DIM}{result.rag_type:<35}{RESET}  {status}")
            except Exception as e:
                print(f"  {RED}ERROR in {rag_key}: {e}{RESET}")
                row[rag_key] = {"error": str(e)}

        if not verbose:
            print_comparison_table(query, query_results)

        summary_rows.append(row)

    return summary_rows


# ─────────────────────────────────────────────────────────────────────────────
# Single RAG deep-dive
# ─────────────────────────────────────────────────────────────────────────────
def run_single_rag(rags: Dict, rag_name: str, queries: List):
    rag_map = {
        "naive":      "1_naive",
        "advanced":   "2_advanced",
        "hybrid":     "3_hybrid",
        "self":       "4_self",
        "corrective": "5_corrective",
        "adaptive":   "6_adaptive",
        "graph":      "7_graph",
    }
    key = rag_map.get(rag_name.lower())
    if key not in rags:
        print(f"{RED}Unknown RAG '{rag_name}'. Options: {list(rag_map)}{RESET}")
        return

    rag = rags[key]
    header(f"DEEP DIVE — {rag.RAG_TYPE}")
    for domain, query in queries:
        hr("·")
        result = rag.retrieve(query, top_k=5)
        print_result(result, show_chunks=True)


# ─────────────────────────────────────────────────────────────────────────────
# Custom query mode
# ─────────────────────────────────────────────────────────────────────────────
def run_custom_query(rags: Dict, query: str):
    header(f"CUSTOM QUERY: '{query}'")
    results = {}
    for key in sorted(rags):
        rag = rags[key]
        result = rag.retrieve(query, top_k=5)
        results[key] = result
        print_result(result, show_chunks=True)
        hr("·")
    print_comparison_table(query, results)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation summary (precision + MRR using utils helpers)
# ─────────────────────────────────────────────────────────────────────────────
# Maps test query keywords → expected source doc_id (ground truth)
GROUND_TRUTH: Dict[str, str] = {
    "type 2 diabetes":           "med_001",
    "hypertension":              "med_002",
    "covid":                     "med_003",
    "2008 financial crisis":     "fin_001",
    "keynesian":                 "fin_002",
    "venture capital":           "fin_003",
    "transformer architecture":  "tech_001",
    "kubernetes":                "tech_002",
    "retrieval augmented":       "tech_003",
    "pacific war":               "hist_001",
    "industrial revolution":     "hist_002",
    "crispr":                    "sci_001",
    "climate change":            "sci_002",
    "refund policy":             "biz_001",
    "pricing strateg":           "biz_002",
}

def get_ground_truth(query: str) -> str | None:
    q = query.lower()
    for kw, doc_id in GROUND_TRUTH.items():
        if kw in q:
            return doc_id
    return None

def run_evaluation(rags: Dict, queries: List):
    from utils import context_precision, mean_reciprocal_rank, semantic_similarity

    header("EVALUATION SUMMARY (Context Precision, MRR, Semantic Similarity)")
    hr()

    corpus_map = {doc_id: text for doc_id, _, text in __import__("corpus").ALL_TEXTS}

    # Aggregate per RAG
    agg: Dict[str, Dict[str, list]] = {k: {"cp": [], "mrr": [], "ss": []} for k in rags}

    for domain, query in queries:
        gt = get_ground_truth(query)
        if gt is None:
            continue  # skip cross-domain / edge for eval

        for key, rag in rags.items():
            result = rag.retrieve(query, top_k=5)
            chunks = result.retrieved_chunks
            cp  = context_precision(chunks, gt, corpus_map)
            mrr = mean_reciprocal_rank(chunks, gt, corpus_map)
            ss  = semantic_similarity(query, chunks)
            agg[key]["cp"].append(cp)
            agg[key]["mrr"].append(mrr)
            agg[key]["ss"].append(ss)

    import numpy as np
    col = "{:<35} {:>12} {:>10} {:>14}"
    print(col.format("RAG Type", "Context Prec", "MRR", "Semantic Sim"))
    hr("·")

    for key in sorted(agg):
        vals = agg[key]
        if not vals["cp"]:
            continue
        cp  = float(np.mean(vals["cp"]))
        mrr = float(np.mean(vals["mrr"]))
        ss  = float(np.mean(vals["ss"]))
        name = rags[key].RAG_TYPE[:34]
        print(col.format(name, f"{cp:.4f}", f"{mrr:.4f}", f"{ss:.4f}"))
    hr()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="RAG Benchmark Runner — no LLM required",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python main.py                          # full benchmark, summary mode
              python main.py --verbose                # full benchmark, show all chunks
              python main.py --rag naive              # deep-dive single RAG
              python main.py --rag graph              # deep-dive Graph RAG
              python main.py --query "how does CRISPR work?"   # custom query
              python main.py --eval                   # evaluation metrics only
              python main.py --domain medical         # only medical queries
        """),
    )
    parser.add_argument("--rag",     type=str, help="Run a single RAG type: naive|advanced|hybrid|self|corrective|adaptive|graph")
    parser.add_argument("--query",   type=str, help="Run a custom query against all RAGs")
    parser.add_argument("--verbose", action="store_true", help="Show full chunk content in benchmark mode")
    parser.add_argument("--eval",    action="store_true", help="Print evaluation metrics table")
    parser.add_argument("--domain",  type=str, help="Filter queries to a specific domain")
    args = parser.parse_args()

    # ── Load RAGs (this takes ~10–30s for model + index init) ─────────────
    print(f"{BOLD}Loading RAG systems…{RESET}  (sentence-transformers + FAISS, ~10–30s first run)")
    t0 = time.perf_counter()

    from rags import build_all_rags
    rags = build_all_rags()

    elapsed = time.perf_counter() - t0
    print(f"{GREEN}✓ All RAGs ready in {elapsed:.1f}s{RESET}\n")

    # ── Filter queries if --domain set ────────────────────────────────────
    queries = TEST_QUERIES
    if args.domain:
        queries = [(d, q) for d, q in TEST_QUERIES if d == args.domain.lower()]
        if not queries:
            print(f"{RED}No queries found for domain '{args.domain}'.{RESET}")
            print(f"Available domains: {sorted(set(d for d, _ in TEST_QUERIES))}")
            sys.exit(1)

    # ── Dispatch ──────────────────────────────────────────────────────────
    if args.query:
        run_custom_query(rags, args.query)
    elif args.rag:
        run_single_rag(rags, args.rag, queries)
    elif args.eval:
        run_evaluation(rags, queries)
    else:
        run_benchmark(rags, queries, verbose=args.verbose)
        if not args.verbose:
            print(f"\n{DIM}Tip: add --verbose to see full chunk content, or --eval for metrics.{RESET}\n")


if __name__ == "__main__":
    main()