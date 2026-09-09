"""
Experiment 03 — Retrieval benchmark with OpenRouter + fastembed.
mem0ai v2.0.11 — captures memory IDs at ingest time for accurate mapping.

Run: python experiments/03_openrouter_benchmark.py
"""

import time
from mem0 import Memory

OPENROUTER_API_KEY = "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"
USER = "benchmark_sadik"

CONFIG = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "openai/gpt-4o-mini",
            "api_key": OPENROUTER_API_KEY,
            "openai_base_url": "https://openrouter.ai/api/v1",
        },
    },
    "embedder": {
        "provider": "fastembed",
        "config": {"model": "BAAI/bge-small-en-v1.5"},
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "openrouter_benchmark",
            "embedding_model_dims": 384,
            "on_disk": False,
        },
    },
    "version": "v1.1",
}

# Ground truth facts
FACTS = [
    "Sadik uses PyTorch for deep learning experiments.",
    "The dataset used is the LOCOMO long-term conversation benchmark.",
    "The baseline model achieves 72.3% recall@5 on LongMemEval.",
    "Transformer-based memory retrieval outperforms BM25 by 14% on the thesis dataset.",
    "The thesis proposes a hybrid scoring method combining dense vectors and sparse BM25.",
    "Graph-augmented memory shows 8% improvement over flat vector retrieval.",
    "Experiment runs use an NVIDIA A100 80GB GPU.",
    "The evaluation metrics are Recall@K and MRR (Mean Reciprocal Rank).",
]

# Queries mapped to the fact index they should retrieve
QUERIES = [
    ("What deep learning framework is used?",               0),
    ("Which benchmark dataset is used for evaluation?",     1),
    ("What is the baseline recall performance?",            2),
    ("How much better is transformer retrieval vs BM25?",   3),
    ("What is the proposed method in the thesis?",          4),
    ("What improvement does graph memory provide?",         5),
    ("What GPU is used for experiments?",                   6),
    ("What evaluation metrics are reported?",               7),
]

def precision_at_k(retrieved: list, relevant_set: set, k: int) -> float:
    return 1.0 if any(r in relevant_set for r in retrieved[:k]) else 0.0

def mrr(retrieved: list, relevant_set: set) -> float:
    for rank, idx in enumerate(retrieved, 1):
        if idx in relevant_set:
            return 1.0 / rank
    return 0.0

print("=" * 65)
print("mem0 v2 Retrieval Benchmark — OpenRouter + fastembed")
print("=" * 65)
print("LLM:      openai/gpt-4o-mini via OpenRouter")
print("Embedder: BAAI/bge-small-en-v1.5 (local, 384-dim)")
print()

m = Memory.from_config(CONFIG)
m.delete_all(user_id=USER)  # clean slate

# ── Ingest: capture IDs returned by add() at index time ──────────────────────
# mem0 may split one fact into multiple memories, or deduplicate — we map ALL
# returned IDs back to their source fact index.
print(f"Indexing {len(FACTS)} facts into mem0...")
fact_idx_to_ids: dict[int, set[str]] = {}   # fact_index → set of memory IDs
id_to_fact_idx: dict[str, int] = {}          # memory_id  → fact_index

t_ingest = time.perf_counter()
for i, fact in enumerate(FACTS):
    result = m.add(fact, user_id=USER)
    ids = {r["id"] for r in result.get("results", [])}
    fact_idx_to_ids[i] = ids
    for mid in ids:
        id_to_fact_idx[mid] = i
    status = f"→ {len(ids)} memory/memories" if ids else "→ (deduplicated/merged)"
    print(f"  [{i}] {fact[:60]:<60}  {status}")

ingest_ms = (time.perf_counter() - t_ingest) * 1000

# Show what actually got stored (mem0 may merge/paraphrase facts)
all_mems = m.get_all(filters={"user_id": USER})
print(f"\n  Stored {len(all_mems['results'])} memories from {len(FACTS)} facts:")
for mem in all_mems["results"]:
    fi = id_to_fact_idx.get(mem["id"], "?")
    print(f"    [fact {fi}] {mem['memory']}")

print(f"\n  ⏱  Ingest: {ingest_ms:.0f}ms total  ({ingest_ms/len(FACTS):.0f}ms/fact)\n")

# ── Search: query mem0 and evaluate against source fact IDs ──────────────────
K_VALUES = [1, 3, 5]
p_at_k, mrr_scores, search_times = {k: [] for k in K_VALUES}, [], []

print(f"{'Query':<50} {'P@1':>4} {'P@3':>4} {'P@5':>4} {'MRR':>6} {'ms':>5}")
print("─" * 75)

for query, expected_fact_idx in QUERIES:
    relevant_ids = fact_idx_to_ids.get(expected_fact_idx, set())

    t0 = time.perf_counter()
    hits = m.search(query, filters={"user_id": USER}, limit=max(K_VALUES))
    ms = (time.perf_counter() - t0) * 1000
    search_times.append(ms)

    retrieved_ids = [h["id"] for h in hits["results"]]

    # A hit is correct if its ID maps to the expected fact index
    retrieved_fact_idxs = [id_to_fact_idx.get(rid, -1) for rid in retrieved_ids]
    relevant_set = {expected_fact_idx}

    mrr_score = mrr(retrieved_fact_idxs, relevant_set)
    mrr_scores.append(mrr_score)

    row = f"{query[:49]:<50}"
    for k in K_VALUES:
        p = precision_at_k(retrieved_fact_idxs, relevant_set, k)
        p_at_k[k].append(p)
        row += f" {'✓' if p else '✗':>4}"
    row += f" {mrr_score:>6.3f} {ms:>5.0f}"
    print(row)

    # Show what was actually retrieved
    if hits["results"]:
        top = hits["results"][0]
        fi = id_to_fact_idx.get(top["id"], "?")
        print(f"    → top result [fact {fi}]: {top['memory'][:70]}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("─" * 75)
avg_mrr = sum(mrr_scores) / len(mrr_scores)
summary = f"{'AVERAGE':<50}"
for k in K_VALUES:
    avg_p = sum(p_at_k[k]) / len(p_at_k[k])
    summary += f" {avg_p:>4.2f}"
summary += f" {avg_mrr:>6.3f} {sum(search_times)/len(search_times):>5.0f}"
print(summary)

print(f"""
📊 Final Results
  ┌─────────────────┬──────────┐
  │ Metric          │  Score   │
  ├─────────────────┼──────────┤
  │ Precision@1     │ {sum(p_at_k[1])/len(p_at_k[1]):>7.1%}  │
  │ Precision@3     │ {sum(p_at_k[3])/len(p_at_k[3]):>7.1%}  │
  │ Precision@5     │ {sum(p_at_k[5])/len(p_at_k[5]):>7.1%}  │
  │ MRR             │ {avg_mrr:>7.3f}  │
  ├─────────────────┼──────────┤
  │ Avg ingest      │ {ingest_ms/len(FACTS):>5.0f}ms  │
  │ Avg search      │ {sum(search_times)/len(search_times):>5.0f}ms  │
  │ Facts stored    │ {len(all_mems['results']):>7}  │
  └─────────────────┴──────────┘
""")

m.delete_all(user_id=USER)  # v2 API: user_id= not filters=
print("✅ Benchmark complete.")
