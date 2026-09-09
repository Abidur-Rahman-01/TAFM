"""
Experiment 03 — Retrieval accuracy micro-benchmark.
Tests memory recall precision/recall at different top-K values.

Run: python experiments/03_retrieval_benchmark.py

This is a thesis-relevant experiment: measures how well mem0's
hybrid retrieval (vector + BM25) surfaces the correct memories.
"""

import os
from dataclasses import dataclass, field
from mem0 import Memory

USER = "benchmark_user"

# ─── Ground truth dataset ────────────────────────────────────────────────────
FACTS = [
    "Sadik uses PyTorch for deep learning experiments.",
    "The dataset used is the LOCOMO long-term conversation benchmark.",
    "The baseline model achieves 72.3% recall@5 on LongMemEval.",
    "Transformer-based memory retrieval outperforms BM25 by 14% on the thesis dataset.",
    "The thesis proposes a hybrid scoring method combining dense vectors and sparse BM25.",
    "Graph-augmented memory shows 8% improvement over flat vector retrieval.",
    "Experiment runs use an NVIDIA A100 80GB GPU.",
    "The evaluation metric is Recall@K and MRR (Mean Reciprocal Rank).",
]

# Queries paired with the expected fact index they should retrieve
QUERIES = [
    ("What deep learning framework is used?", 0),
    ("Which benchmark dataset is used for evaluation?", 1),
    ("What is the baseline recall performance?", 2),
    ("How much better is transformer retrieval vs BM25?", 3),
    ("What is the proposed method in the thesis?", 4),
    ("What improvement does graph memory provide?", 5),
    ("What GPU is used for experiments?", 6),
    ("What evaluation metrics are reported?", 7),
]

def precision_at_k(retrieved_indices: list[int], relevant_index: int, k: int) -> float:
    return 1.0 if relevant_index in retrieved_indices[:k] else 0.0

def mean_reciprocal_rank(retrieved_indices: list[int], relevant_index: int) -> float:
    for rank, idx in enumerate(retrieved_indices, 1):
        if idx == relevant_index:
            return 1.0 / rank
    return 0.0

def run_benchmark(k_values: list[int] = [1, 3, 5]):
    print("=" * 60)
    print("mem0 Retrieval Micro-Benchmark")
    print("=" * 60)

    m = Memory()
    m.delete_all(user_id=USER)

    # Index all facts
    print(f"\nIndexing {len(FACTS)} facts...")
    memory_ids = []
    for i, fact in enumerate(FACTS):
        result = m.add(fact, user_id=USER)
        ids = [r["id"] for r in result.get("results", [])]
        memory_ids.append(ids[0] if ids else None)
        print(f"  [{i}] {fact[:60]}…" if len(fact) > 60 else f"  [{i}] {fact}")

    # Build id → index map
    all_mems = m.get_all(user_id=USER)
    id_to_fact = {}
    for mem in all_mems["results"]:
        for i, fact in enumerate(FACTS):
            if mem["memory"][:30] in fact or fact[:30] in mem["memory"]:
                id_to_fact[mem["id"]] = i
                break

    print(f"\n{'─' * 60}")
    print(f"{'Query':<45} {'K=1':>5} {'K=3':>5} {'K=5':>5} {'MRR':>6}")
    print('─' * 60)

    mrr_scores = []
    p_at_k = {k: [] for k in k_values}

    for query, expected_idx in QUERIES:
        hits = m.search(query, user_id=USER, limit=max(k_values))
        retrieved_ids = [h["id"] for h in hits["results"]]
        retrieved_indices = [id_to_fact.get(rid, -1) for rid in retrieved_ids]

        mrr = mean_reciprocal_rank(retrieved_indices, expected_idx)
        mrr_scores.append(mrr)

        row = f"{query[:44]:<45}"
        for k in k_values:
            p = precision_at_k(retrieved_indices, expected_idx, k)
            p_at_k[k].append(p)
            row += f" {'✓' if p else '✗':>5}"
        row += f" {mrr:>6.3f}"
        print(row)

    print('─' * 60)
    avg_mrr = sum(mrr_scores) / len(mrr_scores)
    summary = f"{'AVERAGE':<45}"
    for k in k_values:
        avg_p = sum(p_at_k[k]) / len(p_at_k[k])
        summary += f" {avg_p:>5.2f}"
    summary += f" {avg_mrr:>6.3f}"
    print(summary)
    print('─' * 60)

    print(f"\n📊 Summary:")
    for k in k_values:
        avg_p = sum(p_at_k[k]) / len(p_at_k[k])
        print(f"  Precision@{k}: {avg_p:.1%}")
    print(f"  MRR:          {avg_mrr:.3f}")

    m.delete_all(user_id=USER)

if __name__ == "__main__":
    run_benchmark()
