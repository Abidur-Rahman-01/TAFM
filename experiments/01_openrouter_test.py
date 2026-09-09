"""
Experiment 01 — Quick smoke test with OpenRouter LLM + fastembed (local embeddings).
mem0ai v2.0.11 API: search/get_all/delete_all use filters={} not top-level user_id.

Run: python experiments/01_openrouter_test.py
"""

from mem0 import Memory

OPENROUTER_API_KEY = "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"
USER = "thesis_sadik"

config = {
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
            "collection_name": "openrouter_thesis",
            "embedding_model_dims": 384,
            "on_disk": False,
        },
    },
    "version": "v1.1",
}

print("=" * 60)
print("mem0 v2 — OpenRouter + fastembed Test")
print("=" * 60)
print("LLM:      openai/gpt-4o-mini  (via OpenRouter)")
print("Embedder: BAAI/bge-small-en-v1.5  (local fastembed)")
print()

print("Initialising Memory...")
m = Memory.from_config(config)
print("✅ Memory initialised\n")

# ── 1. Add memories ───────────────────────────────────────────────────────────
print("[1] Adding memories...")
r1 = m.add(
    "My name is Sadik. I am a PhD student researching persistent memory systems for AI agents.",
    user_id=USER,
)
print(f"  → {[x['memory'] for x in r1.get('results', [])]}")

r2 = m.add(
    "I prefer Python. My thesis focuses on long-term memory retrieval accuracy using hybrid scoring.",
    user_id=USER,
)
print(f"  → {[x['memory'] for x in r2.get('results', [])]}")

r3 = m.add(
    [
        {"role": "user",      "content": "Which frameworks should I use for memory experiments?"},
        {"role": "assistant", "content": "Consider mem0, LangChain memory, and MemGPT."},
    ],
    user_id=USER,
)
print(f"  → {[x['memory'] for x in r3.get('results', [])]}")

# ── 2. Search (v2 API: filters={}) ───────────────────────────────────────────
print("\n[2] Searching memories...")
queries = [
    "What is Sadik studying?",
    "What language does the user prefer?",
    "What frameworks were recommended?",
]
for q in queries:
    hits = m.search(q, filters={"user_id": USER}, limit=3)
    print(f"\n  Query: '{q}'")
    for h in hits["results"]:
        print(f"    [{h['score']:.4f}] {h['memory']}")

# ── 3. Get all (v2 API) ───────────────────────────────────────────────────────
print("\n[3] All stored memories:")
all_mems = m.get_all(filters={"user_id": USER})
for i, mem in enumerate(all_mems["results"], 1):
    print(f"  {i}. [{mem['id'][:8]}…] {mem['memory']}")

# ── 4. Update + history ───────────────────────────────────────────────────────
if all_mems["results"]:
    mem_id = all_mems["results"][0]["id"]
    print(f"\n[4] Updating memory {mem_id[:8]}…")
    m.update(mem_id, "Sadik is a PhD researcher specialising in AI memory and retrieval systems.")
    updated = m.get(mem_id)
    print(f"  → Updated: {updated['memory']}")

    print("\n[5] Memory history (audit trail):")
    for h in m.history(mem_id):
        print(f"  [{h.get('event','?')}] {h.get('old_memory','—')} → {h.get('new_memory','—')}")

# ── 5. Cleanup ────────────────────────────────────────────────────────────────
m.delete_all(user_id=USER)
print(f"\n✅ Done! Total memories stored and retrieved: {len(all_mems['results'])}")
