"""
Experiment 01 — Quick smoke test for the mem0 Python SDK.
Run: python experiments/01_quick_test.py

Requires: OPENAI_API_KEY set in environment (or use Ollama config below).
"""

import os
from mem0 import Memory

# ─── Configuration ────────────────────────────────────────────────────────────
# Default: uses OpenAI + in-memory Qdrant (no Docker needed)
# Uncomment the block below to use Ollama 100% locally (no API key needed):

# config = {
#     "llm": {
#         "provider": "ollama",
#         "config": {"model": "llama3.1:latest", "ollama_base_url": "http://localhost:11434"}
#     },
#     "embedder": {
#         "provider": "ollama",
#         "config": {"model": "nomic-embed-text", "ollama_base_url": "http://localhost:11434"}
#     },
# }
# m = Memory.from_config(config)

m = Memory()  # uses OPENAI_API_KEY, in-memory Qdrant

USER = "thesis_user"

print("=" * 60)
print("mem0 Quick Smoke Test")
print("=" * 60)

# ─── 1. Add memories ──────────────────────────────────────────────────────────
print("\n[1] Adding memories...")
result = m.add(
    "My name is Sadik. I am a PhD student researching persistent memory systems for AI agents.",
    user_id=USER
)
print(f"  → {result}")

result2 = m.add(
    "I prefer Python over other languages. My thesis focuses on long-term memory retrieval accuracy.",
    user_id=USER
)
print(f"  → {result2}")

result3 = m.add(
    [
        {"role": "user", "content": "What frameworks should I use for my memory experiments?"},
        {"role": "assistant", "content": "You should look at mem0, LangChain memory, and MemGPT."},
    ],
    user_id=USER
)
print(f"  → {result3}")

# ─── 2. Search ────────────────────────────────────────────────────────────────
print("\n[2] Searching memories...")
queries = [
    "What is Sadik studying?",
    "What programming language does the user prefer?",
    "What frameworks were recommended?",
]

for q in queries:
    hits = m.search(q, user_id=USER, limit=3)
    print(f"\n  Query: '{q}'")
    for h in hits["results"]:
        print(f"    [{h['score']:.4f}] {h['memory']}")

# ─── 3. Get all ───────────────────────────────────────────────────────────────
print("\n[3] All stored memories:")
all_mems = m.get_all(user_id=USER)
for i, mem in enumerate(all_mems["results"], 1):
    print(f"  {i}. [{mem['id'][:8]}…] {mem['memory']}")

# ─── 4. Update a memory ───────────────────────────────────────────────────────
print("\n[4] Updating first memory...")
if all_mems["results"]:
    mem_id = all_mems["results"][0]["id"]
    m.update(mem_id, "Sadik is a PhD researcher specializing in AI memory and retrieval systems.")
    updated = m.get(mem_id)
    print(f"  → Updated: {updated['memory']}")

# ─── 5. History ───────────────────────────────────────────────────────────────
print("\n[5] Memory history (audit trail):")
if all_mems["results"]:
    mem_id = all_mems["results"][0]["id"]
    history = m.history(mem_id)
    for entry in history:
        print(f"  [{entry.get('event','?')}] {entry.get('old_memory','—')} → {entry.get('new_memory','—')}")

print("\n✅ All tests passed!")
print(f"   Total memories stored: {len(all_mems['results'])}")
