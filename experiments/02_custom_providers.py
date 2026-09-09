"""
Experiment 02 — Swapping providers without changing application code.
Demonstrates the provider plugin architecture.

Run: python experiments/02_custom_providers.py

Configs tested:
  A) OpenAI LLM + Qdrant (default, cloud)
  B) Ollama LLM + Chroma (fully local, no API key)
  C) Custom instructions (domain-specific fact extraction)
"""

import os
import time
from mem0 import Memory

USER = "provider_test_user"

def run_experiment(name: str, config: dict | None = None):
    print(f"\n{'─' * 55}")
    print(f"Config: {name}")
    print('─' * 55)

    m = Memory.from_config(config) if config else Memory()

    t0 = time.perf_counter()
    result = m.add(
        "The user is experimenting with different memory backends for their thesis on AI memory retrieval.",
        user_id=USER
    )
    add_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    hits = m.search("memory backends thesis", user_id=USER, limit=3)
    search_ms = (time.perf_counter() - t0) * 1000

    print(f"  add()    → {add_ms:.0f}ms  | facts extracted: {len(result.get('results', []))}")
    print(f"  search() → {search_ms:.0f}ms | top result: {hits['results'][0]['memory'] if hits['results'] else 'none'}")

    m.delete_all(user_id=USER)


# ── A: Default (OpenAI + Qdrant in-memory) ──────────────────────────────────
run_experiment("OpenAI + Qdrant (default)")

# ── B: Chroma persistent vector store ────────────────────────────────────────
chroma_config = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "thesis_experiments",
            "path": "./experiments/.chroma_db",
        }
    }
    # LLM + embedder inherit OpenAI defaults
}
run_experiment("OpenAI LLM + Chroma (persistent)", chroma_config)

# ── C: Custom extraction instructions ────────────────────────────────────────
custom_instructions_config = {
    "custom_instructions": (
        "Focus only on extracting research-relevant facts: "
        "methodologies, datasets, models, metrics, and findings. "
        "Ignore personal preferences and conversational filler."
    )
}
run_experiment("Custom domain instructions (research-focused)", custom_instructions_config)

# ── D: Fully local with Ollama (uncomment if Ollama is running) ──────────────
# ollama_config = {
#     "llm": {
#         "provider": "ollama",
#         "config": {"model": "llama3.1:latest", "ollama_base_url": "http://localhost:11434"}
#     },
#     "embedder": {
#         "provider": "ollama",
#         "config": {"model": "nomic-embed-text", "ollama_base_url": "http://localhost:11434"}
#     },
#     "vector_store": {
#         "provider": "chroma",
#         "config": {"collection_name": "ollama_test", "path": "./experiments/.chroma_ollama"}
#     }
# }
# run_experiment("Ollama (fully local, no API key)", ollama_config)

print("\n✅ Provider comparison complete.")
