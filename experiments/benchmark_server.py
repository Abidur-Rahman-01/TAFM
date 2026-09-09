"""
mem0 Benchmark Bridge Server
=============================
A lightweight FastAPI server that wraps the mem0 Python SDK directly,
exposing exactly the REST endpoints the benchmark runner expects:
  POST /memories       → m.add()
  POST /search         → m.search()
  DELETE /memories     → m.delete_all()
  GET  /health         → alive check

No Docker, no PostgreSQL needed. Uses:
  LLM:      OpenRouter (openai/gpt-4o-mini)
  Embedder: fastembed BAAI/bge-small-en-v1.5 (local)
  Store:    Qdrant in-memory

Run:
  python experiments/benchmark_server.py

Then in another terminal:
  cd memory-benchmarks
  python -m benchmarks.locomo.run --project-name locomo-v1
"""

import os, sys, logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mem0 import Memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bridge")

OPENROUTER_KEY = "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"

CONFIG = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "openai/gpt-4o-mini",
            "api_key": OPENROUTER_KEY,
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
            "collection_name": "benchmark_store",
            "embedding_model_dims": 384,
            "on_disk": False,
        },
    },
    "version": "v1.1",
}

logger.info("Initialising mem0 Memory instance...")
mem = Memory.from_config(CONFIG)
logger.info("✅ mem0 ready")

app = FastAPI(title="mem0 Benchmark Bridge", version="1.0")


@app.get("/health")
async def health():
    return {"status": "ok", "backend": "mem0-python-sdk", "model": "openai/gpt-4o-mini"}


@app.post("/memories")
async def add_memory(request: Request):
    """POST /memories — ingests a conversation turn into mem0."""
    body = await request.json()
    messages = body.get("messages", [])
    user_id  = body.get("user_id", "default")
    custom_instructions = body.get("custom_instructions")

    try:
        kwargs = {"user_id": user_id}
        if custom_instructions:
            kwargs["prompt"] = custom_instructions
        result = mem.add(messages, **kwargs)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"add error: {e}")
        return JSONResponse(content={"results": []}, status_code=200)


@app.post("/search")
async def search_memory(request: Request):
    """POST /search — semantic search over stored memories."""
    body = await request.json()
    query   = body.get("query", "")
    user_id = body.get("user_id", "default")
    limit   = body.get("limit", 200)

    try:
        result = mem.search(query, filters={"user_id": user_id}, limit=min(limit, 500))
        # Normalise to format benchmark expects
        results = []
        for r in result.get("results", []):
            results.append({
                "id":         r.get("id", ""),
                "memory":     r.get("memory", ""),
                "score":      r.get("score", 0.0),
                "created_at": r.get("created_at", ""),
                "updated_at": r.get("updated_at", ""),
            })
        return JSONResponse(content={"results": results})
    except Exception as e:
        logger.error(f"search error: {e}")
        return JSONResponse(content={"results": []}, status_code=200)


@app.delete("/memories")
async def delete_memories(user_id: str = "default"):
    """DELETE /memories?user_id=xxx — wipes all memories for a user."""
    try:
        mem.delete_all(user_id=user_id)
        logger.info(f"Deleted all memories for user={user_id}")
        return JSONResponse(content={"status": "deleted"})
    except Exception as e:
        logger.error(f"delete error: {e}")
        return JSONResponse(content={"status": "error", "detail": str(e)}, status_code=200)


@app.get("/memories")
async def get_memories(user_id: str = "default"):
    """GET /memories?user_id=xxx — list all memories."""
    try:
        result = mem.get_all(filters={"user_id": user_id})
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"get_all error: {e}")
        return JSONResponse(content={"results": []}, status_code=200)


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  mem0 Benchmark Bridge Server")
    print("=" * 60)
    print(f"  LLM:      openai/gpt-4o-mini (OpenRouter)")
    print(f"  Embedder: BAAI/bge-small-en-v1.5 (local fastembed)")
    print(f"  Store:    Qdrant in-memory")
    print(f"  URL:      http://localhost:8888")
    print()
    print("  Run benchmarks in another terminal:")
    print("  cd memory-benchmarks")
    print("  python -m benchmarks.locomo.run --project-name locomo-v1")
    print("=" * 60)
    print()
    uvicorn.run(app, host="0.0.0.0", port=8888, log_level="warning")
