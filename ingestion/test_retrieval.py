"""Validation: vector search in Milvus piade_telemetry collection."""

import logging
import os
import sys

import requests
from dotenv import load_dotenv
from pymilvus import Collection, connections

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT: str = os.getenv("MILVUS_PORT", "19530")
OLLAMA_BASE: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL: str = "nomic-embed-text"
COLLECTION_NAME: str = "piade_telemetry"
TOP_K: int = 5


def _get_query_embedding(query: str) -> list[float]:
    resp = requests.post(
        f"{OLLAMA_BASE}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": query},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def search(query: str) -> None:
    """Embed query and print top-K similar telemetry windows from Milvus."""
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection(COLLECTION_NAME)
    collection.load()

    logger.info("Gerando embedding para a query...")
    embedding = _get_query_embedding(query)

    results = collection.search(
        data=[embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 16}},
        limit=TOP_K,
        output_fields=[
            "machine_id", "interval_start",
            "pct_idle", "pct_downtime", "pct_perf_loss",
            "count_sum", "log_text",
        ],
    )

    print(f"\nQuery: {query!r}\n")
    for rank, hit in enumerate(results[0], start=1):
        machine  = hit.entity.get("machine_id", "?")
        ts       = hit.entity.get("interval_start", "?")
        downtime = (hit.entity.get("pct_downtime") or 0) * 100
        idle     = (hit.entity.get("pct_idle") or 0) * 100
        perf     = (hit.entity.get("pct_perf_loss") or 0) * 100
        count    = hit.entity.get("count_sum") or 0
        print(f"Rank {rank} (score: {hit.score:.4f}): Máquina {machine} | Janela {ts}")
        print(f"  → Downtime {downtime:.1f}% | Idle {idle:.1f}% | Perf. Loss {perf:.1f}% | Alarmes {count:.0f}\n")


def main() -> None:
    if len(sys.argv) < 2:
        print('Uso: python ingestion/test_retrieval.py "<query>"', file=sys.stderr)
        sys.exit(1)
    search(sys.argv[1])


if __name__ == "__main__":
    main()
