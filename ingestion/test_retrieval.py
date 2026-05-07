"""Sprint 5 — validação do RAG retrieval via busca vetorial no Milvus."""

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
COLLECTION_NAME: str = "alarm_logs"
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
    """Embed query and print top-K similar logs from Milvus."""
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
        output_fields=["alarm_code", "machine_id", "severity", "log_text"],
    )

    print(f"\nQuery: {query!r}\n")
    for rank, hit in enumerate(results[0], start=1):
        log_text = hit.entity.get("log_text", "")
        alarm = hit.entity.get("alarm_code", "?")
        machine = hit.entity.get("machine_id", "?")
        severity = hit.entity.get("severity", "?")
        print(f"Rank {rank} (score: {hit.score:.4f}): {log_text}")
        print(f"  → Alarme {alarm} | Máquina {machine} | {severity}\n")


def main() -> None:
    if len(sys.argv) < 2:
        print('Uso: python ingestion/test_retrieval.py "<query>"', file=sys.stderr)
        sys.exit(1)
    search(sys.argv[1])


if __name__ == "__main__":
    main()
