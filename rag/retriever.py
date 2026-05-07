"""Sprint 6 — RAG retriever: embed query and search Milvus."""

import logging
import os

import requests
from dotenv import load_dotenv
from pymilvus import Collection, connections

load_dotenv()

logger = logging.getLogger(__name__)

MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT: str = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME: str = "alarm_logs"

OLLAMA_BASE: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL: str = "nomic-embed-text"

# Lazy singleton — connects and loads collection once per process.
_collection: Collection | None = None


def _get_collection() -> Collection:
    global _collection
    if _collection is None:
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        _collection = Collection(COLLECTION_NAME)
        _collection.load()
        logger.info("Milvus collection '%s' carregada.", COLLECTION_NAME)
    return _collection


def _embed_query(query: str) -> list[float]:
    resp = requests.post(
        f"{OLLAMA_BASE}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": query},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Search Milvus for the top_k logs most similar to query.

    Args:
        query: Natural language question from the technician.
        top_k: Number of results to retrieve.

    Returns:
        List of dicts with keys: alarm_code, machine_id, source,
        event_type, severity, log_text, score.
    """
    logger.info("Recuperando top-%d logs para query: %r", top_k, query)

    try:
        embedding = _embed_query(query)
    except requests.ConnectionError as exc:
        raise ConnectionError(
            f"Ollama indisponível em {OLLAMA_BASE}. Verifique se o serviço está rodando."
        ) from exc

    try:
        collection = _get_collection()
    except Exception as exc:
        raise ConnectionError(
            f"Milvus indisponível em {MILVUS_HOST}:{MILVUS_PORT}."
        ) from exc

    results = collection.search(
        data=[embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 16}},
        limit=top_k,
        output_fields=["alarm_code", "machine_id", "source", "event_type", "severity", "log_text"],
    )

    hits: list[dict] = []
    for hit in results[0]:
        event = hit.entity.get("event_type", "") or None  # "" was stored for NULL
        hits.append({
            "alarm_code": hit.entity.get("alarm_code", ""),
            "machine_id": hit.entity.get("machine_id", ""),
            "source": hit.entity.get("source", ""),
            "event_type": event,
            "severity": hit.entity.get("severity", ""),
            "log_text": hit.entity.get("log_text", ""),
            "score": float(hit.score),
        })

    logger.info("Recuperados %d resultados.", len(hits))
    return hits
