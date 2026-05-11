"""RAG retriever: embed query, search Milvus piade_telemetry collection."""

import logging
import os

from dotenv import load_dotenv
from fastembed import TextEmbedding
from pymilvus import Collection, connections

load_dotenv()

logger = logging.getLogger(__name__)

MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT: str = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME: str = "piade_telemetry"

EMBED_MODEL_NAME: str = "nomic-ai/nomic-embed-text-v1.5"

# Lazy singletons
_collection: Collection | None = None
_embed_model: TextEmbedding | None = None


def _get_collection() -> Collection:
    global _collection
    if _collection is None:
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        _collection = Collection(COLLECTION_NAME)
        _collection.load()
        logger.info("Milvus collection '%s' carregada.", COLLECTION_NAME)
    return _collection


def _get_embed_model() -> TextEmbedding:
    global _embed_model
    if _embed_model is None:
        _embed_model = TextEmbedding(model_name=EMBED_MODEL_NAME)
        logger.info("Modelo de embedding '%s' carregado.", EMBED_MODEL_NAME)
    return _embed_model


def _embed_query(query: str) -> list[float]:
    model = _get_embed_model()
    prefixed = f"search_query: {query}"
    embeddings = list(model.embed([prefixed]))
    return embeddings[0].tolist()


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Search Milvus for the top_k telemetry windows most similar to query.

    Args:
        query: Natural language question from the technician.
        top_k: Number of results to retrieve.

    Returns:
        List of dicts with keys: machine_id, interval_start, pct_idle,
        pct_downtime, pct_perf_loss, count_sum, log_text, score.
    """
    logger.info("Recuperando top-%d janelas de telemetria para query: %r", top_k, query)

    try:
        embedding = _embed_query(query)
    except Exception as exc:
        raise ConnectionError(f"Erro ao gerar embedding: {exc}") from exc

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
        output_fields=[
            "machine_id", "interval_start",
            "pct_idle", "pct_downtime", "pct_perf_loss",
            "count_sum", "log_text",
        ],
    )

    hits: list[dict] = []
    for hit in results[0]:
        hits.append({
            "machine_id":     hit.entity.get("machine_id", ""),
            "interval_start": hit.entity.get("interval_start", ""),
            "pct_idle":       float(hit.entity.get("pct_idle") or 0),
            "pct_downtime":   float(hit.entity.get("pct_downtime") or 0),
            "pct_perf_loss":  float(hit.entity.get("pct_perf_loss") or 0),
            "count_sum":      float(hit.entity.get("count_sum") or 0),
            "log_text":       hit.entity.get("log_text", ""),
            "score":          float(hit.score),
        })

    logger.info("Recuperados %d resultados.", len(hits))
    return hits
