"""RAG retriever: embed query, search Milvus smartfactory_logs collection."""

import logging
import os

from dotenv import load_dotenv
from fastembed import TextEmbedding
from pymilvus import Collection, connections

from rag.query_parser import QueryMetadata, parse_query

load_dotenv()

logger = logging.getLogger(__name__)

MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT: str = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME: str = "smartfactory_logs"

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
    """Search Milvus for the top_k log events most similar to query.

    Args:
        query: Natural language question from the technician.
        top_k: Number of results to retrieve.

    Returns:
        List of dicts with keys: station, event_timestamp, current_state,
        current_task, current_sub_task, log_text, score.
    """
    logger.info("Recuperando top-%d eventos de log para query: %r", top_k, query)

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
            "event_id", "station", "event_timestamp",
            "current_state", "current_task", "current_sub_task", "log_text",
        ],
    )

    hits: list[dict] = []
    for hit in results[0]:
        hits.append({
            "event_id":         hit.entity.get("event_id", ""),
            "station":          hit.entity.get("station", ""),
            "event_timestamp":  hit.entity.get("event_timestamp", ""),
            "current_state":    hit.entity.get("current_state", ""),
            "current_task":     hit.entity.get("current_task", ""),
            "current_sub_task": hit.entity.get("current_sub_task", ""),
            "log_text":         hit.entity.get("log_text", ""),
            "score":            float(hit.score),
        })

    logger.info("Recuperados %d resultados.", len(hits))
    return hits


# ── Hybrid retrieval (filtro escalar + multi-query) ──────────────────────────

_OUTPUT_FIELDS = [
    "event_id", "station", "event_timestamp",
    "current_state", "current_task", "current_sub_task", "log_text",
]


def _hit_to_dict(hit) -> dict:
    return {
        "event_id":         hit.entity.get("event_id", ""),
        "station":          hit.entity.get("station", ""),
        "event_timestamp":  hit.entity.get("event_timestamp", ""),
        "current_state":    hit.entity.get("current_state", ""),
        "current_task":     hit.entity.get("current_task", ""),
        "current_sub_task": hit.entity.get("current_sub_task", ""),
        "log_text":         hit.entity.get("log_text", ""),
        "score":            float(hit.score),
    }


def _search_with_expr(
    collection: Collection, embedding: list[float], expr: str | None, top_k: int,
) -> list[dict]:
    """Wrapper para collection.search com expr opcional."""
    kwargs = dict(
        data=[embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 16}},
        limit=top_k,
        output_fields=_OUTPUT_FIELDS,
    )
    if expr:
        kwargs["expr"] = expr
    results = collection.search(**kwargs)
    return [_hit_to_dict(h) for h in results[0]]


def _build_expr(station: str, md: QueryMetadata) -> str:
    """Monta filtro escalar para uma estação, com janela temporal se disponível."""
    parts = [f"station == '{station}'"]
    if md.window_lo and md.window_hi:
        parts.append(f"event_timestamp >= '{md.window_lo}'")
        parts.append(f"event_timestamp <= '{md.window_hi}'")
    return " && ".join(parts)


def retrieve_hybrid(query: str, top_k: int = 5) -> list[dict]:
    """Hybrid retrieval: filtro escalar (station + timestamp) + vector search.

    Estratégia:
      - 0 estações detectadas: fallback para retrieve() vetorial puro.
      - 1 estação: 1 busca filtrada por station (+ janela temporal se houver).
      - 2+ estações: multi-query — 1 busca por estação, merge + dedup por event_id.

    Se a busca filtrada retorna 0 hits (janela vazia), faz fallback para a mesma
    busca SEM filtro temporal (mantém só station).
    """
    md = parse_query(query)
    logger.info(
        "Hybrid: stations=%s timestamp=%s window=%s..%s",
        md.stations, md.timestamp, md.window_lo, md.window_hi,
    )

    if not md.stations:
        logger.info("Nenhuma estação detectada — fallback para vector search puro.")
        return retrieve(query, top_k=top_k)

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

    if len(md.stations) == 1:
        station = md.stations[0]
        expr = _build_expr(station, md)
        hits = _search_with_expr(collection, embedding, expr, top_k)
        if not hits and md.window_lo:
            logger.info("Janela temporal vazia — refazendo sem filtro de timestamp.")
            hits = _search_with_expr(
                collection, embedding, f"station == '{station}'", top_k,
            )
        return hits

    per_station = max(2, top_k // len(md.stations) + 1)
    pooled: dict[str, dict] = {}
    for station in md.stations:
        expr = _build_expr(station, md)
        sub = _search_with_expr(collection, embedding, expr, per_station)
        if not sub and md.window_lo:
            sub = _search_with_expr(
                collection, embedding, f"station == '{station}'", per_station,
            )
        for h in sub:
            eid = h["event_id"]
            if eid and (eid not in pooled or h["score"] > pooled[eid]["score"]):
                pooled[eid] = h

    merged = sorted(pooled.values(), key=lambda h: h["score"], reverse=True)
    return merged[:top_k]
