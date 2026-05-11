"""Sprint 6 — RAG retriever: embed query, search Milvus, enrich with alarm_dictionary."""

import logging
import os

import psycopg2
from dotenv import load_dotenv
from fastembed import TextEmbedding
from pymilvus import Collection, connections

load_dotenv()

logger = logging.getLogger(__name__)

MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT: str = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME: str = "alarm_logs"

EMBED_MODEL_NAME: str = "nomic-ai/nomic-embed-text-v1.5"

POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB: str = os.getenv("POSTGRES_DB", "aiops_industry")
POSTGRES_USER: str = os.getenv("POSTGRES_USER", "aiops")
POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "")

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
    # nomic-embed-text-v1.5 usa prefixo "search_query:" para queries
    prefixed = f"search_query: {query}"
    embeddings = list(model.embed([prefixed]))
    return embeddings[0].tolist()


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
    except Exception as exc:
        raise ConnectionError(
            f"Erro ao gerar embedding: {exc}"
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
            "dict_title": None,
            "dict_description": None,
            "dict_probable_causes": None,
            "dict_corrective_actions": None,
        })

    _enrich_with_dictionary(hits)

    logger.info("Recuperados %d resultados.", len(hits))
    return hits


def _enrich_with_dictionary(hits: list[dict]) -> None:
    """Query alarm_dictionary for each unique alarm_code and enrich hits in-place."""
    unique_codes = {h["alarm_code"] for h in hits if h["alarm_code"]}
    if not unique_codes:
        return

    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=int(POSTGRES_PORT),
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
        )
    except Exception as exc:
        logger.warning("Não foi possível conectar ao PostgreSQL para enriquecer resultados: %s", exc)
        return

    dict_cache: dict[str, dict] = {}
    try:
        with conn, conn.cursor() as cur:
            for code in unique_codes:
                cur.execute(
                    "SELECT title, description, probable_causes, corrective_actions, severity "
                    "FROM alarm_dictionary WHERE alarm_code = %s LIMIT 1",
                    (code,),
                )
                row = cur.fetchone()
                if row:
                    dict_cache[code] = {
                        "dict_title": row[0],
                        "dict_description": row[1],
                        "dict_probable_causes": row[2],
                        "dict_corrective_actions": row[3],
                    }
    except Exception as exc:
        logger.warning("Erro ao consultar alarm_dictionary: %s", exc)
    finally:
        conn.close()

    for hit in hits:
        entry = dict_cache.get(hit["alarm_code"])
        if entry:
            hit.update(entry)
