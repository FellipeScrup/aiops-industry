"""Gold → Vector store: embeds smart factory EPISODES into Milvus.

Reads aggregated episodes from PostgreSQL (smartfactory_episodes, produced by
parse_episodes.py), generates 768-dim embeddings via fastembed
(nomic-embed-text-v1.5) over the human-readable episode text, and indexes them
in the Milvus collection `smartfactory_episodes`.

Why episodes and not raw events: a query like "why did the station stop?" must
retrieve a meaningful not-ready episode with a duration and a cause, not a
single 100 ms snapshot of an idle station. See parse_episodes.py.

Indexing policy: ALL not-ready episodes (which includes every duration anomaly)
are indexed; ready/idle episodes are only sampled (READY_SAMPLE) so the store
is not flooded with "everything is fine" vectors.

Each embedded text is enriched with BPM activity context (next station / process)
so the generator can reason about production-line impact.
"""

import logging
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastembed import TextEmbedding
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ingestion.bpm_context import format_bpm_context  # noqa: E402

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

POSTGRES_USER: str = os.getenv("POSTGRES_USER", "aiops")
POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "aiops")
POSTGRES_DB: str = os.getenv("POSTGRES_DB", "aiops_industry")
POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")

MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT: str = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME: str = "smartfactory_episodes"

EMBED_MODEL_NAME: str = "nomic-ai/nomic-embed-text-v1.5"
EMBED_DIM: int = 768

# Number of ready/idle episodes to keep (all not-ready are always indexed).
# Default 0: this is a fault-diagnosis assistant, and idle episodes hijack
# ambiguous queries like "why did the station stop?" (semantically near "idle").
# Set READY_SAMPLE>0 to add idle context for non-diagnostic questions.
READY_SAMPLE: int = int(os.getenv("READY_SAMPLE", "0"))
EMBED_BATCH: int = 256


# ── Fastembed ─────────────────────────────────────────────────────────────────

_embed_model: TextEmbedding | None = None


def _get_embed_model() -> TextEmbedding:
    global _embed_model
    if _embed_model is None:
        logger.info("Carregando modelo de embedding '%s'...", EMBED_MODEL_NAME)
        _embed_model = TextEmbedding(model_name=EMBED_MODEL_NAME)
        logger.info("Modelo carregado.")
    return _embed_model


def _embed_batch(texts: list[str]) -> list[list[float]]:
    model = _get_embed_model()
    return [e.tolist() for e in model.embed(texts)]


# ── PostgreSQL ────────────────────────────────────────────────────────────────

def _build_engine() -> Engine:
    url = (
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    return create_engine(url)


def _load_episodes(engine: Engine) -> pd.DataFrame:
    logger.info("Carregando episódios do PostgreSQL (smartfactory_episodes)...")
    query = text("""
        SELECT episode_id, station, current_state, current_task, current_sub_task,
               start_ts, end_ts, duration_s, is_anomaly, text
        FROM smartfactory_episodes
        ORDER BY start_ts
    """)
    df = pd.read_sql(query, engine)
    logger.info("  %d episódios carregados.", len(df))
    return df


def _select_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """All not-ready episodes + a small sample of ready/idle episodes."""
    not_ready = df[df["current_state"] == "not ready"]
    ready = df[df["current_state"] == "ready"]
    if len(ready) > READY_SAMPLE:
        ready = ready.sample(n=READY_SAMPLE, random_state=42)
    selected = pd.concat([not_ready, ready]).reset_index(drop=True)
    logger.info(
        "Selecionados %d episódios para indexar (not ready=%d, anomalias=%d, ready amostrados=%d).",
        len(selected),
        len(not_ready),
        int(df["is_anomaly"].sum()),
        len(ready),
    )
    return selected


# ── Milvus ────────────────────────────────────────────────────────────────────

def _connect_milvus() -> None:
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    logger.info("Conectado ao Milvus em %s:%s", MILVUS_HOST, MILVUS_PORT)


def _recreate_collection() -> Collection:
    if utility.has_collection(COLLECTION_NAME):
        logger.info("Collection '%s' já existe — dropando para reconstruir.", COLLECTION_NAME)
        utility.drop_collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="event_id",         dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="station",          dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="event_timestamp",  dtype=DataType.VARCHAR, max_length=30),
        FieldSchema(name="end_ts",           dtype=DataType.VARCHAR, max_length=30),
        FieldSchema(name="current_state",    dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="current_task",     dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="current_sub_task", dtype=DataType.VARCHAR, max_length=300),
        FieldSchema(name="duration_s",       dtype=DataType.FLOAT),
        FieldSchema(name="is_anomaly",       dtype=DataType.BOOL),
        FieldSchema(name="log_text",         dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="embedding",        dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),
    ]
    schema = CollectionSchema(fields=fields, description="Smart Factory episode embeddings")
    col = Collection(name=COLLECTION_NAME, schema=schema)
    logger.info("Collection '%s' criada.", COLLECTION_NAME)
    return col


def _build_index_and_load(collection: Collection) -> None:
    logger.info("Criando index IVF_FLAT (COSINE, nlist=128)...")
    collection.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        },
    )
    collection.load()
    logger.info("Collection carregada. Total de vetores: %d", collection.num_entities)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_embed_text(row: pd.Series) -> str:
    """Episode text + BPM context, prefixed for nomic document embedding."""
    bpm = format_bpm_context(str(row["station"]), str(row["current_task"] or ""))
    base = f"search_document: {row['text']}"
    return f"{base} | {bpm}" if bpm else base


# ── Pipeline ──────────────────────────────────────────────────────────────────

def _run(df: pd.DataFrame, collection: Collection) -> None:
    buf: dict[str, list] = {f.name: [] for f in collection.schema.fields}
    total = 0

    for start in range(0, len(df), EMBED_BATCH):
        batch = df.iloc[start : start + EMBED_BATCH]
        texts = [_build_embed_text(row) for _, row in batch.iterrows()]
        embeddings = _embed_batch(texts)

        for i, (_, row) in enumerate(batch.iterrows()):
            buf["event_id"].append(str(row["episode_id"])[:64])
            buf["station"].append(str(row["station"])[:20])
            buf["event_timestamp"].append(str(row["start_ts"])[:30])
            buf["end_ts"].append(str(row["end_ts"])[:30])
            buf["current_state"].append(str(row["current_state"])[:20])
            buf["current_task"].append(str(row["current_task"] or "")[:500])
            buf["current_sub_task"].append(str(row["current_sub_task"] or "")[:300])
            buf["duration_s"].append(float(row["duration_s"]))
            buf["is_anomaly"].append(bool(row["is_anomaly"]))
            buf["log_text"].append(str(row["text"])[:2000])
            buf["embedding"].append(embeddings[i])

        total += len(batch)
        logger.info("Embeddados %d/%d episódios.", total, len(df))

    data = [buf[f.name] for f in collection.schema.fields]
    collection.insert(data)
    collection.flush()
    logger.info("Inseridos %d vetores no Milvus.", total)


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    _connect_milvus()
    collection = _recreate_collection()

    engine = _build_engine()
    df = _load_episodes(engine)
    if df.empty:
        logger.warning("Tabela smartfactory_episodes vazia. Rode `make ingest-episodes` antes.")
        return

    selected = _select_episodes(df)
    _run(selected, collection)
    _build_index_and_load(collection)

    logger.info("Pipeline concluído. Vetores indexados: %d", collection.num_entities)


if __name__ == "__main__":
    main()
