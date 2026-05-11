"""Silver → Gold: embedding pipeline.

Reads logs from PostgreSQL, stratified-samples EMBED_LIMIT rows,
generates 768-dim embeddings via fastembed (nomic-embed-text-v1.5),
and indexes them in Milvus. Supports checkpoint-based resumption.
"""

import logging
import os
from pathlib import Path

import numpy as np
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
COLLECTION_NAME: str = "alarm_logs"

EMBED_MODEL_NAME: str = "nomic-ai/nomic-embed-text-v1.5"
EMBED_DIM: int = 768

EMBED_LIMIT: int = int(os.getenv("EMBED_LIMIT", "50000"))
EMBED_BATCH: int = 256   # textos por lote de embedding
MILVUS_BATCH: int = 500
LOG_EVERY: int = 1000
CHECKPOINT_PATH: Path = Path("ingestion/.embed_checkpoint")


# ── Fastembed ────────────────────────────────────────────────────────────────

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
    embeddings = list(model.embed(texts))
    return [e.tolist() for e in embeddings]


# ── PostgreSQL ────────────────────────────────────────────────────────────────

def _build_engine() -> Engine:
    url = (
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    return create_engine(url)


def _load_and_label(engine: Engine) -> pd.DataFrame:
    logger.info("Carregando logs do PostgreSQL...")
    query = text("""
        SELECT source, machine_id, alarm_code, event_type, duration_ms, raw_timestamp
        FROM logs
    """)
    df = pd.read_sql(query, engine)
    logger.info("  %d linhas carregadas.", len(df))
    return _assign_severity(df)


def _assign_severity(df: pd.DataFrame) -> pd.DataFrame:
    freq = df["alarm_code"].value_counts()
    df = df.copy()
    df["alarm_frequency"] = df["alarm_code"].map(freq)

    conditions = [
        (df["source"] == "PIADE") & (df["event_type"] == "downtime"),
        (df["source"] == "PIADE") & (df["event_type"] == "performance_loss"),
        (df["source"] == "PIADE") & (df["event_type"] == "scheduled_downtime"),
        (df["source"] == "PIADE") & (df["event_type"] == "idle"),
        (df["source"] == "PIADE") & (df["event_type"] == "production"),
        df["alarm_frequency"] < 500,
        df["alarm_frequency"] > 10_000,
    ]
    choices = ["critical", "warning", "info", "info", "info", "critical", "warning"]
    df["severity"] = np.select(conditions, choices, default="info")
    logger.info(
        "  Distribuição de severidade: %s",
        df["severity"].value_counts().to_dict(),
    )
    return df


def _stratified_sample(df: pd.DataFrame, total: int) -> pd.DataFrame:
    dist = df["severity"].value_counts(normalize=True)
    frames: list[pd.DataFrame] = []
    for sev, frac in dist.items():
        n = round(frac * total)
        group = df[df["severity"] == sev]
        frames.append(group.sample(min(len(group), n), random_state=42))
    sampled = (
        pd.concat(frames)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    logger.info(
        "Amostra estratificada: %d logs | %s",
        len(sampled),
        sampled["severity"].value_counts().to_dict(),
    )
    return sampled


# ── Milvus ────────────────────────────────────────────────────────────────────

def _connect_milvus() -> None:
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    logger.info("Conectado ao Milvus em %s:%s", MILVUS_HOST, MILVUS_PORT)


def _ensure_collection() -> Collection:
    if utility.has_collection(COLLECTION_NAME):
        logger.info("Collection '%s' já existe. Pulando criação.", COLLECTION_NAME)
        return Collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="machine_id", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="alarm_code", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="event_type", dtype=DataType.VARCHAR, max_length=30),
        FieldSchema(name="severity", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="log_text", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),
    ]
    schema = CollectionSchema(fields=fields, description="AIOps alarm log embeddings")
    col = Collection(name=COLLECTION_NAME, schema=schema)
    logger.info("Collection '%s' criada.", COLLECTION_NAME)
    return col


def _flush_batch(collection: Collection, buf: dict[str, list]) -> None:
    data = [
        buf["source"],
        buf["machine_id"],
        buf["alarm_code"],
        buf["event_type"],
        buf["severity"],
        buf["log_text"],
        buf["embedding"],
    ]
    collection.insert(data)
    collection.flush()


def _build_index_and_load(collection: Collection) -> None:
    if collection.indexes:
        logger.info("Index já existe. Pulando criação.")
    else:
        logger.info("Criando index IVF_FLAT (COSINE, nlist=128)...")
        collection.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            },
        )
        logger.info("Index criado.")
    collection.load()
    logger.info(
        "Collection carregada. Total de vetores: %d",
        collection.num_entities,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_log_text(row: pd.Series) -> str:
    event = row["event_type"] if pd.notna(row["event_type"]) else "N/A"
    return (
        f"search_document: "
        f"Alarme {row['alarm_code']} | "
        f"Máquina {row['machine_id']} | "
        f"Fonte {row['source']} | "
        f"Tipo {event} | "
        f"Severidade {row['severity']}"
    )


# ── Checkpoint ────────────────────────────────────────────────────────────────

def _read_checkpoint() -> int:
    if CHECKPOINT_PATH.exists():
        try:
            offset = int(CHECKPOINT_PATH.read_text().strip())
            logger.info("Checkpoint encontrado: %d linhas já processadas.", offset)
            return offset
        except ValueError:
            logger.warning("Checkpoint corrompido. Reiniciando do zero.")
    return 0


def _write_checkpoint(offset: int) -> None:
    CHECKPOINT_PATH.write_text(str(offset))


# ── Pipeline ──────────────────────────────────────────────────────────────────

def _run_pipeline(df: pd.DataFrame, collection: Collection) -> None:
    offset = _read_checkpoint()
    total = len(df)

    logger.info(
        "Milvus: %d vetores existentes. Checkpoint: %d/%d linhas processadas.",
        collection.num_entities,
        offset,
        total,
    )

    if offset >= total:
        logger.info("Nada a processar — todas as %d linhas já foram inseridas.", total)
        return

    remaining = df.iloc[offset:].reset_index(drop=True)
    milvus_buf: dict[str, list] = {
        "source": [], "machine_id": [], "alarm_code": [],
        "event_type": [], "severity": [], "log_text": [], "embedding": [],
    }
    total_inserted = offset

    for batch_start in range(0, len(remaining), EMBED_BATCH):
        batch = remaining.iloc[batch_start: batch_start + EMBED_BATCH]

        texts = [_build_log_text(row) for _, row in batch.iterrows()]
        embeddings = _embed_batch(texts)

        for i, (_, row) in enumerate(batch.iterrows()):
            event = str(row["event_type"]) if pd.notna(row["event_type"]) else ""
            log_text = texts[i]
            milvus_buf["source"].append(str(row["source"])[:10])
            milvus_buf["machine_id"].append(str(row["machine_id"])[:20])
            milvus_buf["alarm_code"].append(str(row["alarm_code"])[:20])
            milvus_buf["event_type"].append(event[:30])
            milvus_buf["severity"].append(str(row["severity"])[:10])
            milvus_buf["log_text"].append(log_text[:500])
            milvus_buf["embedding"].append(embeddings[i])

        abs_pos = offset + batch_start + len(batch)

        if len(milvus_buf["embedding"]) >= MILVUS_BATCH:
            _flush_batch(collection, milvus_buf)
            total_inserted += len(milvus_buf["embedding"])
            _write_checkpoint(abs_pos)
            logger.info(
                "Inseridos %d vetores no Milvus (total: %d / %d)",
                len(milvus_buf["embedding"]),
                total_inserted,
                total,
            )
            for k in milvus_buf:
                milvus_buf[k].clear()

        if abs_pos % LOG_EVERY == 0 or abs_pos == total:
            pct = 100.0 * abs_pos / total
            logger.info("Processados: %d/%d (%.1f%%)", abs_pos, total, pct)

    if milvus_buf["embedding"]:
        _flush_batch(collection, milvus_buf)
        total_inserted += len(milvus_buf["embedding"])
        _write_checkpoint(total)
        logger.info(
            "Inseridos %d vetores no Milvus (total: %d)",
            len(milvus_buf["embedding"]),
            total_inserted,
        )


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    _connect_milvus()
    collection = _ensure_collection()

    engine = _build_engine()
    df = _load_and_label(engine)
    sample_df = _stratified_sample(df, EMBED_LIMIT)

    _run_pipeline(sample_df, collection)
    _build_index_and_load(collection)

    logger.info(
        "Pipeline concluído. Vetores indexados: %d",
        collection.num_entities,
    )


if __name__ == "__main__":
    main()
