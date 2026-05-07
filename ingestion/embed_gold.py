"""Sprint 5 — Silver → Gold: embedding pipeline.

Reads logs from PostgreSQL, stratified-samples EMBED_LIMIT rows,
generates 768-dim embeddings via Ollama (nomic-embed-text), and
indexes them in Milvus. Supports checkpoint-based resumption.
"""

import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
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

OLLAMA_BASE: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL: str = "nomic-embed-text"
EMBED_DIM: int = 768

EMBED_LIMIT: int = int(os.getenv("EMBED_LIMIT", "50000"))
MILVUS_BATCH: int = 500
LOG_EVERY: int = 1000
CHECKPOINT_PATH: Path = Path("ingestion/.embed_checkpoint")


# ── PostgreSQL ────────────────────────────────────────────────────────────────

def _build_engine() -> Engine:
    url = (
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    return create_engine(url)


def _load_and_label(engine: Engine) -> pd.DataFrame:
    """Load logs from Silver and assign severity labels."""
    logger.info("Carregando logs do PostgreSQL...")
    query = text("""
        SELECT source, machine_id, alarm_code, event_type, duration_ms, raw_timestamp
        FROM logs
    """)
    df = pd.read_sql(query, engine)
    logger.info("  %d linhas carregadas.", len(df))
    return _assign_severity(df)


def _assign_severity(df: pd.DataFrame) -> pd.DataFrame:
    """Same severity rules as mlflow/preprocess.py (alarm_frequency computed globally)."""
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
    """Proportional stratified sample, deterministic (random_state=42)."""
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
    """Create collection if absent — idempotent."""
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
    # ORM Collection.insert() expects List[List] in schema field order, not dict[str, list]
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


# ── Ollama ────────────────────────────────────────────────────────────────────

def _embed_with_retry(log_text: str) -> list[float] | None:
    """Request embedding from Ollama, retrying up to 3× on ConnectionError."""
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{OLLAMA_BASE}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": log_text},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
        except requests.ConnectionError:
            wait = 5
            logger.warning(
                "Ollama indisponível (tentativa %d/3). Aguardando %ds...",
                attempt + 1,
                wait,
            )
            if attempt < 2:
                time.sleep(wait)
    logger.error("Ollama não respondeu após 3 tentativas. Pulando registro.")
    return None


def _build_log_text(row: pd.Series) -> str:
    event = row["event_type"] if pd.notna(row["event_type"]) else "N/A"
    return (
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
    """Main embedding loop with checkpoint and batched Milvus inserts.

    Checkpoint tracks rows iterated (not rows inserted) so failed Ollama
    calls don't cause duplicate inserts on restart.
    """
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

    buf: dict[str, list] = {
        "source": [], "machine_id": [], "alarm_code": [],
        "event_type": [], "severity": [], "log_text": [], "embedding": [],
    }
    milvus_inserted: int = offset

    for local_i, row in remaining.iterrows():
        abs_pos: int = offset + int(local_i) + 1  # 1-indexed position in full sample_df

        log_text = _build_log_text(row)
        embedding = _embed_with_retry(log_text)

        if embedding is not None:
            event = str(row["event_type"]) if pd.notna(row["event_type"]) else ""
            buf["source"].append(str(row["source"])[:10])
            buf["machine_id"].append(str(row["machine_id"])[:20])
            buf["alarm_code"].append(str(row["alarm_code"])[:20])
            buf["event_type"].append(event[:30])
            buf["severity"].append(str(row["severity"])[:10])
            buf["log_text"].append(log_text[:500])
            buf["embedding"].append(embedding)

        if len(buf["embedding"]) >= MILVUS_BATCH:
            _flush_batch(collection, buf)
            milvus_inserted += len(buf["embedding"])
            _write_checkpoint(abs_pos)
            logger.info(
                "Inseridos %d vetores no Milvus (total: %d)",
                len(buf["embedding"]),
                milvus_inserted,
            )
            for k in buf:
                buf[k].clear()

        if abs_pos % LOG_EVERY == 0:
            pct = 100.0 * abs_pos / total
            logger.info("Processados: %d/%d (%.1f%%)", abs_pos, total, pct)

    if buf["embedding"]:
        _flush_batch(collection, buf)
        milvus_inserted += len(buf["embedding"])
        _write_checkpoint(total)
        logger.info(
            "Inseridos %d vetores no Milvus (total: %d)",
            len(buf["embedding"]),
            milvus_inserted,
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
        "Sprint 5 concluída. Vetores indexados: %d",
        collection.num_entities,
    )


if __name__ == "__main__":
    main()
