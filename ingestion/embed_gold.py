"""Silver → Gold: embedding pipeline for PIADE telemetry windows.

Reads 1h telemetry windows from PostgreSQL (piade_telemetry), generates
768-dim embeddings via fastembed (nomic-embed-text-v1.5), and indexes them
in Milvus. Supports checkpoint-based resumption.
"""

import logging
import os
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
COLLECTION_NAME: str = "piade_telemetry"

EMBED_MODEL_NAME: str = "nomic-ai/nomic-embed-text-v1.5"
EMBED_DIM: int = 768

EMBED_LIMIT: int = int(os.getenv("EMBED_LIMIT", "50000"))
EMBED_BATCH: int = 256
MILVUS_BATCH: int = 500
LOG_EVERY: int = 1000
CHECKPOINT_PATH: Path = Path("ingestion/.embed_checkpoint")


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
    embeddings = list(model.embed(texts))
    return [e.tolist() for e in embeddings]


# ── PostgreSQL ────────────────────────────────────────────────────────────────

def _build_engine() -> Engine:
    url = (
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    return create_engine(url)


def _load_telemetry(engine: Engine) -> pd.DataFrame:
    logger.info("Carregando telemetria do PostgreSQL (piade_telemetry)...")
    query = text("""
        SELECT
            equipment_id,
            interval_start,
            count_sum,
            num_changes,
            pct_idle,
            pct_production,
            pct_downtime,
            pct_perf_loss,
            pct_sched_downtime
        FROM piade_telemetry
        ORDER BY equipment_id, interval_start
    """)
    df = pd.read_sql(query, engine)
    logger.info("  %d janelas carregadas.", len(df))
    return df


def _sample(df: pd.DataFrame, total: int) -> pd.DataFrame:
    if len(df) <= total:
        logger.info("Dataset completo (%d janelas) será indexado.", len(df))
        return df.reset_index(drop=True)
    sampled = df.sample(n=total, random_state=42).reset_index(drop=True)
    logger.info("Amostra aleatória: %d/%d janelas.", len(sampled), len(df))
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
        FieldSchema(name="id",             dtype=DataType.INT64,         is_primary=True, auto_id=True),
        FieldSchema(name="machine_id",     dtype=DataType.VARCHAR,       max_length=20),
        FieldSchema(name="interval_start", dtype=DataType.VARCHAR,       max_length=30),
        FieldSchema(name="pct_idle",       dtype=DataType.FLOAT),
        FieldSchema(name="pct_downtime",   dtype=DataType.FLOAT),
        FieldSchema(name="pct_perf_loss",  dtype=DataType.FLOAT),
        FieldSchema(name="count_sum",      dtype=DataType.FLOAT),
        FieldSchema(name="log_text",       dtype=DataType.VARCHAR,       max_length=500),
        FieldSchema(name="embedding",      dtype=DataType.FLOAT_VECTOR,  dim=EMBED_DIM),
    ]
    schema = CollectionSchema(fields=fields, description="PIADE 1h telemetry window embeddings")
    col = Collection(name=COLLECTION_NAME, schema=schema)
    logger.info("Collection '%s' criada.", COLLECTION_NAME)
    return col


def _flush_batch(collection: Collection, buf: dict[str, list]) -> None:
    data = [
        buf["machine_id"],
        buf["interval_start"],
        buf["pct_idle"],
        buf["pct_downtime"],
        buf["pct_perf_loss"],
        buf["count_sum"],
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
    idle     = float(row["pct_idle"] or 0) * 100
    down     = float(row["pct_downtime"] or 0) * 100
    perf     = float(row["pct_perf_loss"] or 0) * 100
    count    = float(row["count_sum"] or 0)
    changes  = float(row["num_changes"] or 0)
    ts       = str(row["interval_start"])[:19]  # truncate to minute precision
    return (
        f"search_document: "
        f"Máquina {row['equipment_id']} | "
        f"Janela {ts} | "
        f"Downtime: {down:.1f}% | "
        f"Idle: {idle:.1f}% | "
        f"Perda de Performance: {perf:.1f}% | "
        f"Ocorrências de alarme: {count:.0f} | "
        f"Mudanças de estado: {changes:.0f}"
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
        "Milvus: %d vetores existentes. Checkpoint: %d/%d janelas processadas.",
        collection.num_entities,
        offset,
        total,
    )

    if offset >= total:
        logger.info("Nada a processar — todas as %d janelas já foram inseridas.", total)
        return

    remaining = df.iloc[offset:].reset_index(drop=True)
    milvus_buf: dict[str, list] = {
        "machine_id": [], "interval_start": [],
        "pct_idle": [], "pct_downtime": [], "pct_perf_loss": [],
        "count_sum": [], "log_text": [], "embedding": [],
    }
    total_inserted = offset

    for batch_start in range(0, len(remaining), EMBED_BATCH):
        batch = remaining.iloc[batch_start : batch_start + EMBED_BATCH]

        texts = [_build_log_text(row) for _, row in batch.iterrows()]
        embeddings = _embed_batch(texts)

        for i, (_, row) in enumerate(batch.iterrows()):
            milvus_buf["machine_id"].append(str(row["equipment_id"])[:20])
            milvus_buf["interval_start"].append(str(row["interval_start"])[:30])
            milvus_buf["pct_idle"].append(float(row["pct_idle"] or 0))
            milvus_buf["pct_downtime"].append(float(row["pct_downtime"] or 0))
            milvus_buf["pct_perf_loss"].append(float(row["pct_perf_loss"] or 0))
            milvus_buf["count_sum"].append(float(row["count_sum"] or 0))
            milvus_buf["log_text"].append(texts[i][:500])
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
    df = _load_telemetry(engine)
    sample_df = _sample(df, EMBED_LIMIT)

    _run_pipeline(sample_df, collection)
    _build_index_and_load(collection)

    logger.info(
        "Pipeline concluído. Vetores indexados: %d",
        collection.num_entities,
    )


if __name__ == "__main__":
    main()
