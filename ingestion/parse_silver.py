"""Bronze → Silver: reads CSVs from MinIO, normalizes, and persists to PostgreSQL."""

import io
import logging
import os
from typing import Any

import boto3
import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# MinIO
MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ROOT_USER: str = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_ROOT_PASSWORD: str = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
BUCKET: str = "bronze"

# PostgreSQL
POSTGRES_USER: str = os.getenv("POSTGRES_USER", "aiops")
POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "aiops")
POSTGRES_DB: str = os.getenv("POSTGRES_DB", "aiops_industry")
POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")

CHUNK_SIZE: int = 50_000

INSERT_SQL = text("""
    INSERT INTO logs (source, machine_id, alarm_code, event_type, duration_ms, raw_timestamp)
    VALUES (:source, :machine_id, :alarm_code, :event_type, :duration_ms, :raw_timestamp)
    ON CONFLICT (source, machine_id, alarm_code, raw_timestamp) DO NOTHING
""")


def get_minio_client() -> boto3.client:
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ROOT_USER,
        aws_secret_access_key=MINIO_ROOT_PASSWORD,
    )


def get_engine() -> Engine:
    url = (
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    return create_engine(url, pool_pre_ping=True)


def read_csv_from_minio(client: boto3.client, s3_key: str) -> pd.DataFrame:
    logger.info("Lendo s3://%s/%s ...", BUCKET, s3_key)
    response = client.get_object(Bucket=BUCKET, Key=s3_key)
    df = pd.read_csv(io.BytesIO(response["Body"].read()))
    logger.info("  %d linhas lidas de %s", len(df), s3_key)
    return df


def _clean_records(chunk: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert chunk to list of dicts, replacing NaN/NA/NaT with None for SQL."""
    records = chunk.to_dict(orient="records")
    for row in records:
        for k, v in row.items():
            try:
                if pd.isna(v):
                    row[k] = None
            except (TypeError, ValueError):
                pass
    return records


def transform_alpi(df: pd.DataFrame) -> pd.DataFrame:
    # alarm column is integer (1-154); stored as string for unified alarm_code
    return pd.DataFrame({
        "source": "ALPI",
        "machine_id": df["serial"].astype(str),
        "alarm_code": df["alarm"].astype(int).astype(str),
        "event_type": None,
        "duration_ms": None,
        "raw_timestamp": pd.to_datetime(df["timestamp"], utc=True),
    })


def transform_piade(df: pd.DataFrame) -> pd.DataFrame:
    # elapsed stored as-is (unit per source dataset); nullable for non-downtime rows
    return pd.DataFrame({
        "source": "PIADE",
        "machine_id": df["equipment_ID"].astype(str),
        "alarm_code": df["alarm"].astype(str),
        "event_type": df["type"].astype(str),
        "duration_ms": pd.to_numeric(df["elapsed"], errors="coerce").round().astype("Int64"),
        "raw_timestamp": pd.to_datetime(df["interval_start"], format="ISO8601", utc=True),
    })


def count_by_source(engine: Engine, source: str) -> int:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT COUNT(*) FROM logs WHERE source = :source"),
            {"source": source},
        ).one()
        return int(row[0])


def ingest_dataframe(engine: Engine, df: pd.DataFrame, label: str) -> int:
    """Insert df into logs in chunks. Returns total rows attempted."""
    total_rows = len(df)
    n_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
    total_attempted = 0

    for i in range(n_chunks):
        chunk = df.iloc[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
        records = _clean_records(chunk)

        with engine.begin() as conn:
            conn.execute(INSERT_SQL, records)

        total_attempted += len(records)
        logger.info(
            "%s chunk %d/%d: %d linhas tentadas (acumulado: %d)",
            label,
            i + 1,
            n_chunks,
            len(records),
            total_attempted,
        )

    return total_attempted


def main() -> None:
    logger.info("Iniciando pipeline Bronze → Silver")

    try:
        minio = get_minio_client()
        engine = get_engine()
    except (BotoCoreError, ClientError, SQLAlchemyError) as exc:
        logger.error("Falha ao conectar aos serviços: %s", exc)
        raise

    # ── ALPI ────────────────────────────────────────────────────────────────
    try:
        alpi_raw = read_csv_from_minio(minio, "alpi/alarms.csv")
        alpi_df = transform_alpi(alpi_raw)
        alpi_attempted = ingest_dataframe(engine, alpi_df, "ALPI")
        alpi_in_db = count_by_source(engine, "ALPI")
        logger.info(
            "ALPI — tentadas: %d | no banco: %d | duplicatas ignoradas: %d",
            alpi_attempted,
            alpi_in_db,
            alpi_attempted - alpi_in_db,
        )
    except Exception as exc:
        logger.error("Erro ao processar ALPI: %s", exc)
        raise

    # ── PIADE ───────────────────────────────────────────────────────────────
    try:
        piade_raw = read_csv_from_minio(minio, "piade/raw_data.csv")
        piade_df = transform_piade(piade_raw)
        piade_attempted = ingest_dataframe(engine, piade_df, "PIADE")
        piade_in_db = count_by_source(engine, "PIADE")
        logger.info(
            "PIADE — tentadas: %d | no banco: %d | duplicatas ignoradas: %d",
            piade_attempted,
            piade_in_db,
            piade_attempted - piade_in_db,
        )
    except Exception as exc:
        logger.error("Erro ao processar PIADE: %s", exc)
        raise

    total_in_db = alpi_in_db + piade_in_db
    total_attempted = alpi_attempted + piade_attempted
    logger.info(
        "Pipeline Silver concluído. Total no banco: %d | Total tentado: %d | Ignorados: %d",
        total_in_db,
        total_attempted,
        total_attempted - total_in_db,
    )


if __name__ == "__main__":
    main()
