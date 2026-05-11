"""Bronze → Silver: reads PIADE sequences_1h_data.csv from MinIO and persists to PostgreSQL."""

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
    INSERT INTO piade_telemetry
        (equipment_id, interval_start, count_sum, num_changes,
         pct_idle, pct_production, pct_downtime, pct_perf_loss, pct_sched_downtime)
    VALUES (:equipment_id, :interval_start, :count_sum, :num_changes,
            :pct_idle, :pct_production, :pct_downtime, :pct_perf_loss, :pct_sched_downtime)
    ON CONFLICT (equipment_id, interval_start) DO NOTHING
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


def transform_piade_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """Selects and normalizes the key telemetry columns from sequences_1h_data.csv."""
    return pd.DataFrame({
        "equipment_id":      df["equipment_ID"].astype(str),
        "interval_start":    pd.to_datetime(df["interval_start"], utc=True),
        "count_sum":         pd.to_numeric(df["count_sum"], errors="coerce"),
        "num_changes":       pd.to_numeric(df["#changes"], errors="coerce"),
        "pct_idle":          pd.to_numeric(df["%idle"], errors="coerce"),
        "pct_production":    pd.to_numeric(df["%production"], errors="coerce"),
        "pct_downtime":      pd.to_numeric(df["%downtime"], errors="coerce"),
        "pct_perf_loss":     pd.to_numeric(df["%performance_loss"], errors="coerce"),
        "pct_sched_downtime":pd.to_numeric(df["%scheduled_downtime"], errors="coerce"),
    })


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


def count_telemetry(engine: Engine) -> int:
    with engine.connect() as conn:
        row = conn.execute(text("SELECT COUNT(*) FROM piade_telemetry")).one()
        return int(row[0])


def ingest_dataframe(engine: Engine, df: pd.DataFrame) -> int:
    """Insert df into piade_telemetry in chunks. Returns total rows attempted."""
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
            "PIADE chunk %d/%d: %d linhas tentadas (acumulado: %d)",
            i + 1,
            n_chunks,
            len(records),
            total_attempted,
        )

    return total_attempted


def main() -> None:
    logger.info("Iniciando pipeline Bronze → Silver (PIADE sequences_1h_data)")

    try:
        minio = get_minio_client()
        engine = get_engine()
    except (BotoCoreError, ClientError, SQLAlchemyError) as exc:
        logger.error("Falha ao conectar aos serviços: %s", exc)
        raise

    try:
        raw = read_csv_from_minio(minio, "piade/sequences_1h_data.csv")
        df = transform_piade_sequences(raw)
        attempted = ingest_dataframe(engine, df)
        in_db = count_telemetry(engine)
        logger.info(
            "PIADE — tentadas: %d | no banco: %d | duplicatas ignoradas: %d",
            attempted,
            in_db,
            attempted - in_db,
        )
    except Exception as exc:
        logger.error("Erro ao processar PIADE: %s", exc)
        raise

    logger.info("Pipeline Silver concluído. Total em piade_telemetry: %d", in_db)


if __name__ == "__main__":
    main()
