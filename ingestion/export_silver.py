"""Silver export: PostgreSQL smartfactory_logs → Parquet → MinIO silver bucket.

Destination: s3://silver/smartfactory/smartfactory_logs.parquet
"""

import io
import logging
import os

import boto3
import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

POSTGRES_USER:     str = os.getenv("POSTGRES_USER", "aiops")
POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "aiops")
POSTGRES_DB:       str = os.getenv("POSTGRES_DB", "aiops_industry")
POSTGRES_HOST:     str = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT:     str = os.getenv("POSTGRES_PORT", "5432")

MINIO_ENDPOINT:      str = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ROOT_USER:     str = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_ROOT_PASSWORD: str = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")

BUCKET:  str = "silver"
S3_KEY:  str = "smartfactory/smartfactory_logs.parquet"

CHUNK_SIZE: int = 50_000


# ── PostgreSQL ────────────────────────────────────────────────────────────────

def _build_engine():
    url = (
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    return create_engine(url, pool_pre_ping=True)


def _load_silver(engine) -> pd.DataFrame:
    logger.info("Carregando smartfactory_logs do PostgreSQL...")
    query = text("""
        SELECT id, station, event_timestamp, current_state,
               current_task, current_task_duration, current_sub_task,
               sensors, split
        FROM smartfactory_logs
        ORDER BY event_timestamp
    """)
    df = pd.read_sql(query, engine)
    logger.info("  %d eventos carregados.", len(df))
    return df


# ── MinIO ─────────────────────────────────────────────────────────────────────

def _get_minio_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ROOT_USER,
        aws_secret_access_key=MINIO_ROOT_PASSWORD,
    )


def _ensure_bucket(client, bucket: str) -> None:
    try:
        client.head_bucket(Bucket=bucket)
        logger.info("Bucket '%s' já existe.", bucket)
    except ClientError:
        client.create_bucket(Bucket=bucket)
        logger.info("Bucket '%s' criado.", bucket)


def _upload_parquet(client, df: pd.DataFrame, bucket: str, key: str) -> None:
    logger.info("Serializando %d linhas para Parquet...", len(df))
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    size_mb = buf.tell() / (1024 * 1024)
    buf.seek(0)

    logger.info(
        "Enviando %.2f MB → s3://%s/%s ...", size_mb, bucket, key,
    )
    client.upload_fileobj(buf, bucket, key)
    logger.info("Upload concluído: s3://%s/%s", bucket, key)


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("Iniciando Silver export: PostgreSQL → s3://%s/%s", BUCKET, S3_KEY)

    try:
        engine = _build_engine()
        df = _load_silver(engine)
    except Exception as exc:
        logger.error("Erro ao ler PostgreSQL: %s", exc)
        raise

    try:
        client = _get_minio_client()
        _ensure_bucket(client, BUCKET)
        _upload_parquet(client, df, BUCKET, S3_KEY)
    except (BotoCoreError, ClientError) as exc:
        logger.error("Erro no MinIO: %s", exc)
        raise

    logger.info(
        "Silver export concluído. %d eventos em s3://%s/%s",
        len(df), BUCKET, S3_KEY,
    )


if __name__ == "__main__":
    main()
