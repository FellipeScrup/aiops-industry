"""Bronze → Silver: reads Smart Factory NDJSON logs from MinIO and persists to PostgreSQL."""

import io
import json
import logging
import os
from typing import Any

import boto3
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

BATCH_SIZE: int = 1_000

_KNOWN_FIELDS = frozenset({
    "id", "station", "timestamp", "current_state",
    "current_task", "current_task_duration", "current_sub_task",
})

INSERT_SQL = text("""
    INSERT INTO smartfactory_logs
        (id, station, event_timestamp, current_state, current_task,
         current_task_duration, current_sub_task, sensors, split)
    VALUES (:id, :station, :event_timestamp, :current_state, :current_task,
            :current_task_duration, :current_sub_task, :sensors, :split)
    ON CONFLICT (id) DO NOTHING
""")

_LOG_FILES: list[tuple[str, str]] = [
    ("smartfactory/logs/training.txt", "train"),
    ("smartfactory/logs/test.txt",     "test"),
]


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


def _parse_line(line: str, split: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None

    sensors = {k: v for k, v in obj.items() if k not in _KNOWN_FIELDS}

    return {
        "id":                    str(obj.get("id", ""))[:36],
        "station":               str(obj.get("station", ""))[:20],
        "event_timestamp":       obj.get("timestamp"),
        "current_state":         str(obj.get("current_state", ""))[:20],
        "current_task":          obj.get("current_task"),
        "current_task_duration": obj.get("current_task_duration"),
        "current_sub_task":      obj.get("current_sub_task"),
        "sensors":               json.dumps(sensors),
        "split":                 split,
    }


def ingest_stream(engine: Engine, client: boto3.client, s3_key: str, split: str) -> int:
    logger.info("Lendo s3://%s/%s (split=%s)...", BUCKET, s3_key, split)
    response = client.get_object(Bucket=BUCKET, Key=s3_key)
    stream = io.TextIOWrapper(response["Body"], encoding="utf-8")

    batch: list[dict[str, Any]] = []
    total_attempted = 0
    line_num = 0

    for raw_line in stream:
        line_num += 1
        record = _parse_line(raw_line, split)
        if record is None:
            continue
        batch.append(record)

        if len(batch) >= BATCH_SIZE:
            with engine.begin() as conn:
                conn.execute(INSERT_SQL, batch)
            total_attempted += len(batch)
            logger.info(
                "  %s — %d linhas lidas | %d inseridas (acumulado)",
                s3_key, line_num, total_attempted,
            )
            batch.clear()

    if batch:
        with engine.begin() as conn:
            conn.execute(INSERT_SQL, batch)
        total_attempted += len(batch)

    logger.info(
        "  %s concluído: %d linhas lidas | %d inseridas (acumulado)",
        s3_key, line_num, total_attempted,
    )
    return total_attempted


def count_logs(engine: Engine) -> int:
    with engine.connect() as conn:
        row = conn.execute(text("SELECT COUNT(*) FROM smartfactory_logs")).one()
        return int(row[0])


def main() -> None:
    logger.info("Iniciando pipeline Bronze → Silver (Smart Factory Logs)")

    try:
        minio = get_minio_client()
        engine = get_engine()
    except (BotoCoreError, ClientError, SQLAlchemyError) as exc:
        logger.error("Falha ao conectar aos serviços: %s", exc)
        raise

    total = 0
    for s3_key, split in _LOG_FILES:
        try:
            n = ingest_stream(engine, minio, s3_key, split)
            total += n
        except Exception as exc:
            logger.error("Erro ao processar %s: %s", s3_key, exc)
            raise

    in_db = count_logs(engine)
    logger.info(
        "Pipeline Silver concluído. Tentadas: %d | No banco: %d | Duplicatas ignoradas: %d",
        total, in_db, total - in_db,
    )


if __name__ == "__main__":
    main()
