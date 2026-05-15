"""Gold export: ML features parquet + Milvus index metadata → MinIO gold bucket.

Destinations:
  s3://gold/smartfactory/processed_smartfactory.parquet  — features para AD/ML
  s3://gold/smartfactory/embed_index_meta.json           — metadados do índice Milvus
"""

import io
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

MINIO_ENDPOINT:      str = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ROOT_USER:     str = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_ROOT_PASSWORD: str = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
MILVUS_HOST:         str = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT:         str = os.getenv("MILVUS_PORT", "19530")

BUCKET: str = "gold"

PARQUET_LOCAL: Path = Path("data/silver/processed_smartfactory.parquet")
PARQUET_KEY:   str  = "smartfactory/processed_smartfactory.parquet"
META_KEY:      str  = "smartfactory/embed_index_meta.json"

COLLECTION_NAME: str = "smartfactory_logs"
EMBED_MODEL:     str = "nomic-ai/nomic-embed-text-v1.5"
EMBED_DIM:       int = 768


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


def _upload_file(client, local_path: Path, bucket: str, key: str) -> None:
    size_mb = local_path.stat().st_size / (1024 * 1024)
    logger.info("Enviando %s (%.2f MB) → s3://%s/%s ...", local_path.name, size_mb, bucket, key)
    client.upload_file(str(local_path), bucket, key)
    logger.info("Upload concluído: s3://%s/%s", bucket, key)


def _upload_json(client, data: dict, bucket: str, key: str) -> None:
    body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    logger.info("Enviando metadata JSON (%.1f KB) → s3://%s/%s ...", len(body) / 1024, bucket, key)
    client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    logger.info("Upload concluído: s3://%s/%s", bucket, key)


# ── Milvus metadata ───────────────────────────────────────────────────────────

def _collect_milvus_meta() -> dict:
    """Lê metadados do índice Milvus sem transferir os vetores."""
    try:
        from pymilvus import Collection, connections  # noqa: PLC0415
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        col = Collection(COLLECTION_NAME)
        col.load()
        num_entities = col.num_entities
        indexes = [
            {
                "field":       idx.field_name,
                "index_type":  idx.params.get("index_type", ""),
                "metric_type": idx.params.get("metric_type", ""),
                "params":      {k: v for k, v in idx.params.items()
                                if k not in ("index_type", "metric_type")},
            }
            for idx in col.indexes
        ]
        logger.info("Milvus: %d vetores indexados na collection '%s'.", num_entities, COLLECTION_NAME)
        return {
            "collection":    COLLECTION_NAME,
            "num_entities":  num_entities,
            "embed_model":   EMBED_MODEL,
            "embed_dim":     EMBED_DIM,
            "indexes":       indexes,
            "exported_at":   datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        logger.warning("Milvus indisponível — gravando metadata parcial: %s", exc)
        return {
            "collection":  COLLECTION_NAME,
            "embed_model": EMBED_MODEL,
            "embed_dim":   EMBED_DIM,
            "error":       str(exc),
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("Iniciando Gold export → s3://%s/smartfactory/", BUCKET)

    try:
        client = _get_minio_client()
        _ensure_bucket(client, BUCKET)
    except (BotoCoreError, ClientError) as exc:
        logger.error("Erro ao conectar ao MinIO: %s", exc)
        raise

    # 1. ML features parquet
    if PARQUET_LOCAL.exists():
        _upload_file(client, PARQUET_LOCAL, BUCKET, PARQUET_KEY)
    else:
        logger.warning(
            "Parquet não encontrado em %s — execute 'make preprocess' primeiro.", PARQUET_LOCAL,
        )

    # 2. Milvus index metadata
    meta = _collect_milvus_meta()
    _upload_json(client, meta, BUCKET, META_KEY)

    logger.info("Gold export concluído.")


if __name__ == "__main__":
    main()
