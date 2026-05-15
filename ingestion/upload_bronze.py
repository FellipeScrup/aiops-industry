"""Upload Smart Factory log files from local data/bronze/ into MinIO bucket 'bronze'."""

import logging
import os
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

MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ROOT_USER: str = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_ROOT_PASSWORD: str = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
BUCKET: str = "bronze"

_DATASET_DIR = Path("data/bronze/14441997")

# (local_path, s3_key)  — prefixo smartfactory/ organiza datasets futuros no mesmo bucket
FILES: list[tuple[Path, str]] = [
    (_DATASET_DIR / "training_tenhertz_log_20230411-095748.txt", "smartfactory/logs/training.txt"),
    (_DATASET_DIR / "test_tenhertz_log_20230411-103455.txt",     "smartfactory/logs/test.txt"),
    (_DATASET_DIR / "camunda-activity.json",                     "smartfactory/bpmn/camunda-activity.json"),
]


def get_client() -> boto3.client:
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ROOT_USER,
        aws_secret_access_key=MINIO_ROOT_PASSWORD,
    )


def ensure_bucket(client: boto3.client, bucket: str) -> None:
    try:
        client.head_bucket(Bucket=bucket)
        logger.info("Bucket '%s' já existe.", bucket)
    except ClientError:
        client.create_bucket(Bucket=bucket)
        logger.info("Bucket '%s' criado.", bucket)


def upload_file(client: boto3.client, local_path: Path, s3_key: str) -> None:
    if not local_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {local_path}")

    size_mb = local_path.stat().st_size / (1024 * 1024)
    logger.info(
        "Iniciando upload: %s (%.2f MB) → s3://%s/%s",
        local_path,
        size_mb,
        BUCKET,
        s3_key,
    )
    client.upload_file(str(local_path), BUCKET, s3_key)
    logger.info("Upload concluído: s3://%s/%s", BUCKET, s3_key)


def main() -> None:
    logger.info("Conectando ao MinIO em %s", MINIO_ENDPOINT)
    try:
        client = get_client()
        ensure_bucket(client, BUCKET)

        for local_path, s3_key in FILES:
            upload_file(client, local_path, s3_key)

    except (BotoCoreError, ClientError) as exc:
        logger.error("Erro de conexão com MinIO: %s", exc)
        raise
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        raise

    logger.info(
        "Bronze upload concluído. %d arquivo(s) enviado(s) para s3://%s/smartfactory/",
        len(FILES),
        BUCKET,
    )


if __name__ == "__main__":
    main()
