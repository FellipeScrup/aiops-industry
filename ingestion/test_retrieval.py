"""Validation: vector search in Milvus smartfactory_episodes collection."""

import logging
import os
import sys

from dotenv import load_dotenv
from fastembed import TextEmbedding
from pymilvus import Collection, connections

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT: str = os.getenv("MILVUS_PORT", "19530")
EMBED_MODEL_NAME: str = "nomic-ai/nomic-embed-text-v1.5"
COLLECTION_NAME: str = "smartfactory_episodes"
TOP_K: int = 5


def _field(hit, key, default=""):
    """pymilvus 2.4.x Hit.get() aceita só o nome do campo (sem default)."""
    value = hit.entity.get(key)
    return default if value is None else value

_embed_model: TextEmbedding | None = None


def _get_embed_model() -> TextEmbedding:
    global _embed_model
    if _embed_model is None:
        _embed_model = TextEmbedding(model_name=EMBED_MODEL_NAME)
    return _embed_model


def _get_query_embedding(query: str) -> list[float]:
    model = _get_embed_model()
    prefixed = f"search_query: {query}"
    embeddings = list(model.embed([prefixed]))
    return embeddings[0].tolist()


def search(query: str) -> None:
    """Embed query and print top-K similar log events from Milvus."""
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection(COLLECTION_NAME)
    collection.load()

    logger.info("Gerando embedding para a query...")
    embedding = _get_query_embedding(query)

    results = collection.search(
        data=[embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 16}},
        limit=TOP_K,
        output_fields=[
            "station", "event_timestamp", "current_state",
            "current_task", "current_sub_task", "duration_s", "is_anomaly",
        ],
    )

    print(f"\nQuery: {query!r}\n")
    for rank, hit in enumerate(results[0], start=1):
        station = _field(hit, "station", "?")
        ts      = _field(hit, "event_timestamp", "?")
        state   = _field(hit, "current_state", "?")
        task    = _field(hit, "current_task") or "idle"
        subtask = _field(hit, "current_sub_task")
        dur     = float(_field(hit, "duration_s", 0.0))
        anomaly = " [ANOMALIA]" if _field(hit, "is_anomaly", False) else ""
        print(f"Rank {rank} (score: {hit.score:.4f}){anomaly}: {station} | {ts} | {dur:.1f}s | State: {state}")
        print(f"  → Task: {task[:80]}")
        if subtask:
            print(f"     Sub-task: {subtask[:60]}\n")
        else:
            print()


def main() -> None:
    if len(sys.argv) < 2:
        for query in [
            "VGR_1 not ready during calibration",
            "workpiece transport to oven",
        ]:
            search(query)
            print("-" * 60)
    else:
        search(sys.argv[1])


if __name__ == "__main__":
    main()
