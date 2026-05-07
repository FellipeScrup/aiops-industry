"""Sprint 6 — RAG pipeline: orchestrates retrieve → generate."""

import logging
import sys

from dotenv import load_dotenv

from rag.generator import generate
from rag.retriever import retrieve

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def query(user_input: str, top_k: int = 5, model: str = "llama3.2:3b") -> dict:
    """Run the full RAG pipeline for a technician query.

    Args:
        user_input: Natural language question.
        top_k: Number of similar logs to retrieve.
        model: LLM model identifier (Ollama name or "gemini").

    Returns:
        Dict with keys: question, context, answer, retrieval_scores, model_used.
    """
    logger.info("Iniciando pipeline RAG para: %r (model=%s)", user_input, model)
    context = retrieve(user_input, top_k=top_k)
    answer = generate(user_input, context, model=model)
    return {
        "question": user_input,
        "context": context,
        "answer": answer,
        "retrieval_scores": [hit["score"] for hit in context],
        "model_used": model,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Uso: make rag-query QUERY="sua pergunta"', file=sys.stderr)
        sys.exit(1)

    result = query(sys.argv[1])

    print("\n" + "=" * 60)
    print(f"PERGUNTA: {result['question']}")
    print("=" * 60)

    print("\nLOGS RECUPERADOS:")
    for rank, hit in enumerate(result["context"], start=1):
        print(
            f"  Rank {rank} (score: {hit['score']:.4f}):"
            f" Alarme {hit['alarm_code']}"
            f" | Máquina {hit['machine_id']}"
            f" | {hit['severity']}"
        )

    print("\nRESPOSTA DO ASSISTENTE:")
    print(result["answer"])
    print("=" * 60 + "\n")
