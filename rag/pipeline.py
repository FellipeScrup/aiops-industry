"""Sprint 6 — RAG pipeline: orchestrates retrieve → generate."""

import logging
import sys

from dotenv import load_dotenv

from rag.generator import generate
from rag.retriever import retrieve, retrieve_hybrid

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def query(
    user_input: str,
    top_k: int = 5,
    model: str = "llama3.2:3b",
    use_hybrid: bool = False,
) -> dict:
    """Run the full RAG pipeline for a technician query.

    Args:
        user_input: Natural language question.
        top_k: Number of similar telemetry windows to retrieve.
        model: LLM model identifier (Ollama name or "gemini").
        use_hybrid: If True, use retrieve_hybrid (filtro escalar station/timestamp
            + multi-query para cross-station). Default False (vector search puro).

    Returns:
        Dict with keys: question, context, answer, retrieval_scores, model_used.
    """
    logger.info("Iniciando pipeline RAG para: %r (model=%s, hybrid=%s)",
                user_input, model, use_hybrid)
    retriever_fn = retrieve_hybrid if use_hybrid else retrieve
    context = retriever_fn(user_input, top_k=top_k)
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

    print("\nJANELAS DE TELEMETRIA RECUPERADAS:")
    for rank, hit in enumerate(result["context"], start=1):
        print(
            f"  Rank {rank} (score: {hit['score']:.4f}):"
            f" Máquina {hit['machine_id']}"
            f" | Janela {hit['interval_start']}"
            f" | Downtime {hit['pct_downtime'] * 100:.1f}%"
            f" | Idle {hit['pct_idle'] * 100:.1f}%"
            f" | Perf. Loss {hit['pct_perf_loss'] * 100:.1f}%"
        )

    print("\nRESPOSTA DO ASSISTENTE:")
    print(result["answer"])
    print("=" * 60 + "\n")
