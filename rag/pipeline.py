"""Sprint 6 — RAG pipeline: orchestrates retrieve → generate."""

import logging
import sys

from dotenv import load_dotenv

from rag.generator import DEFAULT_MODEL, generate
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
    model: str = DEFAULT_MODEL,
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
    args = sys.argv[1:]
    use_hybrid = False
    if "--hybrid" in args:
        use_hybrid = True
        args = [a for a in args if a != "--hybrid"]

    if not args or not args[0].strip():
        print(
            'Uso: make rag-query QUERY="sua pergunta" [FLAGS="--hybrid"]',
            file=sys.stderr,
        )
        sys.exit(1)

    result = query(args[0], use_hybrid=use_hybrid)

    print("\n" + "=" * 60)
    print(f"PERGUNTA: {result['question']}")
    print("=" * 60)

    print("\nLOGS RECUPERADOS:")
    for rank, hit in enumerate(result["context"], start=1):
        sub = hit.get("current_sub_task") or ""
        sub_part = f" | Sub-task: {sub}" if sub else ""
        print(
            f"  Rank {rank} (score: {hit['score']:.4f}):"
            f" Estação {hit['station']}"
            f" | {hit['event_timestamp']}"
            f" | State: {hit['current_state']}"
            f" | Task: {hit.get('current_task') or 'idle'}"
            f"{sub_part}"
        )

    print("\nRESPOSTA DO ASSISTENTE:")
    print(result["answer"])
    print("=" * 60 + "\n")
