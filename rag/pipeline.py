"""Sprint 6 — RAG pipeline: orchestrates retrieve → generate."""

import logging
import sys

from dotenv import load_dotenv

from rag.generator import DEFAULT_MODEL, generate, generate_baseline
from rag.query_parser import parse_query
from rag.retriever import retrieve, retrieve_hybrid

# Modos de operação (nomes do TCC): LLM sem RAG, RAG vetorial e RAG com filtragem.
MODES = ("sem_rag", "vetorial", "filtrada")

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
    mode: str | None = None,
) -> dict:
    """Run the full RAG pipeline for a technician query.

    Args:
        user_input: Natural language question.
        top_k: Number of similar telemetry windows to retrieve.
        model: LLM model identifier (Ollama name or "gemini").
        use_hybrid: If True, use retrieve_hybrid (filtro escalar station/timestamp
            + multi-query para cross-station). Default False (vector search puro).
        mode: "sem_rag" | "vetorial" | "filtrada". Quando informado, tem precedência sobre
            use_hybrid ("filtrada" = use_hybrid=True). "sem_rag" não recupera episódios.

    Returns:
        Dict with keys: question, context, answer, retrieval_scores, model_used,
        token_usage, mode, parsed (estações/janela detectadas pelo parser da pergunta).
    """
    if mode is None:
        mode = "filtrada" if use_hybrid else "vetorial"
    if mode not in MODES:
        raise ValueError(f"Modo inválido: {mode!r}. Use: {', '.join(MODES)}.")

    md = parse_query(user_input)
    parsed = {
        "stations": md.stations,
        "timestamp": md.timestamp,
        "window_lo": md.window_lo,
        "window_hi": md.window_hi,
    }
    logger.info("Iniciando pipeline RAG para: %r (model=%s, mode=%s)", user_input, model, mode)

    if mode == "sem_rag":
        context: list[dict] = []
        answer, token_usage = generate_baseline(user_input, model=model)
    else:
        retriever_fn = retrieve_hybrid if mode == "filtrada" else retrieve
        context = retriever_fn(user_input, top_k=top_k)
        answer, token_usage = generate(user_input, context, model=model)
    return {
        "question": user_input,
        "context": context,
        "answer": answer,
        "retrieval_scores": [hit["score"] for hit in context],
        "model_used": model,
        "token_usage": token_usage,
        "mode": mode,
        "parsed": parsed,
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

    usage = result.get("token_usage", {})
    print("\nTOKENS UTILIZADOS:")
    print(f"  Prompt:     {usage.get('prompt_tokens', 0):>6}")
    print(f"  Resposta:   {usage.get('completion_tokens', 0):>6}")
    print(f"  Total:      {usage.get('total_tokens', 0):>6}")
    print("=" * 60 + "\n")
