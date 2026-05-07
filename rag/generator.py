"""Sprint 6 — RAG generator: format context and call Ollama LLM."""

import logging
import os

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

OLLAMA_BASE: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3.2:3b")

_PROMPT_TEMPLATE = """\
Você é um assistente especialista em manutenção industrial de máquinas de embalagem.

Com base nos logs históricos similares abaixo, responda à pergunta do técnico.

LOGS SIMILARES ENCONTRADOS:
{context_str}

PERGUNTA DO TÉCNICO:
{query}

Responda em português, de forma clara e objetiva. Inclua:
1. O que este alarme/falha significa
2. Causa provável
3. Ação corretiva recomendada

Se não houver informação suficiente nos logs, diga isso claramente."""


def _format_context(context: list[dict]) -> str:
    lines: list[str] = []
    for rank, hit in enumerate(context, start=1):
        event = hit.get("event_type") or "N/A"
        lines.append(
            f"- [Rank {rank}, score: {hit['score']:.2f}, severidade: {hit['severity']}]"
            f" Alarme {hit['alarm_code']} | Máquina {hit['machine_id']}"
            f" | Fonte {hit['source']} | Tipo {event}"
        )
    return "\n".join(lines)


def generate(query: str, context: list[dict]) -> str:
    """Assemble prompt and call Ollama LLM, returning the response text.

    Args:
        query: Original technician question.
        context: Retrieved log dicts from retriever.retrieve().

    Returns:
        LLM response as a string.
    """
    context_str = _format_context(context)
    prompt = _PROMPT_TEMPLATE.format(context_str=context_str, query=query)

    logger.info("Gerando resposta com %s...", LLM_MODEL)

    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
    except requests.ConnectionError as exc:
        raise ConnectionError(
            f"Ollama indisponível em {OLLAMA_BASE}. Verifique se o serviço está rodando."
        ) from exc

    answer: str = resp.json()["response"]
    logger.info("Resposta gerada (%d chars).", len(answer))
    return answer
