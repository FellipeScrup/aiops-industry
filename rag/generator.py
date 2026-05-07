"""Sprint 7 — RAG generator: multi-backend LLM support (Ollama + Gemini)."""

import logging
import os

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

OLLAMA_BASE: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL: str = os.getenv("LLM_MODEL", "llama3.2:3b")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

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
        header = (
            f"- [Rank {rank}, score: {hit['score']:.2f}, severidade: {hit['severity']}]"
            f" Alarme {hit['alarm_code']} | Máquina {hit['machine_id']}"
            f" | Fonte {hit['source']} | Tipo {event}"
        )
        lines.append(header)
        if hit.get("dict_title"):
            lines.append(f"   Descrição técnica: {hit['dict_title']}")
        if hit.get("dict_description"):
            lines.append(f"   Detalhes: {hit['dict_description']}")
        if hit.get("dict_probable_causes"):
            lines.append(f"   Causas prováveis: {hit['dict_probable_causes']}")
        if hit.get("dict_corrective_actions"):
            lines.append(f"   Ações corretivas: {hit['dict_corrective_actions']}")
    return "\n".join(lines)


def _generate_ollama(prompt: str, model: str) -> str:
    """Call local Ollama API."""
    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
    except requests.ConnectionError as exc:
        raise ConnectionError(
            f"Ollama indisponível em {OLLAMA_BASE}. Verifique se o serviço está rodando."
        ) from exc
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            raise ValueError(
                f"Modelo '{model}' não encontrado no Ollama. "
                f"rode: ollama pull {model}"
            ) from exc
        raise

    answer: str = resp.json()["response"]
    return answer


def _generate_gemini(prompt: str) -> str:
    """Call Google Generative AI API."""
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY não definida no .env. "
            "Adicione sua chave para usar o Gemini."
        )

    try:
        from google import genai  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Pacote google-genai não instalado. "
            "Execute: pip install google-genai"
        ) from exc

    client = genai.Client(api_key=GEMINI_API_KEY)

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return response.text
    except Exception as exc:
        error_msg = str(exc)
        if "API_KEY_INVALID" in error_msg or "API key not valid" in error_msg:
            raise ValueError(
                f"GEMINI_API_KEY inválida. Verifique sua chave no .env."
            ) from exc
        if "quota" in error_msg.lower() or "RESOURCE_EXHAUSTED" in error_msg:
            raise RuntimeError(
                f"Quota do Gemini esgotada. Tente novamente mais tarde."
            ) from exc
        raise RuntimeError(f"Erro na API do Gemini: {error_msg}") from exc


def generate(query: str, context: list[dict], model: str = DEFAULT_MODEL) -> str:
    """Assemble prompt and call the selected LLM backend.

    Args:
        query: Original technician question.
        context: Retrieved log dicts from retriever.retrieve().
        model: Model identifier — Ollama model name or "gemini".

    Returns:
        LLM response as a string.
    """
    context_str = _format_context(context)
    prompt = _PROMPT_TEMPLATE.format(context_str=context_str, query=query)

    logger.info("Gerando resposta com %s...", model)

    if "gemini" in model.lower():
        answer = _generate_gemini(prompt)
    else:
        answer = _generate_ollama(prompt, model)

    logger.info("Resposta gerada (%d chars).", len(answer))
    return answer
