"""RAG generator: multi-backend LLM support (Ollama + Gemini) for telemetry analysis."""

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
Você é um engenheiro de confiabilidade especializado em equipamentos industriais de embalagem.

Com base nos registros históricos de telemetria (janelas de 1 hora) similares abaixo, \
analise o estado de funcionamento da máquina e responda à pergunta do técnico.

REGISTROS DE TELEMETRIA SIMILARES (janelas de 1h):
{context_str}

PERGUNTA DO TÉCNICO:
{query}

Responda em português, de forma clara e objetiva. Inclua:
1. Diagnóstico: o que a telemetria indica sobre o comportamento da máquina
2. Causa provável da degradação de desempenho (se houver)
3. Ação corretiva recomendada

Se não houver informação suficiente nos registros, diga isso claramente."""


def _format_context(context: list[dict]) -> str:
    lines: list[str] = []
    for rank, hit in enumerate(context, start=1):
        idle  = hit["pct_idle"] * 100
        down  = hit["pct_downtime"] * 100
        perf  = hit["pct_perf_loss"] * 100
        lines.append(
            f"- [Rank {rank}, similaridade: {hit['score']:.2f}]"
            f" Máquina {hit['machine_id']} | Janela {hit['interval_start']}"
        )
        lines.append(f"   Downtime: {down:.1f}%")
        lines.append(f"   Idle: {idle:.1f}%")
        lines.append(f"   Perda de Performance: {perf:.1f}%")
        lines.append(f"   Ocorrências de alarme: {hit['count_sum']:.0f}")
    return "\n".join(lines)


def _generate_ollama(prompt: str, model: str) -> str:
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

    return resp.json()["response"]


def _generate_gemini(prompt: str) -> str:
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
            raise ValueError("GEMINI_API_KEY inválida. Verifique sua chave no .env.") from exc
        if "quota" in error_msg.lower() or "RESOURCE_EXHAUSTED" in error_msg:
            raise RuntimeError("Quota do Gemini esgotada. Tente novamente mais tarde.") from exc
        raise RuntimeError(f"Erro na API do Gemini: {error_msg}") from exc


def generate(query: str, context: list[dict], model: str = DEFAULT_MODEL) -> str:
    """Assemble prompt and call the selected LLM backend.

    Args:
        query: Original technician question.
        context: Retrieved telemetry dicts from retriever.retrieve().
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
