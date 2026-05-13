"""RAG generator: multi-backend LLM support (Ollama + Gemini) for smart factory log analysis."""

import logging
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

OLLAMA_BASE: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL: str = os.getenv("LLM_MODEL", "llama3.2:3b")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

_PROMPT_TEMPLATE = """\
Você é um engenheiro de manutenção especializado em fábricas inteligentes com automação industrial.

Com base nos registros de log abaixo de uma fábrica inteligente com 7 estações \
(MM_1, EC_1, SM_1, HBW_1, OV_1, VGR_1, WT_1), analise o comportamento do sistema \
e responda à pergunta do técnico.

REGISTROS DE LOG SIMILARES:
{context_str}

PERGUNTA DO TÉCNICO:
{query}

Responda em português, de forma clara e objetiva. Inclua:
1. Diagnóstico: o que os logs indicam sobre o comportamento do sistema
2. Causa provável do problema ou comportamento observado
3. Ação corretiva recomendada

Se não houver informação suficiente nos registros, diga isso claramente."""


def _format_context(context: list[dict]) -> str:
    lines: list[str] = []
    for rank, hit in enumerate(context, start=1):
        lines.append(
            f"- [Rank {rank}, similaridade: {hit['score']:.2f}]"
            f" Estação {hit['station']} | {hit['event_timestamp']}"
        )
        lines.append(f"   State: {hit['current_state']}")
        lines.append(f"   Task: {hit.get('current_task', '') or 'idle'}")
        if hit.get("current_sub_task"):
            lines.append(f"   Sub-task: {hit['current_sub_task']}")
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


_GEMINI_RETRY_DELAYS = (5, 15, 45)  # backoff em segundos para 429


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

    delays = iter(_GEMINI_RETRY_DELAYS)
    attempt = 0
    while True:
        attempt += 1
        try:
            response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            return response.text
        except Exception as exc:
            error_msg = str(exc)
            if "API_KEY_INVALID" in error_msg or "API key not valid" in error_msg:
                raise ValueError("GEMINI_API_KEY inválida. Verifique sua chave no .env.") from exc
            is_rate_limit = (
                "429" in error_msg
                or "quota" in error_msg.lower()
                or "RESOURCE_EXHAUSTED" in error_msg
            )
            if is_rate_limit:
                delay = next(delays, None)
                if delay is None:
                    raise RuntimeError(
                        "Rate limit do Gemini após 3 tentativas. "
                        "Aguarde alguns minutos e tente novamente."
                    ) from exc
                logger.warning(
                    "Rate limit Gemini (tentativa %d). Aguardando %ds...", attempt, delay
                )
                time.sleep(delay)
                continue
            raise RuntimeError(f"Erro na API do Gemini: {error_msg}") from exc


def call_raw(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Call the LLM with a raw prompt (no context formatting).

    Same backend routing as generate() — use for golden set generation and
    baseline evaluation where the caller builds the prompt directly.
    """
    if "gemini" in model.lower():
        return _generate_gemini(prompt)
    return _generate_ollama(prompt, model)


def generate(query: str, context: list[dict], model: str = DEFAULT_MODEL) -> str:
    """Assemble prompt and call the selected LLM backend.

    Args:
        query: Original technician question.
        context: Retrieved log event dicts from retriever.retrieve().
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
