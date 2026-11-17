"""Interface Gradio do RAG de diagnóstico industrial (TCC Facens 2026).

Cliente fino da API (`make api`, porta 8001). Permite demonstrar ao vivo as três condições
avaliadas no TCC (LLM sem RAG, RAG vetorial e RAG com filtragem) e mostra o que o parser
da pergunta detectou (estação e janela de ±5 s), que é a base do filtro escalar no Milvus.
"""

import html
import os
import re

import gradio as gr
import requests


def _format_answer(text: str) -> str:
    """Normaliza Diagnóstico / Causa / Ação em 3 blocos (sem lista 1. 2. 3. e sem **)."""
    out = (text or "").strip()
    if not out:
        return out
    out = out.replace("**", "")
    # Variantes do prompt: "Causa provável do problema:", "Ação corretiva recomendada:".
    out = re.sub(r"(?i)\s*Diagnóstico\s*:", "\n\nDiagnóstico:", out)
    out = re.sub(
        r"(?i)\s*Causa\s+prov[aá]vel(?:\s+do\s+problema)?(?:\s+ou\s+comportamento\s+observado)?\s*:",
        "\n\nCausa provável:",
        out,
    )
    out = re.sub(
        r"(?i)\s*A[cç][aã]o\s+corretiva(?:\s+recomendada)?\s*:",
        "\n\nAção corretiva:",
        out,
    )
    # Limpa "1." / "2." / "3." órfãos, sem tocar em "motor 3.".
    out = re.sub(r"(?m)^\s*[123][\.\)]\s*$", "", out)
    out = re.sub(r"(?m)^\s*[123][\.\)]\s*(?=Diagnóstico:|Causa|Ação)", "", out)
    out = re.sub(r"(?<=[.!?:;])\s+[123][\.\)]\s*(?=\n\n(?:Diagnóstico|Causa|Ação))", "", out)
    return re.sub(r"\n{3,}", "\n\n", out).strip()

API_URL = os.getenv("API_URL", "http://localhost:8001/query")

# Geradores avaliados no TCC (todos locais, via Ollama) + Llama 3.2 3B, citado na seção 4.3 do TCC
# como alternativa leve de geração local (não faz parte da avaliação).
_MODEL_CHOICES = [
    ("qwen2.5:7b (avaliado)", "qwen2.5:7b"),
    ("qwen3.5:9b (avaliado)", "qwen3.5:9b"),
    ("gemma4:e4b (avaliado)", "gemma4:e4b"),
    ("llama3.2:3b (alternativa leve, fora da avaliação)", "llama3.2:3b"),
]

# (rótulo exibido, valor enviado à API)
_MODE_CHOICES = [
    ("RAG com filtragem", "filtrada"),
    ("RAG vetorial", "vetorial"),
    ("LLM sem RAG", "sem_rag"),
]
_MODE_LABEL = {value: label for label, value in _MODE_CHOICES}

# Perguntas do golden set do TCC (fac-01, cro-01, cau-01) + uma pergunta livre.
_EXAMPLES = [
    ["A estação HBW_1 ficou not ready em 2023-04-11 10:09:48. Qual tarefa e sub-tarefa ela estava executando e quanto tempo durou?",
     5, "gemma4:e4b", "filtrada"],
    ["Por volta de 2023-04-11 10:01, as estações VGR_1 e HBW_1 ficaram not ready quase ao mesmo tempo. O que cada uma estava fazendo e por quanto tempo?",
     5, "gemma4:e4b", "filtrada"],
    ["A calibração do motor 4 da estação HBW_1 iniciada em 2023-04-11 10:29 foi mais longa em relação ao tempo típico?",
     5, "gemma4:e4b", "filtrada"],
    ["Por que a estação VGR_1 ficou parada?", 5, "qwen2.5:7b", "filtrada"],
]

_CSS = """
.ep-card { border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 10px 14px;
           margin-bottom: 10px; background: var(--block-background-fill); }
.ep-card.anom { border-left: 5px solid #d95f02; }
.ep-head { font-weight: 600; margin-bottom: 4px; }
.ep-meta { font-size: 0.9em; opacity: 0.85; }
.ep-badge { display: inline-block; font-size: 0.75em; font-weight: 600; padding: 1px 8px; margin-left: 6px;
            border-radius: 999px; background: #d95f02; color: #fff; vertical-align: middle; }
.ep-card details { margin-top: 6px; font-size: 0.85em; }
.ep-card details summary { cursor: pointer; opacity: 0.8; }
.ep-empty { opacity: 0.75; font-style: italic; }
"""


def _parser_panel(mode: str, parsed: dict, n_hits: int) -> str:
    """Resume o que o parser detectou e se o filtro escalar foi (ou seria) aplicado."""
    if mode == "sem_rag":
        return ("**Modo sem RAG:** nenhum episódio é recuperado. O modelo responde apenas com o "
                "conhecimento próprio, sem acesso aos logs da fábrica.")
    stations = parsed.get("stations") or []
    lo, hi = parsed.get("window_lo"), parsed.get("window_hi")
    st = ", ".join(f"`{s}`" for s in stations) if stations else "nenhuma"
    ts = parsed.get("timestamp")
    if ts:
        win = f"`{ts}` (janela: `{lo}` a `{hi}`)"
    else:
        win = "nenhum horário detectado"
    if mode == "vetorial":
        filtro = "**não aplicado** (modo vetorial: o filtro por estação e janela não é usado)"
    elif stations:
        filtro = "**aplicado** no Milvus (estação" + (" + janela temporal)" if ts else ")")
    else:
        filtro = "**não aplicado** (nenhuma estação citada: busca vetorial pura)"
    return (f"**Parser da pergunta:** estação: {st} | horário: {win}\n\n"
            f"**Filtro escalar:** {filtro} | **Episódios recuperados:** {n_hits}")


def _episode_cards(mode: str, context: list[dict]) -> str:
    if mode == "sem_rag":
        return '<div class="ep-empty">Sem RAG: nenhum episódio foi fornecido ao modelo.</div>'
    if not context:
        return '<div class="ep-empty">Nenhum episódio recuperado.</div>'
    cards = []
    for rank, hit in enumerate(context, start=1):
        esc = lambda v: html.escape(str(v if v not in (None, "") else "-"))  # noqa: E731
        dur = hit.get("duration_s")
        dur_txt = f"{dur:.1f} s" if dur is not None else "-"
        anom = bool(hit.get("is_anomaly"))
        badge = '<span class="ep-badge">ANOMALIA DE DURAÇÃO</span>' if anom else ""
        task = hit.get("current_task") or "idle"
        sub = hit.get("current_sub_task") or ""
        sub_html = f"<br>Sub-tarefa: {esc(sub)}" if sub else ""
        log_text = hit.get("log_text") or ""
        details = (f"<details><summary>Texto do episódio enviado ao LLM</summary>{html.escape(log_text)}</details>"
                   if log_text else "")
        cards.append(
            f'<div class="ep-card{" anom" if anom else ""}">'
            f'<div class="ep-head">#{rank} · {esc(hit.get("station"))} · {esc(hit.get("event_timestamp"))}'
            f'{badge}</div>'
            f'<div class="ep-meta">Duração: {dur_txt} · similaridade: {hit.get("score", 0):.3f}<br>'
            f'Estado: {esc(hit.get("current_state"))}<br>Tarefa: {esc(task)}{sub_html}</div>'
            f'{details}</div>'
        )
    return "".join(cards)


def analyze(question: str, top_k: int, model: str, mode: str) -> tuple[str, str, str, str]:
    """Chama a API e devolve (painel do parser, resposta, cartões de episódios, métricas)."""
    if not question or not question.strip():
        return "", "Digite uma pergunta para analisar.", "", ""
    try:
        resp = requests.post(
            API_URL,
            json={"question": question, "top_k": int(top_k), "model": model, "mode": mode},
            timeout=330,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.ConnectionError:
        return "", "**Erro:** API indisponível. Execute `make api` (porta 8001) e confirme que o Docker e o Ollama estão no ar.", "", ""
    except requests.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        return "", f"**Erro:** {detail}", "", ""
    except Exception as exc:
        return "", f"**Erro inesperado:** {exc}", "", ""

    used_mode = data.get("mode", mode)
    context = data.get("context", [])
    answer = _format_answer(data.get("answer", "") or "") or "_(resposta vazia)_"

    usage = data.get("token_usage", {})
    stats = (f"**Modo:** {_MODE_LABEL.get(used_mode, used_mode)} | "
             f"**Modelo:** {data.get('model_used', model)} | "
             f"**Tempo:** {data.get('processing_time_s', 0):.1f} s")
    if usage:
        stats += (f" | **Tokens:** prompt {usage.get('prompt_tokens', 0)}, "
                  f"resposta {usage.get('completion_tokens', 0)}, total {usage.get('total_tokens', 0)}")

    return (
        _parser_panel(used_mode, data.get("parsed", {}), len(context)),
        answer,
        _episode_cards(used_mode, context),
        stats,
    )


with gr.Blocks(title="AIOps Industry: diagnóstico de paradas com RAG") as demo:
    gr.Markdown(
        "# AIOps Industry: diagnóstico de paradas em linha industrial\n"
        "**RAG local: Milvus + Ollama (Qwen 2.5, Qwen 3.5, Gemma 4)** | "
        "Smart Factory Logs (406 episódios `not ready`) | TCC Facens 2026"
    )

    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Descreva o comportamento da estação",
                placeholder="Ex.: A estação HBW_1 ficou not ready em 2023-04-11 10:09. "
                            "Qual tarefa ela estava executando e quanto tempo durou?",
                lines=3,
            )
            mode_radio = gr.Radio(
                choices=_MODE_CHOICES, value="filtrada", label="Modo de resposta",
            )
            with gr.Row():
                model_dropdown = gr.Dropdown(choices=_MODEL_CHOICES, value="qwen2.5:7b", label="Modelo LLM (local)")
                top_k_slider = gr.Slider(
                    label="Episódios no contexto (top_k)", minimum=1, maximum=10, value=5, step=1,
                )
            analyze_btn = gr.Button("Analisar", variant="primary")

        with gr.Column(scale=3):
            parser_output = gr.Markdown()
            stats_output = gr.Markdown()
            gr.Markdown("### Diagnóstico do assistente")
            answer_output = gr.Markdown()
            gr.Markdown("### Episódios recuperados (evidências enviadas ao LLM)")
            episodes_output = gr.HTML()

    outputs = [parser_output, answer_output, episodes_output, stats_output]
    inputs = [question_input, top_k_slider, model_dropdown, mode_radio]
    analyze_btn.click(fn=analyze, inputs=inputs, outputs=outputs, show_progress="full")
    question_input.submit(fn=analyze, inputs=inputs, outputs=outputs, show_progress="full")

    gr.Examples(examples=_EXAMPLES, inputs=inputs, label="Exemplos (perguntas do golden set do TCC)")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft(), css=_CSS)
