"""Sprint 7 — Gradio UI: interface web para o RAG de diagnóstico industrial."""

import requests
import gradio as gr

API_URL = "http://localhost:8001/query"

_EXAMPLES = [
    ["Por que a estação VGR_1 ficou parada?", 5, "qwen2.5:7b"],
    ["HBW_1 demorou muito calibrando o motor, o que isso indica?", 5, "qwen2.5:7b"],
    ["O que aconteceu na estação OV_1 durante o aquecimento?", 5, "qwen2.5:7b"],
    ["A estação SM_1 ficou not ready, por quê?", 5, "qwen2.5:7b"],
]

_MODEL_CHOICES = ["qwen2.5:7b", "llama3.2:3b", "gemini"]


def analyze(question: str, top_k: int, model: str) -> tuple[str, str, str]:
    """Call RAG API and format results for the three output fields."""
    try:
        resp = requests.post(
            API_URL,
            json={"question": question, "top_k": int(top_k), "model": model},
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.ConnectionError:
        err = "Erro: API indisponível. Execute `make api` antes de usar a interface."
        return err, "", ""
    except requests.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        return f"Erro: {detail}", "", ""
    except Exception as exc:
        return f"Erro inesperado: {exc}", "", ""

    answer: str = data.get("answer", "")
    model_used: str = data.get("model_used", model)

    context_lines: list[str] = []
    for rank, hit in enumerate(data.get("context", []), start=1):
        task = hit.get("current_task") or "idle"
        sub = hit.get("current_sub_task") or ""
        sub_part = f" | Sub-tarefa: {sub}" if sub else ""
        duration = hit.get("duration_s")
        dur_part = f" | {duration:.1f}s" if duration is not None else ""
        anomaly_part = "  [ANOMALIA]" if hit.get("is_anomaly") else ""
        context_lines.append(
            f"Rank {rank} (score: {hit['score']:.4f}){anomaly_part}: "
            f"Estação {hit.get('station', '?')} | {hit.get('event_timestamp', '?')}{dur_part} "
            f"| Estado: {hit.get('current_state', '?')} | Tarefa: {task}{sub_part}"
        )
    context_str = "\n".join(context_lines)

    scores = data.get("retrieval_scores", [])
    scores_str = "  ".join(f"#{i + 1}: {s:.4f}" for i, s in enumerate(scores))
    elapsed = data.get("processing_time_s", 0)
    usage = data.get("token_usage", {})
    scores_str += f"\n\nModelo usado: {model_used} | Tempo: {elapsed:.1f}s"
    if usage:
        scores_str += (
            f"\n\nTokens utilizados:"
            f"\n  Prompt:    {usage.get('prompt_tokens', 0)}"
            f"\n  Resposta:  {usage.get('completion_tokens', 0)}"
            f"\n  Total:     {usage.get('total_tokens', 0)}"
        )

    return answer, context_str, scores_str


with gr.Blocks(title="AIOps Industry — Análise de Telemetria Industrial") as demo:
    gr.Markdown("# AIOps Industry — Análise de Telemetria Industrial")
    gr.Markdown("**RAG local powered by Llama 3.2 / Qwen 2.5 / Gemini + Milvus | TCC Facens 2026**")

    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(
                label="Descreva o comportamento da estação:",
                placeholder="Ex: Por que a estação VGR_1 ficou parada? ou: HBW_1 demorou muito calibrando o motor",
                lines=3,
            )
            with gr.Row():
                top_k_slider = gr.Slider(
                    label="Número de episódios similares (top_k)",
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                )
                model_dropdown = gr.Dropdown(
                    choices=_MODEL_CHOICES,
                    value="qwen2.5:7b",
                    label="Modelo LLM",
                )
            analyze_btn = gr.Button("Analisar", variant="primary")

        with gr.Column():
            answer_output = gr.Textbox(label="Diagnóstico do Assistente", lines=10)
            context_output = gr.Textbox(label="Episódios Recuperados", lines=6)
            scores_output = gr.Textbox(label="Scores / Modelo / Tempo", lines=3)

    analyze_btn.click(
        fn=analyze,
        inputs=[question_input, top_k_slider, model_dropdown],
        outputs=[answer_output, context_output, scores_output],
    )

    gr.Examples(
        examples=_EXAMPLES,
        inputs=[question_input, top_k_slider, model_dropdown],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
