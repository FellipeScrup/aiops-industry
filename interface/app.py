"""Sprint 7 — Gradio UI: interface web para o RAG de diagnóstico industrial."""

import requests
import gradio as gr

API_URL = "http://localhost:8001/query"

_EXAMPLES = [
    ["VGR_1 not ready during workpiece transport", 5, "llama3.2:3b"],
    ["HBW_1 calibration taking too long", 5, "llama3.2:3b"],
    ["workpiece transport from oven to milling machine", 5, "llama3.2:3b"],
    ["station not ready during pickup", 5, "llama3.2:3b"],
]

_MODEL_CHOICES = ["llama3.2:3b", "qwen2.5:3b", "gemini"]


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
        sub_part = f" | Sub-task: {sub}" if sub else ""
        context_lines.append(
            f"Rank {rank} (score: {hit['score']:.4f}): "
            f"Estação {hit.get('station', '?')} | {hit.get('event_timestamp', '?')} "
            f"| State: {hit.get('current_state', '?')} | Task: {task}{sub_part}"
        )
    context_str = "\n".join(context_lines)

    scores = data.get("retrieval_scores", [])
    scores_str = "  ".join(f"#{i + 1}: {s:.4f}" for i, s in enumerate(scores))
    elapsed = data.get("processing_time_s", 0)
    scores_str += f"\n\nModelo usado: {model_used} | Tempo: {elapsed:.1f}s"

    return answer, context_str, scores_str


with gr.Blocks(title="AIOps Industry — Análise de Telemetria Industrial") as demo:
    gr.Markdown("# AIOps Industry — Análise de Telemetria Industrial")
    gr.Markdown("**RAG local powered by Llama 3.2 / Qwen 2.5 / Gemini + Milvus | TCC Facens 2026**")

    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(
                label="Descreva o comportamento da estação:",
                placeholder="Ex: VGR_1 not ready during workpiece transport, ou: HBW_1 calibration taking too long",
                lines=3,
            )
            with gr.Row():
                top_k_slider = gr.Slider(
                    label="Número de janelas similares (top_k)",
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                )
                model_dropdown = gr.Dropdown(
                    choices=_MODEL_CHOICES,
                    value="llama3.2:3b",
                    label="Modelo LLM",
                )
            analyze_btn = gr.Button("Analisar", variant="primary")

        with gr.Column():
            answer_output = gr.Textbox(label="Diagnóstico do Assistente", lines=10)
            context_output = gr.Textbox(label="Logs Recuperados", lines=6)
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
