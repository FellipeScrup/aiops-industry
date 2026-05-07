"""Sprint 7 — Gradio UI: interface web para o RAG de diagnóstico industrial."""

import requests
import gradio as gr

API_URL = "http://localhost:8001/query"

_EXAMPLES = [
    ["alarme 139 na máquina 4", 5],
    ["downtime na máquina s_1", 5],
    ["alarme crítico parada de emergência", 5],
    ["falha sensor temperatura", 5],
]


def analyze(question: str, top_k: int) -> tuple[str, str, str]:
    """Call RAG API and format results for the three output fields."""
    try:
        resp = requests.post(
            API_URL,
            json={"question": question, "top_k": int(top_k)},
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.ConnectionError:
        err = "Erro: API indisponível. Execute `make api` antes de usar a interface."
        return err, "", ""
    except Exception as exc:
        return f"Erro inesperado: {exc}", "", ""

    answer: str = data.get("answer", "")

    context_lines: list[str] = []
    for rank, hit in enumerate(data.get("context", []), start=1):
        event = hit.get("event_type") or "N/A"
        context_lines.append(
            f"Rank {rank} (score: {hit['score']:.4f}): "
            f"Alarme {hit['alarm_code']} | Máquina {hit['machine_id']} "
            f"| {hit['source']} | Tipo {event} | {hit['severity']}"
        )
    context_str = "\n".join(context_lines)

    scores = data.get("retrieval_scores", [])
    scores_str = "  ".join(f"#{i + 1}: {s:.4f}" for i, s in enumerate(scores))
    elapsed = data.get("processing_time_s", 0)
    scores_str += f"\n\nTempo de processamento: {elapsed:.1f}s"

    return answer, context_str, scores_str


with gr.Blocks(title="AIOps Industry — Diagnóstico de Falhas Industriais") as demo:
    gr.Markdown("# 🏭 AIOps Industry — Diagnóstico de Falhas Industriais")
    gr.Markdown("**RAG local powered by Llama 3.2 + Milvus | TCC Facens 2026**")

    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(
                label="Descreva o alarme ou cole o código:",
                placeholder="Ex: alarme 139 na máquina 4, ou: downtime na máquina s_1",
                lines=3,
            )
            top_k_slider = gr.Slider(
                label="Número de logs similares (top_k)",
                minimum=1,
                maximum=10,
                value=5,
                step=1,
            )
            analyze_btn = gr.Button("Analisar", variant="primary")

        with gr.Column():
            answer_output = gr.Textbox(label="Diagnóstico do Assistente", lines=10)
            context_output = gr.Textbox(label="Logs Similares Recuperados", lines=6)
            scores_output = gr.Textbox(label="Scores de Similaridade", lines=3)

    analyze_btn.click(
        fn=analyze,
        inputs=[question_input, top_k_slider],
        outputs=[answer_output, context_output, scores_output],
    )

    gr.Examples(
        examples=_EXAMPLES,
        inputs=[question_input, top_k_slider],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
