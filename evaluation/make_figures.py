"""Gera as figuras do TCC a partir dos resultados em evaluation/results/ (sem GPU, sem juiz).

Saída: evaluation/results/figuras_tcc/*.png (300 dpi). Convenções:
  - eixo Y sempre começa em 0 (sem truncar), ICs 95% bootstrap (10.000 reamostragens, seed 42);
  - barras sem valor (métrica não se aplica ao baseline) aparecem como "n/a", nunca como zero;
  - n impresso nos rótulos; legendas fora da área de plotagem.

Figuras:
  fig1_ragas_por_gerador   RAGAS (4 métricas) x gerador x sistema (baseline / vetorial / filtrado)
  fig2_recuperacao         Hit@5, MRR, Recall@5 (4 configurações) e Recall@5 por camada
  fig3_recusa              comportamento de recusa x evidência no contexto (estrito; estrito vs brando)
  fig4_validacao_juiz      sensibilidade do juiz a corrupções (só se perturb_scores.jsonl tiver dados)
  fig5_pipeline_rag        diagrama do pipeline RAG atual
  fig6_arquitetura_medallion  diagrama Bronze -> Silver -> Gold -> índice vetorial
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_eval as R  # noqa: E402

RES = R.BASE_RESULTS
OUT = RES / "figuras_tcc"
DPI = 300
BOOT_N, BOOT_SEED = 10_000, 42

C_BASE, C_VEC, C_FIL, C_BM25 = "#8C8C8C", "#0072B2", "#D55E00", "#009E73"  # paleta Okabe-Ito (daltônicos)
SYSTEMS = [("baseline", "LLM sem RAG", C_BASE), ("rag_vetorial", "RAG vetorial", C_VEC),
           ("rag_filtrado", "RAG com filtragem", C_FIL)]
GENERATORS = [("Qwen 2.5 7B", RES), ("Qwen 3.5 9B", RES / "qwen3.5-9b"), ("Gemma 4 e4b", RES / "gemma4-e4b")]

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
                     "savefig.bbox": "tight", "figure.dpi": 100})


# ── utilidades ──────────────────────────────────────────────────────────────

def boot_ci(values) -> tuple[float, float, float] | None:
    arr = np.array([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if arr.size == 0:
        return None
    rng = np.random.default_rng(BOOT_SEED)
    means = rng.choice(arr, size=(BOOT_N, arr.size), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)
    print(f"  ok  {OUT / name}")


def bars(ax, groups: list[str], series: list[tuple[str, str, list]], width: float = 0.8, annotate=True):
    """Barras agrupadas com IC. series = [(rótulo, cor, [valor-ou-None por grupo])], valor = (m, lo, hi)."""
    k = len(series)
    w = width / k
    for j, (_label, color, vals) in enumerate(series):
        for i, v in enumerate(vals):
            x = i - width / 2 + w * (j + 0.5)
            if v is None:
                ax.text(x, 0.02, "n/a", ha="center", va="bottom", fontsize=8, color="#666")
                continue
            m, lo, hi = v
            ax.bar(x, m, w * 0.92, color=color, zorder=2)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color="black", capsize=3, lw=1.1, zorder=3)
            if annotate:
                ax.text(x, hi + 0.015, f"{m:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.12)


def legend_below(fig, handles, ncol: int, y: float = -0.01) -> None:
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, y))


# ── fig1: RAGAS por gerador ─────────────────────────────────────────────────

def fig1() -> None:
    metrics = [("faithfulness", "Faithfulness (fidelidade ao contexto)"),
               ("factual_correctness", "Factual Correctness"),
               ("answer_relevancy", "Answer Relevancy"),
               ("context_recall", "Context Recall")]
    data = {}
    for name, d in GENERATORS:
        pq = load(d / "per_question.json")
        if pq:
            data[name] = pq
    if not data:
        print("  (sem per_question.json: fig1 ignorada)")
        return
    gens = list(data)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))
    for ax, (key, title) in zip(axes.ravel(), metrics):
        series = []
        for sys_key, label, color in SYSTEMS:
            vals = []
            for g in gens:
                v = [r.get(key) for r in data[g] if r["system"] == sys_key]
                vals.append(boot_ci(v))
            series.append((label, color, vals))
        bars(ax, gens, series)
        ax.set_title(title)
        ax.set_ylabel("média (IC 95%)")
    fig.suptitle("Avaliação RAGAS por modelo gerador e sistema (24 perguntas; juiz phi4)", fontsize=13, y=0.995)
    fig.text(0.5, 0.045, "n/a = a métrica exige contexto recuperado e não se aplica ao LLM sem RAG.", ha="center",
             fontsize=9, color="#555")
    legend_below(fig, [mpatches.Patch(color=c, label=l) for _, l, c in SYSTEMS], 3, y=-0.005)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    save(fig, "fig1_ragas_por_gerador.png")


# ── fig2: recuperação ───────────────────────────────────────────────────────

def fig2() -> None:
    ab = load(RES / "retrieval_ablation.json")
    if not ab:
        print("  (sem retrieval_ablation.json: rode 'make eval-retrieval'; fig2 ignorada)")
        return
    cfgs = [("vetorial", "Vetorial", C_VEC), ("filtrado", "Filtrada (estação + janela)", C_FIL),
            ("filtrado+BM25", "Filtrada + BM25 (RRF)", C_BM25)]
    cfgs = [c for c in cfgs if c[0] in ab]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.15]})
    cols = [("hit_at_k", "Hit@5"), ("mrr", "MRR"), ("recall_at_k", "Recall@5")]
    series = [(label, color, [boot_ci([r[c] for r in ab[name].values()]) for c, _ in cols]) for name, label, color in cfgs]
    bars(a1, [t for _, t in cols], series)
    n_all = len(next(iter(ab.values())))
    a1.set_title(f"(a) Recuperação geral (n = {n_all} perguntas)")
    a1.set_ylabel("média (IC 95%)")
    tiers = [t for t in R.TIERS]
    groups = []
    for t in tiers:
        n_t = sum(1 for r in ab[cfgs[0][0]].values() if r["tier"] == t)
        groups.append(f"{R.TIER_LABEL[t]}\n(n = {n_t})")
    series = [(label, color, [boot_ci([r["recall_at_k"] for r in ab[name].values() if r["tier"] == t]) for t in tiers])
              for name, label, color in cfgs]
    bars(a2, groups, series)
    a2.set_title("(b) Recall@5 por camada de perguntas")
    legend_below(fig, [mpatches.Patch(color=c, label=l) for _, l, c in cfgs], len(cfgs), y=-0.04)
    fig.tight_layout()
    save(fig, "fig2_recuperacao.png")


# ── fig3: recusa x evidência ────────────────────────────────────────────────

CATS = [("resp_ev", "Responde (há evidência no contexto)", "#0072B2"),
        ("ressalva", "Responde com ressalva parcial (há evidência)", "#56B4E9"),
        ("recusa_ok", "Recusa correta (sem evidência no contexto)", "#009E73"),
        ("resp_sem_ev", "Responde sem evidência no contexto (risco)", "#E69F00"),
        ("indevida", "Recusa indevida (há evidência no contexto)", "#D55E00")]


def _classify(gens: dict[str, dict], golden: dict, ids: list[str]) -> dict[str, int]:
    out = dict.fromkeys((c for c, _, _ in CATS), 0)
    for qid in ids:
        g, gold = gens[qid], golden[qid]
        hit = R.retrieval_metrics(g["retrieved_ids"], gold["evidence_episode_ids"], gold["evidence_mode"])["hit_at_k"] > 0
        kind = R.classify_answer(g["answer"], R.key_fact_recall(g["answer"], g["reference"], g["question"]))
        if hit:
            out["indevida" if kind == "recusa_total" else "ressalva" if kind == "ressalva_parcial" else "resp_ev"] += 1
        else:
            out["recusa_ok" if kind == "recusa_total" else "resp_sem_ev"] += 1
    return out


def _stack(ax, rows: list[tuple[str, dict[str, int]]]) -> None:
    for i, (_label, cnt) in enumerate(rows):
        n = sum(cnt.values())
        left = 0.0
        for key, _l, color in CATS:
            frac = cnt[key] / n if n else 0
            if frac:
                ax.barh(i, frac, left=left, color=color, edgecolor="white", height=0.62)
                ax.text(left + frac / 2, i, f"{cnt[key]}", ha="center", va="center", color="white", fontsize=10,
                        fontweight="bold")
            left += frac
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{lbl}\n(n = {sum(c.values())})" for lbl, c in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("fração das perguntas (rótulo = nº de perguntas)")
    ax.grid(axis="y", visible=False)


def fig3() -> None:
    golden = {g["id"]: g for g in R._load_golden()}
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    for ax, system in zip(axes[:2], ("rag_filtrado", "rag_vetorial")):
        rows = []
        for name, d in GENERATORS:
            gens = {g["id"]: g for g in R._read_jsonl(d / "generations.jsonl") if g["system"] == system}
            if gens:
                rows.append((name, _classify(gens, golden, sorted(gens))))
        _stack(ax, rows)
        ax.set_title(f"({'a' if system == 'rag_filtrado' else 'b'}) {dict((s, l) for s, l, _ in SYSTEMS)[system]}"
                     " — prompt estrito")
    rows = []
    for name, d_strict, d_soft in (("Qwen 3.5 9B", RES / "qwen3.5-9b", RES / "qwen3.5-9b__brando"),
                                   ("Gemma 4 e4b", RES / "gemma4-e4b", RES / "gemma4-e4b__brando")):
        soft = {g["id"]: g for g in R._read_jsonl(d_soft / "generations.jsonl") if g["system"] == "rag_filtrado"}
        strict = {g["id"]: g for g in R._read_jsonl(d_strict / "generations.jsonl") if g["system"] == "rag_filtrado"}
        ids = sorted(set(soft) & set(strict))
        if ids:
            rows.append((f"{name}\nestrito", _classify(strict, golden, ids)))
            rows.append((f"{name}\nbrando", _classify(soft, golden, ids)))
    if rows:
        _stack(axes[2], rows)
        axes[2].set_title("(c) Estrito vs brando — RAG filtrada\n(mesmas perguntas da amostra)")
    else:
        axes[2].axis("off")
    legend_below(fig, [mpatches.Patch(color=c, label=l) for _, l, c in CATS], 3, y=-0.16)
    fig.tight_layout()
    save(fig, "fig3_recusa.png")


# ── fig4: validação do juiz (perturbações) ──────────────────────────────────

def fig4() -> None:
    rows = R._read_jsonl(RES / "validation" / "perturb_scores.jsonl")
    conds = [("neutra", "Controle neutro\n(reordenar seções)"), ("numero", "Números trocados\n(fora do contexto)"),
             ("estacao", "Estação trocada\n(fora do contexto)"), ("contexto", "Contexto de\noutra pergunta")]
    deltas = {c: [r["perturbada"] - r["original"] for r in rows
                  if r["cond"] == c and r["perturbada"] is not None and r["original"] is not None] for c, _ in conds}
    if sum(len(v) >= 5 for v in deltas.values()) < 3:
        print("  (perturb_scores.jsonl com poucos dados: rode 'make validate-perturb'; fig4 ignorada)")
        return
    refs = [r["perturbada"] for r in rows if r["cond"] == "referencia" and r["perturbada"] is not None]
    fig, ax = plt.subplots(figsize=(9, 5))
    rng = np.random.default_rng(0)
    for i, (c, label) in enumerate(conds):
        v = np.array(deltas[c])
        if not len(v):
            continue
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=22, alpha=0.55, color="#555", zorder=2)
        ci = boot_ci(v)
        ax.errorbar(i, ci[0], yerr=[[ci[0] - ci[1]], [ci[2] - ci[0]]], fmt="D", color=C_FIL if c != "neutra" else C_BASE,
                    capsize=5, ms=8, lw=2, zorder=3)
        ax.text(i, 0.93, f"n = {len(v)}", ha="center", fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([lbl for _, lbl in conds])
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Δ faithfulness (perturbada − original)")
    title = "Sensibilidade do juiz a corrupções com resposta conhecida"
    if refs:
        title += f"\nControle positivo (respostas de referência): faithfulness média {np.mean(refs):.2f} (n = {len(refs)})"
    ax.set_title(title)
    if min(len(v) for v in deltas.values() if len(v)) < 16:
        ax.text(0.99, 0.02, "RESULTADO PARCIAL (medição ainda em andamento)", transform=ax.transAxes, ha="right",
                fontsize=9, color="#B00020", fontweight="bold")
    fig.text(0.5, -0.02, "Pontos = pares; losango = média com IC 95% bootstrap. Esperado: neutro ≈ 0; demais < 0.",
             ha="center", fontsize=9, color="#555")
    fig.tight_layout()
    save(fig, "fig4_validacao_juiz.png")


# ── diagramas ───────────────────────────────────────────────────────────────

def _box(ax, cx, cy, w, h, title, sub, color, dashed=False):
    ax.add_patch(mpatches.FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                         fc=color, ec="#333", lw=1.4, ls="--" if dashed else "-", zorder=2))
    ax.text(cx, cy + 0.27, title, ha="center", va="center", fontsize=10.5, fontweight="bold", zorder=3, linespacing=1.1)
    if sub:
        ax.text(cx, cy - 0.27, sub, ha="center", va="center", fontsize=8.3, color="#222", zorder=3)


def _arrow(ax, p, q, dashed=False, text=None):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.6, ls="--" if dashed else "-"),
                zorder=1)
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.2, text, ha="center", fontsize=8, color="#444", style="italic")


def fig5() -> None:
    fig, ax = plt.subplots(figsize=(15, 5.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.1)
    ax.axis("off")
    ax.set_title("Pipeline RAG de diagnóstico de falhas industriais", fontsize=14, fontweight="bold")
    w, h = 1.78, 1.25
    top = [("Pergunta", "linguagem natural\n(técnico de manutenção)", "#D9EAD3", False),
           ("Embedding\nda consulta", "nomic-embed-text-v1.5\n768 dimensões", "#DCE9FB", False),
           ("Extração\nde filtros", "estação e janela temporal\n(só na variante filtrada)", "#FFF2CC", True),
           ("Busca\nno Milvus", "similaridade de cosseno\n(406 episódios)", "#DCE9FB", False),
           ("Reranking\nBM25 + RRF", "opcional (RAG_RERANK=bm25)\nlexical + vetorial", "#FFF2CC", True)]
    bottom = [("Deduplicação", "janela de 10 s\n(amostragem a 10 Hz)", "#FDE5CF", False),
              ("Top-5 episódios", "contexto enviado\nao modelo", "#FDE5CF", False),
              ("Prompt com\nguardrails", "estrito | brando\n+ anti-injeção", "#F4CCCC", False),
              ("LLM local\n(Ollama)", "Qwen 2.5 7B · Qwen 3.5 9B\nGemma 4 e4b", "#F4CCCC", False),
              ("Diagnóstico", "evento, causa provável,\nação (em português)", "#E4D7EE", False)]
    xs = np.linspace(1.1, 8.9, 5)
    ytop, ybot = 3.2, 0.95
    for x, (t, s, c, dashed) in zip(xs, top):
        _box(ax, x, ytop, w, h, t, s, c, dashed)
    for x, (t, s, c, dashed) in zip(xs, bottom):
        _box(ax, x, ybot, w, h, t, s, c, dashed)
    for i in range(4):
        _arrow(ax, (xs[i] + w / 2, ytop), (xs[i + 1] - w / 2, ytop))
        _arrow(ax, (xs[i] + w / 2, ybot), (xs[i + 1] - w / 2, ybot))
    # cotovelo da linha de cima para a de baixo
    ax.plot([xs[-1], xs[-1]], [ytop - h / 2, 2.07], color="#333", lw=1.6, zorder=1)
    ax.plot([xs[-1], xs[0]], [2.07, 2.07], color="#333", lw=1.6, zorder=1)
    _arrow(ax, (xs[0], 2.07), (xs[0], ybot + h / 2))
    ax.text(5, 1.84, "candidatos recuperados (top-20)", ha="center", fontsize=8.5, color="#444", style="italic")
    ax.legend(handles=[mpatches.Patch(fc="#FFF2CC", ec="#333", ls="--", label="etapa opcional / específica de variante")],
              loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.02))
    save(fig, "fig5_pipeline_rag.png")


def fig6() -> None:
    fig, ax = plt.subplots(figsize=(15, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    ax.set_title("Arquitetura Medallion: do log bruto ao índice vetorial", fontsize=14, fontweight="bold")
    w, h = 2.05, 1.9
    layers = [("Bronze", "MinIO (bucket bronze)\nNDJSON bruto, sem alteração\ntreino + teste (168 MB)",
               "#F5CBA7", "upload_bronze.py"),
              ("Silver", "PostgreSQL: smartfactory_logs\n408.837 eventos a 10 Hz\n(+ Parquet no MinIO)",
               "#AED6F1", "parse_silver.py"),
              ("Gold", "PostgreSQL: smartfactory_episodes\n485 episódios (log de treino)\n415 not ready · 7 anomalias",
               "#A9DFBF", "parse_episodes.py"),
              ("Índice vetorial", "Milvus: 406 episódios indexados\nembeddings de 768 dimensões\ncosseno",
               "#D7BDE2", "embed_gold.py")]
    xs = [1.35, 3.8, 6.25, 8.7]
    for x, (t, s, c, script) in zip(xs, layers):
        ax.add_patch(mpatches.FancyBboxPatch((x - w / 2, 0.9), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=c,
                                             ec="#333", lw=1.5))
        ax.text(x, 0.9 + h - 0.32, t, ha="center", fontsize=13, fontweight="bold")
        ax.text(x, 0.9 + h / 2 - 0.15, s, ha="center", va="center", fontsize=8.6)
        ax.text(x, 0.7, f"gerado por {script}", ha="center", fontsize=8.2, color="#444", style="italic")
    for i in range(3):
        _arrow(ax, (xs[i] + w / 2 + 0.03, 1.85), (xs[i + 1] - w / 2 - 0.03, 1.85))
    ax.text(5, 0.25, "Episódio = sequência de eventos com mesma estação, tarefa, sub-tarefa e estado; anomalia de duração: "
            "duração > mediana + 3·MAD e ≥ 2 s acima da mediana.", ha="center", fontsize=9, color="#444")
    save(fig, "fig6_arquitetura_medallion.png")


def main() -> None:
    print(f"Gerando figuras em {OUT}")
    for fn in (fig1, fig2, fig3, fig4, fig5, fig6):
        fn()


if __name__ == "__main__":
    main()
