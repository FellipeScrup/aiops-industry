"""Avaliação do RAG industrial: baseline x RAG vetorial x RAG com filtragem.

Etapas (cada uma retomável — usa checkpoint em evaluation/results/):

    generate  executa os 3 sistemas sobre o golden set e grava respostas, contextos,
              ids recuperados, tokens e latência          -> generations.jsonl
    ragas     calcula métricas RAGAS (juiz LLM local)      -> ragas_scores.jsonl
    report    agrega tudo, calcula IC bootstrap e gera     -> summary.json + figuras
              tabelas/figuras em evaluation/results/
    compare   tabela lado a lado de todos os geradores já avaliados + diferença pareada

Uso (a partir da raiz, com o venv de avaliação):

    PYTHONPATH=. .venv-eval/Scripts/python evaluation/run_eval.py generate --model qwen3.5:9b
    PYTHONPATH=. .venv-eval/Scripts/python evaluation/run_eval.py ragas    --model qwen3.5:9b [--judge phi4] [--limit N]
    PYTHONPATH=. .venv-eval/Scripts/python evaluation/run_eval.py report   --model qwen3.5:9b
    PYTHONPATH=. .venv-eval/Scripts/python evaluation/run_eval.py compare

Cada gerador grava em evaluation/results/<modelo>/ (o qwen2.5:7b, que já tinha resultados,
permanece na raiz de evaluation/results/).

Decisões metodológicas:
  * Geração com temperature=0, seed fixo e thinking desligado (reprodutibilidade e
    comparação justa entre modelos com e sem modo de raciocínio).
  * Métricas de recuperação (Hit@k, MRR, Recall@k) e key-fact recall são DETERMINÍSTICAS
    (comparam ids de episódios / números e estações da referência) — não dependem de juiz LLM.
    A recuperação independe do gerador: só varia entre RAG vetorial e RAG com filtragem.
  * Métricas RAGAS usam juiz LLM LOCAL de TERCEIRA família (padrão: phi4), diferente dos
    geradores avaliados (Qwen, Gemma), para evitar viés de autopreferência. Um aviso é
    emitido se juiz e gerador forem da mesma família. Notas RAGAS só são comparáveis entre
    geradores avaliados pelo MESMO juiz. Taxa de NaN do juiz é reportada em summary.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("LLM_TEMPERATURE", "0")
os.environ.setdefault("LLM_SEED", "42")
os.environ.setdefault("LLM_THINK", "0")  # sem raciocínio: comparação justa entre geradores

import numpy as np  # noqa: E402

logger = logging.getLogger("evaluation")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

ROOT = Path(__file__).resolve().parent
BASE_RESULTS = ROOT / "results"
GOLDEN = ROOT / "golden_set.json"

# Resultados de cada gerador ficam em results/<modelo>/. O gerador "legado" (qwen2.5:7b),
# que já tinha resultados antes desta separação, continua na raiz de results/ — assim nada
# que já foi gerado (nem as figuras já usadas no texto do TCC) precisa ser movido.
LEGACY_MODEL = "qwen2.5:7b"
GEN_MODEL = LEGACY_MODEL
# Variante da regra de recusa do prompt do RAG (ver rag/generator.py). "estrito" = padrão do
# projeto, usado em todos os resultados já avaliados; as demais vão para results/<modelo>__<prompt>/.
PROMPTS = ("estrito", "brando")
DEFAULT_PROMPT = "estrito"
PROMPT = DEFAULT_PROMPT
# Reranking lexical (BM25 + RRF) sobre os candidatos do retriever (ver rag/lexical.py); "off" = padrão.
RERANKS = ("off", "bm25")
DEFAULT_RERANK = "off"
RERANK = DEFAULT_RERANK
RESULTS = BASE_RESULTS
GENERATIONS = RESULTS / "generations.jsonl"
RAGAS_SCORES = RESULTS / "ragas_scores.jsonl"

TOP_K = 5
SYSTEMS = ("baseline", "rag_vetorial", "rag_filtrado")
SYSTEM_LABEL = {
    "baseline": "LLM sem RAG",
    "rag_vetorial": "RAG vetorial",
    "rag_filtrado": "RAG com filtragem",
}
TIERS = ("factual", "cross_station", "causal")
TIER_LABEL = {"factual": "Factual", "cross_station": "Multi-estação", "causal": "Causal"}

# Juiz de TERCEIRA família (nem Qwen, nem Gemma): evita viés de autopreferência quando o
# juiz avalia texto escrito por um modelo da própria família.
DEFAULT_JUDGE = "phi4"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", DEFAULT_JUDGE)
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42

_BASELINE_PROMPT = """\
Você é um assistente especializado em manutenção de fábricas inteligentes com automação industrial.
Responda sempre em português, usando os três tópicos abaixo.

PERGUNTA DO TÉCNICO:
{query}

Resposta:
1. Diagnóstico: o que a situação indica sobre o comportamento do sistema
2. Causa provável do problema ou comportamento observado
3. Ação corretiva recomendada"""


# ── utilidades ──────────────────────────────────────────────────────────────

def _slug(model: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", model.lower()).strip("-")


def _family(model: str) -> str:
    """Família do modelo pelo prefixo alfabético do nome ("qwen2.5:7b" e "qwen3.5:9b" -> "qwen")."""
    m = re.match(r"[a-z]+", model.lower().split("/")[-1])
    return m.group(0) if m else model.lower()


def _configure(model: str, judge: str, prompt: str = DEFAULT_PROMPT, rerank: str = DEFAULT_RERANK) -> None:
    """Define gerador, juiz, prompt, reranking e as pastas de saída (uma por gerador+prompt+rerank)."""
    global GEN_MODEL, JUDGE_MODEL, PROMPT, RERANK, RESULTS, GENERATIONS, RAGAS_SCORES
    GEN_MODEL, JUDGE_MODEL, PROMPT, RERANK = model, judge, prompt, rerank
    os.environ["LLM_GUARDRAIL"] = prompt  # lido por rag.generator._guardrail()
    os.environ["RAG_RERANK"] = "" if rerank == DEFAULT_RERANK else rerank  # lido por rag.retriever
    if model == LEGACY_MODEL and prompt == DEFAULT_PROMPT and rerank == DEFAULT_RERANK:
        RESULTS = BASE_RESULTS
    else:
        suffix = ("" if prompt == DEFAULT_PROMPT else f"__{prompt}") + \
                 ("" if rerank == DEFAULT_RERANK else f"__{rerank}")
        RESULTS = BASE_RESULTS / f"{_slug(model)}{suffix}"
    GENERATIONS = RESULTS / "generations.jsonl"
    RAGAS_SCORES = RESULTS / "ragas_scores.jsonl"
    if _family(model) == _family(judge):
        logger.warning("Juiz '%s' é da MESMA família do gerador '%s': risco de viés de "
                       "autopreferência. Prefira um juiz de outra família.", judge, model)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_jsonl(path: Path, row: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_golden() -> list[dict]:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def _context_text(hit: dict) -> str:
    """Texto de um episódio recuperado, como visto pelo gerador (usado como retrieved_contexts)."""
    dur = hit.get("duration_s")
    parts = [f"Estação {hit['station']} | início: {hit['event_timestamp']}"]
    if dur is not None:
        parts.append(f"duração: {dur:.1f}s")
    if hit.get("is_anomaly"):
        parts.append("[ANOMALIA DE DURAÇÃO]")
    parts.append(f"Task: {hit.get('current_task') or 'idle'}")
    if hit.get("current_sub_task"):
        parts.append(f"Sub-task: {hit['current_sub_task']}")
    if hit.get("log_text"):
        parts.append(f"Episódio: {hit['log_text']}")
    return " | ".join(parts)


# ── etapa 1: geração ────────────────────────────────────────────────────────

def stage_generate(limit: int | None, sample: int | None = None) -> None:
    from rag.generator import _generate_ollama  # noqa: PLC0415
    from rag.pipeline import query as rag_query  # noqa: PLC0415

    golden = _load_golden()
    if sample:  # N perguntas espaçadas pelo golden set (cobre os 3 tiers, que estão em blocos)
        golden = golden[::max(1, len(golden) // sample)][:sample]
    elif limit:
        golden = golden[:limit]
    done = {(r["id"], r["system"]) for r in _read_jsonl(GENERATIONS)}
    logger.info("Gerando %d perguntas x %d sistemas (%d já feitos) com %s -> %s",
                len(golden), len(SYSTEMS), len(done), GEN_MODEL, RESULTS)

    for item in golden:
        for system in SYSTEMS:
            if (item["id"], system) in done:
                continue
            t0 = time.perf_counter()
            if system == "baseline":
                answer, usage = _generate_ollama(_BASELINE_PROMPT.format(query=item["question"]), GEN_MODEL)
                hits: list[dict] = []
            else:
                out = rag_query(item["question"], top_k=TOP_K, model=GEN_MODEL,
                                use_hybrid=(system == "rag_filtrado"))
                answer, usage, hits = out["answer"], out["token_usage"], out["context"]
            latency = time.perf_counter() - t0

            _append_jsonl(GENERATIONS, {
                "model": GEN_MODEL, "prompt": PROMPT, "rerank": RERANK,
                "id": item["id"], "tier": item["tier"], "subtype": item["subtype"], "system": system,
                "question": item["question"], "reference": item["reference"], "answer": answer,
                "retrieved_ids": [h["event_id"] for h in hits],
                "retrieved_contexts": [_context_text(h) for h in hits],
                "scores": [h["score"] for h in hits],
                "usage": usage, "latency_s": round(latency, 2),
            })
            logger.info("[%s | %s] %.1fs, %d tokens", item["id"], system, latency, usage.get("total_tokens", 0))


# ── métricas determinísticas ────────────────────────────────────────────────

def retrieval_metrics(retrieved: list[str], evidence: list[str], mode: str) -> dict[str, float]:
    """Hit@k, MRR e Recall@k sobre ids de episódios-evidência."""
    ev = set(evidence)
    hit_ranks = [i for i, rid in enumerate(retrieved, start=1) if rid in ev]
    found = len({rid for rid in retrieved if rid in ev})
    recall = (1.0 if found else 0.0) if mode == "any" else found / len(ev)
    return {
        "hit_at_k": 1.0 if hit_ranks else 0.0,
        "mrr": 1.0 / hit_ranks[0] if hit_ranks else 0.0,
        "recall_at_k": recall,
    }


_NUM_RE = re.compile(r"\d+\.\d")
_STATION_RE = re.compile(r"\b[A-Z]{2,3}_\d\b")


def key_fact_recall(answer: str, reference: str, question: str = "") -> float:
    """Fração dos fatos-chave da referência (durações, estações, atividade BPM) presentes na resposta.

    Fatos que já aparecem na própria pergunta são descartados: repeti-los não prova conhecimento
    (um LLM sem RAG repete a estação citada na pergunta).
    """
    facts = set(_NUM_RE.findall(reference)) | set(_STATION_RE.findall(reference))
    if reference.startswith("A atividade interrompida"):
        facts |= {q.lower() for q in re.findall(r"'([^']+)'", reference)}
    q_low = question.lower().replace(",", ".")
    facts = {f for f in facts if f.lower() not in q_low}
    if not facts:
        return float("nan")
    ans = answer.replace(",", ".").lower()
    hit = 0
    for f in facts:
        pat = rf"(?<![\d.]){re.escape(f.lower())}(?!\d)" if _NUM_RE.fullmatch(f) else re.escape(f.lower())
        hit += bool(re.search(pat, ans))
    return hit / len(facts)


_NEG_RE = re.compile(
    r"dentro d[oa] (?:normal|normalidade|esperado|faixa|padr|tempo t[ií]pico|dura[cç][aã]o normal)"
    r"|n[aã]o (?:foi|[eé]|est[aá]|houve|h[aá]|indica|apresenta|ocorreu)[^.]{0,40}(?:anormal|an[oô]mal|longa|longo|evid[eê]ncia)"
    r"|n[aã]o (?:foi|[eé]) anormalmente|sem anomalia|n[aã]o h[aá] anomalia"
)
_POS_RE = re.compile(r"anormalmente long|an[oô]mal|anomalia|acima d[ao] (?:m[eé]dia|mediana|normal|limiar)|excede")


def anomaly_verdict(answer: str) -> str | None:
    """Veredito sim/não extraído da resposta (heurística — auditada manualmente no relatório)."""
    low = answer.lower()
    if _NEG_RE.search(low):
        return "nao"
    if _POS_RE.search(low):
        return "sim"
    return None


# ── etapa 2: RAGAS ──────────────────────────────────────────────────────────

def _build_judge():
    from langchain_ollama import ChatOllama  # noqa: PLC0415
    from ragas.embeddings import BaseRagasEmbeddings  # noqa: PLC0415
    from ragas.llms import LangchainLLMWrapper  # noqa: PLC0415

    class FastEmbedRagas(BaseRagasEmbeddings):
        """Embeddings locais (nomic via fastembed) — os mesmos do índice do projeto."""

        def embed_query(self, text: str) -> list[float]:
            from rag.retriever import _embed_query  # noqa: PLC0415
            return _embed_query(text)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_query(t) for t in texts]

        async def aembed_query(self, text: str) -> list[float]:
            return self.embed_query(text)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.embed_documents(texts)

    llm = LangchainLLMWrapper(ChatOllama(
        model=JUDGE_MODEL, temperature=0, seed=42, num_ctx=8192,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    ))
    return llm, FastEmbedRagas()


def stage_ragas(limit: int | None) -> None:
    from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: PLC0415
    from ragas.metrics import (  # noqa: PLC0415
        Faithfulness, FactualCorrectness, LLMContextPrecisionWithReference,
        LLMContextRecall, ResponseRelevancy,
    )
    from ragas.run_config import RunConfig  # noqa: PLC0415

    gens = _read_jsonl(GENERATIONS)
    if limit:
        keep = {g["id"] for g in _load_golden()[:limit]}
        gens = [g for g in gens if g["id"] in keep]
    done = {(r["id"], r["system"]) for r in _read_jsonl(RAGAS_SCORES) if r.get("judge") == JUDGE_MODEL}
    llm, emb = _build_judge()
    run_cfg = RunConfig(timeout=600, max_workers=1, max_retries=2)

    metrics_rag = [Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(),
                   LLMContextRecall(), FactualCorrectness()]
    metrics_base = [ResponseRelevancy(), FactualCorrectness()]  # sem contexto, só o que faz sentido

    todo = [g for g in gens if (g["id"], g["system"]) not in done]
    logger.info("RAGAS: %d amostras a avaliar (%d já feitas) | juiz=%s", len(todo), len(done), JUDGE_MODEL)

    for n, g in enumerate(todo, start=1):
        t0 = time.perf_counter()
        sample = SingleTurnSample(
            user_input=g["question"], response=g["answer"],
            retrieved_contexts=g["retrieved_contexts"], reference=g["reference"],
        )
        metrics = metrics_base if g["system"] == "baseline" else metrics_rag
        try:
            res = evaluate(EvaluationDataset([sample]), metrics=metrics, llm=llm, embeddings=emb,
                           run_config=run_cfg, show_progress=False, raise_exceptions=False)
            row = res.to_pandas().iloc[0].to_dict()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falha em %s/%s: %s", g["id"], g["system"], exc)
            row = {}
        # ragas 0.2.10 nomeia a coluna "factual_correctness"; versões novas, "factual_correctness(mode=f1)".
        row = {("factual_correctness" if k.startswith("factual_correctness") else k): v
               for k, v in row.items()}
        scores = {k: (None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v))
                  for k, v in row.items() if k in {
                      "faithfulness", "answer_relevancy", "llm_context_precision_with_reference",
                      "context_recall", "factual_correctness"}}
        _append_jsonl(RAGAS_SCORES, {"judge": JUDGE_MODEL, "id": g["id"], "system": g["system"],
                                     "tier": g["tier"], **scores})
        logger.info("[%d/%d] %s/%s em %.0fs -> %s", n, len(todo), g["id"], g["system"],
                    time.perf_counter() - t0, {k: (round(v, 2) if v is not None else None) for k, v in scores.items()})


# ── etapa 3: relatório ──────────────────────────────────────────────────────

RAGAS_COLS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "llm_context_precision_with_reference": "Context Precision",
    "context_recall": "Context Recall",
    "factual_correctness": "Factual Correctness",
}


def _boot_ci(values: list[float], rng: np.random.Generator) -> tuple[float, float, float]:
    arr = np.array([v for v in values if v is not None and not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    means = rng.choice(arr, size=(BOOTSTRAP_N, arr.size), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _paired_diff(a: dict[str, float], b: dict[str, float], rng: np.random.Generator) -> dict:
    """Diferença média pareada (a - b) por pergunta, com IC bootstrap e p aproximado."""
    keys = [k for k in a if k in b and a[k] is not None and b[k] is not None
            and not math.isnan(a[k]) and not math.isnan(b[k])]
    if not keys:
        return {"n": 0}
    d = np.array([a[k] - b[k] for k in keys])
    means = rng.choice(d, size=(BOOTSTRAP_N, d.size), replace=True).mean(axis=1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {"n": int(d.size), "diff": float(d.mean()),
            "ci_low": float(np.percentile(means, 2.5)), "ci_high": float(np.percentile(means, 97.5)),
            "p_approx": float(min(1.0, p))}


def stage_report() -> None:
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    golden = {g["id"]: g for g in _load_golden()}
    gens = _read_jsonl(GENERATIONS)
    ragas = {(r["id"], r["system"]): r for r in _read_jsonl(RAGAS_SCORES) if r.get("judge") == JUDGE_MODEL}
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    # tabela longa: uma linha por (pergunta, sistema)
    rows: list[dict] = []
    for g in gens:
        evidence = golden[g["id"]]
        row = {"id": g["id"], "tier": g["tier"], "subtype": g["subtype"], "system": g["system"],
               "key_fact_recall": key_fact_recall(g["answer"], g["reference"], g["question"]),
               "latency_s": g["latency_s"], "total_tokens": g["usage"].get("total_tokens", 0),
               "prompt_tokens": g["usage"].get("prompt_tokens", 0)}
        if g["subtype"].startswith("anomalia_"):
            expected = "sim" if g["subtype"] == "anomalia_sim" else "nao"
            verdict = anomaly_verdict(g["answer"])
            row["verdict"] = verdict
            row["verdict_correct"] = 1.0 if verdict == expected else 0.0
        if g["system"] != "baseline":
            row.update(retrieval_metrics(g["retrieved_ids"], evidence["evidence_episode_ids"],
                                         evidence["evidence_mode"]))
        row.update({c: (ragas.get((g["id"], g["system"]), {}) or {}).get(c) for c in RAGAS_COLS})
        rows.append(row)

    metric_cols = ["hit_at_k", "mrr", "recall_at_k", "key_fact_recall", "verdict_correct", *RAGAS_COLS]
    summary: dict = {"n_questions": len(golden), "top_k": TOP_K, "generator": GEN_MODEL, "prompt": PROMPT, "rerank": RERANK,
                     "judge": JUDGE_MODEL, "bootstrap": {"n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED},
                     "overall": {}, "by_tier": {}, "paired": {}, "cost": {}, "ragas_nan": {}}

    def vals(system: str, col: str, tier: str | None = None) -> dict[str, float]:
        return {r["id"]: r.get(col) for r in rows
                if r["system"] == system and (tier is None or r["tier"] == tier) and r.get(col) is not None}

    for system in SYSTEMS:
        summary["overall"][system] = {}
        for col in metric_cols:
            m, lo, hi = _boot_ci(list(vals(system, col).values()), rng)
            if not math.isnan(m):
                summary["overall"][system][col] = {"mean": m, "ci_low": lo, "ci_high": hi,
                                                   "n": len(vals(system, col))}
        summary["by_tier"][system] = {}
        for tier in TIERS:
            summary["by_tier"][system][tier] = {}
            for col in metric_cols:
                m, lo, hi = _boot_ci(list(vals(system, col, tier).values()), rng)
                if not math.isnan(m):
                    summary["by_tier"][system][tier][col] = {"mean": m, "ci_low": lo, "ci_high": hi,
                                                             "n": len(vals(system, col, tier))}
        sys_rows = [r for r in rows if r["system"] == system]
        summary["cost"][system] = {
            "latency_s_mean": float(np.mean([r["latency_s"] for r in sys_rows])),
            "prompt_tokens_mean": float(np.mean([r["prompt_tokens"] for r in sys_rows])),
            "total_tokens_mean": float(np.mean([r["total_tokens"] for r in sys_rows])),
        }
        expected = [c for c in RAGAS_COLS if system != "baseline" or c in ("answer_relevancy", "factual_correctness")]
        summary["ragas_nan"][system] = {c: sum(1 for r in sys_rows if r.get(c) is None) for c in expected}

    # comparações pareadas (por pergunta)
    for a, b in (("rag_vetorial", "baseline"), ("rag_filtrado", "baseline"), ("rag_filtrado", "rag_vetorial")):
        summary["paired"][f"{a}_vs_{b}"] = {}
        for col in ("key_fact_recall", "factual_correctness", "answer_relevancy"):
            summary["paired"][f"{a}_vs_{b}"][col] = _paired_diff(vals(a, col), vals(b, col), rng)
        if b != "baseline":
            for col in ("hit_at_k", "mrr", "recall_at_k", "faithfulness", "context_recall",
                        "llm_context_precision_with_reference"):
                summary["paired"][f"{a}_vs_{b}"][col] = _paired_diff(vals(a, col), vals(b, col), rng)
        summary["paired"][f"{a}_vs_{b}"]["by_tier_key_fact_recall"] = {
            t: _paired_diff(vals(a, "key_fact_recall", t), vals(b, "key_fact_recall", t), rng) for t in TIERS}

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "per_question.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    _make_figures(summary, plt)
    _print_tables(summary)


COLORS = {"baseline": "#9aa0a6", "rag_vetorial": "#1f77b4", "rag_filtrado": "#d95f02"}


def _bar_group(ax, summary: dict, metrics: list[tuple[str, str]], systems: list[str], source: dict) -> None:
    width = 0.8 / len(systems)
    x = np.arange(len(metrics))
    for i, system in enumerate(systems):
        means, errs = [], [[], []]
        for col, _ in metrics:
            d = source[system].get(col)
            if d is None:
                means.append(0.0); errs[0].append(0); errs[1].append(0)
            else:
                means.append(d["mean"]); errs[0].append(d["mean"] - d["ci_low"]); errs[1].append(d["ci_high"] - d["mean"])
        ax.bar(x + i * width - 0.4 + width / 2, means, width, yerr=errs, capsize=3,
               label=SYSTEM_LABEL[system], color=COLORS[system])
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)


def _make_figures(summary: dict, plt) -> None:
    fig_dir = RESULTS / "figuras"
    fig_dir.mkdir(exist_ok=True)

    # Fig A — recuperação (determinística) por sistema
    fig, ax = plt.subplots(figsize=(7, 4))
    _bar_group(ax, summary, [("hit_at_k", f"Hit@{TOP_K}"), ("mrr", "MRR"), ("recall_at_k", f"Recall@{TOP_K}")],
               ["rag_vetorial", "rag_filtrado"], summary["overall"])
    ax.set_ylabel("Média (IC 95% bootstrap)")
    ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "fig_recuperacao.png", dpi=200); plt.close(fig)

    # Fig B — RAGAS por sistema
    fig, ax = plt.subplots(figsize=(10, 4.2))
    cols = [(c, l) for c, l in RAGAS_COLS.items()]
    _bar_group(ax, summary, cols, list(SYSTEMS), summary["overall"])
    ax.set_ylabel("Média (IC 95% bootstrap)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "fig_ragas.png", dpi=200); plt.close(fig)

    # Fig C — key-fact recall por camada de dificuldade
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / len(SYSTEMS)
    x = np.arange(len(TIERS))
    for i, system in enumerate(SYSTEMS):
        ds = [summary["by_tier"][system][t].get("key_fact_recall") for t in TIERS]
        means = [d["mean"] if d else 0 for d in ds]
        errs = [[(d["mean"] - d["ci_low"]) if d else 0 for d in ds], [(d["ci_high"] - d["mean"]) if d else 0 for d in ds]]
        ax.bar(x + i * width - 0.4 + width / 2, means, width, yerr=errs, capsize=3,
               label=SYSTEM_LABEL[system], color=COLORS[system])
    ax.set_xticks(x); ax.set_xticklabels([TIER_LABEL[t] for t in TIERS])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Recall de fatos-chave (IC 95%)"); ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "fig_fatos_por_camada.png", dpi=200); plt.close(fig)

    # Fig D — recuperação por camada
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / 2
    for i, system in enumerate(("rag_vetorial", "rag_filtrado")):
        ds = [summary["by_tier"][system][t].get("recall_at_k") for t in TIERS]
        means = [d["mean"] if d else 0 for d in ds]
        errs = [[(d["mean"] - d["ci_low"]) if d else 0 for d in ds], [(d["ci_high"] - d["mean"]) if d else 0 for d in ds]]
        ax.bar(x + i * width - 0.4 + width / 2, means, width, yerr=errs, capsize=3,
               label=SYSTEM_LABEL[system], color=COLORS[system])
    ax.set_xticks(x); ax.set_xticklabels([TIER_LABEL[t] for t in TIERS])
    ax.set_ylim(0, 1.05); ax.set_ylabel(f"Recall@{TOP_K} (IC 95%)"); ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "fig_recall_por_camada.png", dpi=200); plt.close(fig)
    logger.info("Figuras em %s", fig_dir)


def _fmt(d: dict | None) -> str:
    return "—" if not d else f"{d['mean']:.2f} [{d['ci_low']:.2f}; {d['ci_high']:.2f}]"


def _print_tables(summary: dict) -> None:
    cols = ["hit_at_k", "mrr", "recall_at_k", "key_fact_recall", *RAGAS_COLS]
    print("\n== GERAL (média [IC95%]) ==")
    for s in SYSTEMS:
        print(f"{SYSTEM_LABEL[s]:20}", " | ".join(f"{c[:14]}={_fmt(summary['overall'][s].get(c))}" for c in cols))
    print("\n== POR CAMADA (key_fact_recall / recall@k) ==")
    for s in SYSTEMS:
        for t in TIERS:
            d = summary["by_tier"][s][t]
            print(f"{SYSTEM_LABEL[s]:20} {t:14} kf={_fmt(d.get('key_fact_recall'))} rec={_fmt(d.get('recall_at_k'))}")
    print("\n== PAREADOS ==")
    for k, v in summary["paired"].items():
        for col, r in v.items():
            if col.startswith("by_tier"):
                continue
            if r.get("n"):
                print(f"{k:28} {col:38} Δ={r['diff']:+.3f} [{r['ci_low']:+.3f}; {r['ci_high']:+.3f}] p≈{r['p_approx']:.3f} n={r['n']}")
    print("\n== CUSTO ==", json.dumps(summary["cost"], indent=1))
    print("== NaN RAGAS ==", json.dumps(summary["ragas_nan"]))


COMPARE_COLS = [
    ("key_fact_recall", "FatosChave"),
    ("verdict_correct", "Veredito"),
    ("faithfulness", "Faithful."),
    ("factual_correctness", "FactualCorr"),
]


def stage_compare() -> None:
    """Compara geradores lado a lado a partir dos summary.json / per_question.json de cada pasta."""
    dirs = [BASE_RESULTS] + sorted(p for p in BASE_RESULTS.iterdir() if p.is_dir() and p.name != "figuras")
    runs: list[tuple[str, dict, list[dict]]] = []
    for d in dirs:
        sf, pf = d / "summary.json", d / "per_question.json"
        if sf.exists() and pf.exists():
            s = json.loads(sf.read_text(encoding="utf-8"))
            label = s.get("generator", d.name)
            if s.get("prompt", DEFAULT_PROMPT) != DEFAULT_PROMPT:
                label += f"[{s['prompt']}]"
            if s.get("rerank", DEFAULT_RERANK) != DEFAULT_RERANK:
                label += f"+{s['rerank']}"
            runs.append((label, s, json.loads(pf.read_text(encoding="utf-8"))))
    if not runs:
        print("Nenhum resultado agregado. Rode 'report' para cada gerador antes de comparar.")
        return

    for system in ("rag_filtrado", "rag_vetorial"):
        print(f"\n== {SYSTEM_LABEL[system]} (média [IC95%]; juiz diferente => notas RAGAS não comparáveis) ==")
        print(f"{'gerador':20} {'juiz':8} " + " ".join(f"{lbl:>20}" for _, lbl in COMPARE_COLS)
              + f" {'lat(s)':>7} {'tokens':>7}")
        for name, s, _ in runs:
            o, c = s["overall"][system], s["cost"][system]
            print(f"{name:20} {s.get('judge', '?'):8} "
                  + " ".join(f"{_fmt(o.get(col)):>20}" for col, _ in COMPARE_COLS)
                  + f" {c['latency_s_mean']:7.1f} {c['total_tokens_mean']:7.0f}")

    # Diferença pareada por pergunta (mesmas 24 perguntas) em relação ao primeiro gerador
    ref_name, _, ref_rows = runs[0]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    print(f"\n== Diferença pareada vs {ref_name} (Δ, IC95% bootstrap; n = nº de perguntas) ==")
    for name, _, rows in runs[1:]:
        for system in ("rag_filtrado", "rag_vetorial"):
            for col in ("key_fact_recall", "verdict_correct"):
                a = {r["id"]: r.get(col) for r in rows if r["system"] == system}
                b = {r["id"]: r.get(col) for r in ref_rows if r["system"] == system}
                r_ = _paired_diff(a, b, rng)
                if r_.get("n"):
                    print(f"{name:20} {system:13} {col:16} Δ={r_['diff']:+.3f} "
                          f"[{r_['ci_low']:+.3f}; {r_['ci_high']:+.3f}] n={r_['n']}")
    print("\nObs.: com 24 perguntas, IC que cruza 0 => diferença não distinguível de ruído.")


# Declínio de resposta, em qualquer redação ("Não há informação suficiente...", "Não há episódios que
# mencionem...", "Não é possível determinar...", "não foi registrado"). Detectar só a frase-padrão do
# prompt estrito subestima as recusas do prompt brando (que explica o que falta com outras palavras).
_DECLINE_RE = re.compile(
    r"n[ãa]o h[áa]\b[^.\n]{0,80}?(?:informa|epis[óo]d|registr|dado|evid[êe]nc)"
    r"|n[ãa]o (?:[ée] poss[íi]vel|foi poss[íi]vel|existe[m]?|foi registrad|consta)"
    r"|(?:sem|nenhum) (?:epis[óo]dio|registro)",
    re.IGNORECASE,
)
_DECIMAL_RE = re.compile(r"(?<![\d.])\d+\.\d+(?!\d)")


def classify_answer(answer: str, key_fact: float) -> str:
    """'recusa_total' | 'ressalva_parcial' | 'resposta' — independente da redação da recusa.

    Critério (auditado manualmente nos resultados): é RECUSA TOTAL quando o declínio aparece na abertura
    da resposta (primeiros 250 caracteres) e a resposta não traz nenhum fato-chave da referência.
    Declínio em qualquer outro ponto (ex.: só na seção "Ação corretiva", ou "Causa: não há informação")
    é RESSALVA PARCIAL — o modelo respondeu, e se errou é erro de conteúdo (key_fact_recall), não recusa.
    """
    text = answer.replace(",", ".")
    if not _DECLINE_RE.search(text):
        return "resposta"
    if _DECLINE_RE.search(text[:250]) and not (key_fact > 0):  # NaN > 0 é False -> NaN conta como sem fatos
        return "recusa_total"
    return "ressalva_parcial"


def ungrounded_numbers(answer: str, question: str, contexts: list[str]) -> int:
    """Nº de valores decimais citados na resposta que não aparecem na pergunta nem nos episódios
    recuperados (proxy determinístico de alucinação numérica; durações derivadas por subtração, p.ex.
    "2.8s acima da mediana", podem gerar falsos positivos — reportar como indicador, não como verdade)."""
    allowed = {float(x) for x in _DECIMAL_RE.findall((question + " " + " ".join(contexts)).replace(",", "."))}
    cited = {float(x) for x in _DECIMAL_RE.findall(answer.replace(",", "."))}
    return sum(1 for v in cited if v not in allowed)


def stage_refusals() -> None:
    """Comportamento de recusa e ancoragem por gerador+prompt — só lê generations.jsonl (sem juiz/GPU).

    Cruza a EVIDÊNCIA NO CONTEXTO (hit@k) com o COMPORTAMENTO da resposta:
      - recusa indevida   : evidência no contexto, mas recusa total (sem fatos-chave);
      - ressalva parcial  : evidência no contexto, respondeu e só ressalvou uma seção (não é erro);
      - recusa correta    : sem evidência no contexto e recusou (comportamento desejado);
      - resp. s/ evidência: sem evidência no contexto e NÃO recusou (risco de alucinação);
      - c/ nº inventado   : % das respostas (todas) citando decimal que não está no contexto/pergunta.
    Compara só as perguntas presentes em TODAS as execuções, para que rodadas parciais (--sample)
    não distorçam a comparação.
    """
    golden = {g["id"]: g for g in _load_golden()}
    dirs = [BASE_RESULTS] + sorted(p for p in BASE_RESULTS.iterdir() if p.is_dir() and p.name != "figuras")
    runs: list[tuple[str, dict[str, dict[str, dict]]]] = []  # rótulo -> sistema -> id -> geração
    for d in dirs:
        gens = [g for g in _read_jsonl(d / "generations.jsonl") if g["system"] != "baseline"]
        if not gens:
            continue
        model_name = gens[0].get("model", LEGACY_MODEL if d == BASE_RESULTS else d.name)
        label = f"{model_name}[{gens[0].get('prompt', DEFAULT_PROMPT)}]"
        if gens[0].get("rerank", DEFAULT_RERANK) != DEFAULT_RERANK:
            label += f"+{gens[0]['rerank']}"
        by_sys: dict[str, dict[str, dict]] = {}
        for g in gens:
            by_sys.setdefault(g["system"], {})[g["id"]] = g
        runs.append((label, by_sys))
    if not runs:
        print("Nenhuma geração encontrada. Rode 'generate' antes.")
        return

    def pct(k: int, n: int) -> str:
        return f"{k:2d}/{n:<2d} ({(100 * k / n) if n else 0:3.0f}%)"

    for system in ("rag_filtrado", "rag_vetorial"):
        common = set.intersection(*[set(r[1].get(system, {})) for r in runs])
        print(f"\n== {SYSTEM_LABEL[system]} — {len(common)} perguntas em comum entre as execuções ==")
        if not common:
            print("  (nenhuma pergunta em comum; gere as mesmas perguntas em todas as execuções)")
            continue
        print(f"{'gerador[prompt]':24} {'recusa indevida':>16} {'ressalva parcial':>17} {'recusa correta':>16} "
              f"{'resp. s/ evid.':>15} {'c/ nº inventado':>16} {'FatosChave':>10}")
        for label, by_sys in runs:
            ev = n_undue = n_partial = n_correct = n_noev_ans = n_invent = 0
            kfs: list[float] = []
            for qid in sorted(common):
                g, gold = by_sys[system][qid], golden[qid]
                hit = retrieval_metrics(g["retrieved_ids"], gold["evidence_episode_ids"],
                                        gold["evidence_mode"])["hit_at_k"] > 0
                kf = key_fact_recall(g["answer"], g["reference"], g["question"])
                kind = classify_answer(g["answer"], kf)
                ev += hit
                n_undue += hit and kind == "recusa_total"
                n_partial += hit and kind == "ressalva_parcial"
                n_correct += (not hit) and kind == "recusa_total"
                n_noev_ans += (not hit) and kind != "recusa_total"
                n_invent += ungrounded_numbers(g["answer"], g["question"], g["retrieved_contexts"]) > 0
                if not math.isnan(kf):
                    kfs.append(kf)
            n, n_noev = len(common), len(common) - ev
            print(f"{label:24} {pct(n_undue, ev):>16} {pct(n_partial, ev):>17} {pct(n_correct, n_noev):>16} "
                  f"{pct(n_noev_ans, n_noev):>15} {pct(n_invent, n):>16} {np.mean(kfs):10.2f}")
    print("\nLeitura: 'recusa indevida' (base = perguntas com evidência no contexto) deve CAIR no prompt brando;\n"
          "'resp. s/ evid.' (base = perguntas sem evidência no contexto) e 'c/ nº inventado' não devem SUBIR —\n"
          "se subirem, o prompt brando trocou recusa indevida por alucinação.")


def stage_retrieval() -> None:
    """Ablação da recuperação (determinística: sem LLM, sem juiz, ~1 min): denso vs denso+BM25 (RRF).

    Compara, no golden set, {vetorial, filtrado} x {sem, com reranking lexical BM25}. É independente do
    gerador. Hiperparâmetros (RRF k=60, BM25 k1=1.5/b=0.75) são os padrões da literatura, fixados antes
    de ver o resultado — o golden set NÃO é usado para ajustá-los. Salva results/retrieval_ablation.json.
    """
    from rag.retriever import retrieve, retrieve_hybrid  # noqa: PLC0415

    golden = _load_golden()
    configs = [("vetorial", retrieve, ""), ("vetorial+BM25", retrieve, "bm25"),
               ("filtrado", retrieve_hybrid, ""), ("filtrado+BM25", retrieve_hybrid, "bm25")]
    per: dict[str, dict[str, dict]] = {}
    previous = os.environ.get("RAG_RERANK")
    try:
        for name, fn, mode in configs:
            os.environ["RAG_RERANK"] = mode
            per[name] = {}
            for item in golden:
                hits = fn(item["question"], top_k=TOP_K)
                per[name][item["id"]] = {
                    **retrieval_metrics([h["event_id"] for h in hits], item["evidence_episode_ids"],
                                        item["evidence_mode"]),
                    "tier": item["tier"], "retrieved_ids": [h["event_id"] for h in hits]}
    finally:
        if previous is None:
            os.environ.pop("RAG_RERANK", None)
        else:
            os.environ["RAG_RERANK"] = previous

    cols = ("hit_at_k", "mrr", "recall_at_k")
    head = f"{'configuração':16} {'Hit@' + str(TOP_K):>8} {'MRR':>8} {'Recall@' + str(TOP_K):>10}"
    print(f"\n== Recuperação — {len(golden)} perguntas (média) ==\n{head}")
    for name, _, _ in configs:
        print(f"{name:16} " + " ".join(f"{np.mean([r[c] for r in per[name].values()]):{w}.3f}"
                                       for c, w in zip(cols, (8, 8, 10))))
    print("\n== Por tier (Hit@k / MRR / Recall@k) ==")
    for tier in TIERS:
        for name, _, _ in configs:
            sel = [r for r in per[name].values() if r["tier"] == tier]
            print(f"{TIER_LABEL[tier]:15} {name:16} " + " / ".join(f"{np.mean([r[c] for r in sel]):.3f}"
                                                                 for c in cols))

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    print("\n== Efeito do BM25 (Δ pareado por pergunta, IC95% bootstrap) ==")
    for base in ("vetorial", "filtrado"):
        for c in ("mrr", "recall_at_k"):
            r_ = _paired_diff({k: v[c] for k, v in per[base + "+BM25"].items()},
                              {k: v[c] for k, v in per[base].items()}, rng)
            print(f"{base:9} {c:12} Δ={r_['diff']:+.3f} [{r_['ci_low']:+.3f}; {r_['ci_high']:+.3f}] n={r_['n']}")
    print("\n== Perguntas cujo ranking mudou com o BM25 (MRR antes -> depois) ==")
    changed = False
    for base in ("vetorial", "filtrado"):
        for qid, before in per[base].items():
            after = per[base + "+BM25"][qid]
            if abs(after["mrr"] - before["mrr"]) > 1e-9 or after["hit_at_k"] != before["hit_at_k"]:
                changed = True
                print(f"{base:9} {qid}: MRR {before['mrr']:.2f} -> {after['mrr']:.2f} | "
                      f"Hit {before['hit_at_k']:.0f} -> {after['hit_at_k']:.0f}")
    if not changed:
        print("(nenhuma pergunta mudou)")

    out = BASE_RESULTS / "retrieval_ablation.json"
    out.write_text(json.dumps(per, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSalvo em {out}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    for noisy in ("httpx", "urllib3", "pymilvus", "rag.retriever", "rag.generator", "rag.pipeline"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["generate", "ragas", "report", "compare", "refusals", "retrieval"])
    parser.add_argument("--limit", type=int, default=None, help="usa só as N primeiras perguntas (smoke test)")
    parser.add_argument("--sample", type=int, default=None,
                        help="só no 'generate': N perguntas espaçadas pelo golden set (cobre os 3 tiers)")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", LEGACY_MODEL),
                        help=f"gerador a avaliar (padrão: LLM_MODEL ou {LEGACY_MODEL}); resultados em results/<modelo>/")
    parser.add_argument("--judge", default=JUDGE_MODEL,
                        help=f"juiz do RAGAS (padrão: JUDGE_MODEL ou {DEFAULT_JUDGE}); use outra família que o gerador")
    parser.add_argument("--prompt", choices=PROMPTS, default=DEFAULT_PROMPT,
                        help="regra de recusa do prompt do RAG (padrão: estrito = prompt dos resultados já "
                             "avaliados; brando = só recusa se NENHUM episódio tratar do assunto). "
                             "Resultados em results/<modelo>__<prompt>/")
    parser.add_argument("--rerank", choices=RERANKS, default=DEFAULT_RERANK,
                        help="reranking lexical BM25+RRF dos candidatos do retriever (padrão: off). "
                             "Resultados em results/<modelo>[__prompt]__bm25/")
    args = parser.parse_args()
    _configure(args.model, args.judge, args.prompt, args.rerank)
    {"generate": lambda: stage_generate(args.limit, args.sample),
     "ragas": lambda: stage_ragas(args.limit),
     "report": stage_report,
     "compare": stage_compare,
     "refusals": stage_refusals,
     "retrieval": stage_retrieval}[args.stage]()


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT.parent))
    main()
