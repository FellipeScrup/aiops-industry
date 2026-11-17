"""Validação do juiz do RAGAS (phi4) contra rótulos humanos e checagens de robustez.

Subcomandos (rodar na ordem):

  export     Sorteia ~48 respostas (estratificadas por gerador x sistema x faixa de faithfulness) e grava
             a planilha de rotulagem CEGA (sem a nota do juiz) + o gabarito separado.
  noise      Ruído do juiz: reavalia a faithfulness de k respostas N vezes com o MESMO juiz/config e
             compara com a nota guardada (reprodutibilidade run-a-run).
  minicheck  Checagem cruzada: faithfulness por sentença com o Bespoke-MiniCheck-7B (verificador de
             grounding especializado, família distinta do phi4), sobre as mesmas respostas.
  perturb    Validação SEM rótulo humano: corrompe respostas de forma controlada (números e estação
             trocados por valores fora do contexto; contexto de outra pergunta) e mede se a nota do juiz
             cai, contra um controle neutro (reordenar seções) e um controle positivo (respostas de
             referência do golden set). Retomável.
  score      Concordância juiz x humano (Spearman, kappa de Cohen, acurácia balanceada, ICs bootstrap)
             e, se existir, MiniCheck x humano / MiniCheck x juiz.

Protocolo de rotulagem (definido ANTES de ver qualquer rótulo ou nota):
  2 = todas as afirmações factuais da resposta estão sustentadas pelo contexto recuperado;
  1 = há pelo menos uma afirmação sem suporte ou contradita, mas a maioria está sustentada;
  0 = a maioria das afirmações NÃO está sustentada (ou contradiz o contexto);
  R = a resposta só declina/recusa (sem afirmações factuais) — analisada à parte.
Critérios (fixados ANTES de ver qualquer rótulo ou nota):
  primário   : Spearman entre o rótulo humano (0/1/2) e a faithfulness contínua do juiz;
  secundários: kappa/acurácia balanceada em dois cortes binários — (A) "totalmente suportada" (humano==2)
               vs juiz>=0,999 e (B) "maioria suportada" (humano>=1) vs juiz>=0,5.
Recusas puras ficam fora da planilha (a faithfulness do RAGAS não é definida para quem não afirma nada);
o comportamento do juiz nelas é reportado à parte, de forma determinística (recusas_juiz.json).
Julgar apenas pelo contexto mostrado, NUNCA por conhecimento externo.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_eval as R  # noqa: E402

logger = logging.getLogger("judge_validation")

VAL_DIR = R.BASE_RESULTS / "validation"
SHEET = VAL_DIR / "planilha_rotulagem.csv"      # o que o humano preenche (cega)
KEY = VAL_DIR / "gabarito_juiz.csv"             # amostra -> origem + nota do juiz (NÃO abrir antes de rotular)
INSTR = VAL_DIR / "instrucoes_rotulagem.txt"
NOISE_OUT = VAL_DIR / "noise_check.json"
MINICHECK_OUT = VAL_DIR / "minicheck_scores.jsonl"
SUMMARY_OUT = VAL_DIR / "validation_summary.json"

CSV_KW = dict(delimiter=";", quoting=csv.QUOTE_ALL)  # ';' = separador do Excel pt-BR
MINICHECK_MODEL = "bespoke-minicheck"
MAX_CLAIMS = 12
SEED = 42

INSTRUCTIONS = """\
ROTULAGEM DE FIDELIDADE (faithfulness) — instruções
====================================================
Abra planilha_rotulagem.csv no Excel (separador ';', UTF-8). Preencha SÓ a coluna 'rotulo'
(e, se quiser, 'comentario'). NÃO abra gabarito_juiz.csv antes de terminar (ele tem a nota do juiz).

Para cada linha, leia a RESPOSTA e verifique cada afirmação factual dela contra o CONTEXTO RECUPERADO
mostrado na própria linha. Use APENAS o contexto — nunca conhecimento externo nem a sua memória do dataset.

  2 = todas as afirmações factuais estão sustentadas pelo contexto
  1 = há pelo menos uma afirmação sem suporte ou contradita, mas a maioria está sustentada
  0 = a maioria das afirmações NÃO está sustentada (ou contradiz o contexto)
  R = a resposta só recusa/declina ("não há informação suficiente...") e não afirma fatos
      (recusas óbvias já foram retiradas da planilha; use R só se sobrar alguma)

O que conta como AFIRMAÇÃO a verificar (o RAGAS faz o mesmo ao decompor a resposta em afirmações):
  - fatos: durações, estações, tarefas, sub-tarefas, horários, estado "not ready", anomalia de duração;
  - CAUSAS PROVÁVEIS e AÇÕES CORRETIVAS recomendadas: se o contexto não as traz nem as implica,
    contam como afirmação SEM suporte (os logs não contêm ações corretivas; recomendar uma é inferir
    além do contexto);
  - NÃO contam: ressalvas do tipo "não há informação sobre a causa/ação" (isso é reconhecer um limite).

Dicas:
  - Números (durações), estações, tarefas e sub-tarefas citados devem aparecer no contexto.
  - Se estiver em dúvida entre dois rótulos, escolha o mais baixo e anote em 'comentario'.
Não precisa de pressa, mas evite consultar as respostas do mesmo sorteio em outra janela.
"""


# ── utilidades ──────────────────────────────────────────────────────────────

def _run_dirs() -> list[tuple[str, Path]]:
    """(rótulo, pasta) de cada execução que tem generations E ragas_scores do juiz atual."""
    out = []
    dirs = [R.BASE_RESULTS] + sorted(p for p in R.BASE_RESULTS.iterdir()
                                     if p.is_dir() and not p.name.startswith("figuras") and p.name != "validation")
    for d in dirs:
        gens = R._read_jsonl(d / "generations.jsonl")
        if not gens or not (d / "ragas_scores.jsonl").exists():
            continue
        model = gens[0].get("model", R.LEGACY_MODEL if d == R.BASE_RESULTS else d.name)
        label = f"{model}[{gens[0].get('prompt', R.DEFAULT_PROMPT)}]"
        if gens[0].get("rerank", R.DEFAULT_RERANK) != R.DEFAULT_RERANK:
            label += f"+{gens[0]['rerank']}"
        out.append((label, d))
    return out


def _rank(x: np.ndarray) -> np.ndarray:
    """Postos com empates pela média (para Spearman)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):
        m = x == v
        ranks[m] = ranks[m].mean()
    return ranks


def spearman(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def kappa_binary(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=bool), np.asarray(y_pred, dtype=bool)
    n = len(y_true)
    if n == 0:
        return float("nan")
    po = float((y_true == y_pred).mean())
    pe = float(y_true.mean() * y_pred.mean() + (1 - y_true.mean()) * (1 - y_pred.mean()))
    return float("nan") if pe == 1.0 else (po - pe) / (1 - pe)


def balanced_accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=bool), np.asarray(y_pred, dtype=bool)
    pos, neg = y_true.sum(), (~y_true).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    return float(0.5 * ((y_pred & y_true).sum() / pos + ((~y_pred) & (~y_true)).sum() / neg))


def _boot(fn, *arrays, n: int = R.BOOTSTRAP_N, seed: int = R.BOOTSTRAP_SEED) -> tuple[float, float, float]:
    """(estimativa, IC95% baixo, alto) por bootstrap de pares; ignora réplicas indefinidas (NaN)."""
    arrays = [np.asarray(a) for a in arrays]
    est = fn(*arrays)
    rng = np.random.default_rng(seed)
    m = len(arrays[0])
    vals = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        v = fn(*[a[idx] for a in arrays])
        if not math.isnan(v):
            vals.append(v)
    if len(vals) < n * 0.5:
        return est, float("nan"), float("nan")
    return est, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _fmt(t: tuple[float, float, float]) -> str:
    return f"{t[0]:.2f} [{t[1]:.2f}; {t[2]:.2f}]"


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh, **CSV_KW))


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, **CSV_KW)
        w.writeheader()
        w.writerows(rows)


def _gen_lookup(pasta: str) -> dict[tuple[str, str], dict]:
    d = R.BASE_RESULTS if pasta in (".", "") else R.BASE_RESULTS / pasta
    return {(g["id"], g["system"]): g for g in R._read_jsonl(d / "generations.jsonl")}


# ── 1) export ───────────────────────────────────────────────────────────────

def cmd_export(n_per_cell: int, force: bool) -> None:
    if SHEET.exists() and not force:
        sys.exit(f"{SHEET} já existe (pode conter rótulos seus). Use --force para sobrescrever "
                 "(você PERDERÁ os rótulos).")
    # Recusas puras ficam FORA da planilha: o RAGAS não define faithfulness para quem não afirma nada
    # (a nota do juiz nelas é arbitrária — analisada de forma determinística, sem humano).
    cells: dict[tuple[str, str], list[dict]] = {}
    refusal_scores: list[float] = []
    for label, d in _run_dirs():
        gens = {(g["id"], g["system"]): g for g in R._read_jsonl(d / "generations.jsonl")}
        scores = {(r["id"], r["system"]): r for r in R._read_jsonl(d / "ragas_scores.jsonl")
                  if r.get("judge") == R.JUDGE_MODEL}
        for (qid, sysname), sc in sorted(scores.items()):
            if sysname == "baseline" or sc.get("faithfulness") is None or (qid, sysname) not in gens:
                continue
            g = gens[(qid, sysname)]
            kind = R.classify_answer(g["answer"], R.key_fact_recall(g["answer"], g["reference"], g["question"]))
            if kind == "recusa_total":
                refusal_scores.append(float(sc["faithfulness"]))
                continue
            cells.setdefault((label, sysname), []).append(
                {"run": label, "pasta": "." if d == R.BASE_RESULTS else d.name, "id": qid, "system": sysname,
                 "tier": g["tier"], "faithfulness_juiz": float(sc["faithfulness"]), "gen": g})
    if not cells:
        sys.exit("Nenhuma resposta com faithfulness do juiz encontrada. Rode 'make eval-ragas' antes.")

    # Cota por célula (gerador x sistema) em rodízio, limitada ao que existe; dentro da célula, amostra
    # espaçada pela nota do juiz (cobre toda a faixa de faithfulness).
    total_target = n_per_cell * len(cells)
    quota = {c: 0 for c in cells}
    while sum(quota.values()) < total_target and any(quota[c] < len(cells[c]) for c in cells):
        for c in sorted(cells):
            if quota[c] < len(cells[c]) and sum(quota.values()) < total_target:
                quota[c] += 1
    pool: list[dict] = []
    for c, rows_c in sorted(cells.items()):
        rows_c.sort(key=lambda r: (r["faithfulness_juiz"], r["id"]))
        take = sorted({int(round(i)) for i in np.linspace(0, len(rows_c) - 1, quota[c])}) if quota[c] else []
        pool.extend(rows_c[i] for i in take)

    rf = np.array(refusal_scores)
    refusal_info = {"n_recusas_total": len(rf),
                    "faithfulness_juiz": {"zero": int((rf == 0).sum()), "intermediaria": int(((rf > 0) & (rf < 0.999)).sum()),
                                          "um": int((rf >= 0.999).sum()), "media": float(rf.mean()) if len(rf) else None},
                    "n_nao_recusa_disponiveis": sum(len(v) for v in cells.values()),
                    "cotas": {f"{c[0]} | {c[1]}": quota[c] for c in sorted(cells)}}
    (VAL_DIR).mkdir(parents=True, exist_ok=True)
    (VAL_DIR / "recusas_juiz.json").write_text(json.dumps(refusal_info, ensure_ascii=False, indent=1), encoding="utf-8")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(pool))
    sheet, key = [], []
    for j, i in enumerate(order, start=1):
        p, g = pool[int(i)], pool[int(i)]["gen"]
        sid = f"A{j:02d}"
        ctx = "\n".join(f"[{k}] {c}" for k, c in enumerate(g["retrieved_contexts"], start=1))
        sheet.append({"amostra": sid, "pergunta": g["question"], "contexto_recuperado": ctx,
                      "resposta": g["answer"], "rotulo": "", "comentario": ""})
        key.append({"amostra": sid, "run": p["run"], "pasta": p["pasta"], "id": p["id"],
                    "system": p["system"], "tier": p["tier"], "faithfulness_juiz": p["faithfulness_juiz"]})
    _write_csv(SHEET, sheet, ["amostra", "pergunta", "contexto_recuperado", "resposta", "rotulo", "comentario"])
    _write_csv(KEY, key, ["amostra", "run", "pasta", "id", "system", "tier", "faithfulness_juiz"])
    INSTR.write_text(INSTRUCTIONS, encoding="utf-8")
    print(f"Recusas puras excluídas da planilha: {len(rf)} (nota do juiz nelas: "
          f"{refusal_info['faithfulness_juiz']['zero']} zero, {refusal_info['faithfulness_juiz']['intermediaria']} "
          f"intermediárias, {refusal_info['faithfulness_juiz']['um']} com 1,0) -> {VAL_DIR / 'recusas_juiz.json'}")
    print(f"{len(sheet)} amostras (de {refusal_info['n_nao_recusa_disponiveis']} respostas que não são recusa) -> {SHEET}\n"
          f"Instruções: {INSTR}\nGabarito (NÃO abrir antes de rotular): {KEY}")


# ── 2) ruído do juiz ────────────────────────────────────────────────────────

def _faithfulness_once(llm, emb, g: dict) -> float | None:
    from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: PLC0415
    from ragas.metrics import Faithfulness  # noqa: PLC0415
    from ragas.run_config import RunConfig  # noqa: PLC0415

    sample = SingleTurnSample(user_input=g["question"], response=g["answer"],
                              retrieved_contexts=g["retrieved_contexts"], reference=g["reference"])
    try:
        res = evaluate(EvaluationDataset([sample]), metrics=[Faithfulness()], llm=llm, embeddings=emb,
                       run_config=RunConfig(timeout=600, max_workers=1, max_retries=2),
                       show_progress=False, raise_exceptions=False)
        v = res.to_pandas().iloc[0].to_dict().get("faithfulness")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Falha: %s", exc)
        return None
    return None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)


def cmd_noise(k: int, repeats: int) -> None:
    if not KEY.exists():
        sys.exit("Rode 'export' antes (o ruído é medido sobre a mesma amostra).")
    key = _read_csv(KEY)
    key.sort(key=lambda r: float(r["faithfulness_juiz"]))
    pick = sorted({int(round(i)) for i in np.linspace(0, len(key) - 1, min(k, len(key)))})
    rows = [key[i] for i in pick]
    llm, emb = R._build_judge()
    out = []
    for n, r in enumerate(rows, start=1):
        g = _gen_lookup(r["pasta"])[(r["id"], r["system"])]
        runs = [float(r["faithfulness_juiz"])]  # medição original (já guardada)
        t0 = time.perf_counter()
        for _ in range(repeats):
            runs.append(_faithfulness_once(llm, emb, g))
        valid = [v for v in runs if v is not None]
        out.append({"amostra": r["amostra"], "run": r["run"], "id": r["id"], "system": r["system"],
                    "medicoes": runs, "amplitude": (max(valid) - min(valid)) if len(valid) > 1 else None})
        logger.info("[%d/%d] %s %s/%s -> %s (%.0fs)", n, len(rows), r["amostra"], r["id"], r["system"],
                    [None if v is None else round(v, 3) for v in runs], time.perf_counter() - t0)
    amps = [o["amplitude"] for o in out if o["amplitude"] is not None]
    summary = {"judge": R.JUDGE_MODEL, "k": len(out), "repeats": repeats,
               "nota": "medição 0 = guardada na avaliação original; as demais = repetições com a mesma config "
                       "(temperature=0, seed=42)",
               "identicas": int(sum(a == 0 for a in amps)), "n_validas": len(amps),
               "amplitude_media": float(np.mean(amps)) if amps else None,
               "amplitude_max": float(np.max(amps)) if amps else None, "linhas": out}
    NOISE_OUT.parent.mkdir(parents=True, exist_ok=True)
    NOISE_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n== Ruído do juiz ({R.JUDGE_MODEL}): {len(out)} respostas x {repeats + 1} medições ==")
    print(f"Idênticas em todas as medições: {summary['identicas']}/{summary['n_validas']}")
    if amps:
        print(f"Amplitude (máx - mín) por resposta: média {summary['amplitude_media']:.3f}, "
              f"máxima {summary['amplitude_max']:.3f}")
    print(f"Salvo em {NOISE_OUT}")


# ── 3) checagem cruzada: Bespoke-MiniCheck ──────────────────────────────────

_MD_RE = re.compile(r"[*#`_]+")
_LABEL_RE = re.compile(r"^\s*(?:\d+[.)]\s*)?(?:[-•]\s*)?"
                       r"(?:diagn[óo]stico|causa prov[áa]vel|a[çc][ãa]o corretiva)[^:\n]{0,60}:\s*", re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ0-9'\"(])")


def split_claims(answer: str) -> tuple[list[str], int]:
    """Quebra a resposta em sentenças verificáveis. Devolve (afirmações, nº de sentenças de declínio).

    Descarta títulos curtos (<4 palavras) e sentenças de declínio ("não há informação..."), que não
    são afirmações factuais a verificar (mesma regra do rótulo humano 'R' e das ressalvas).
    """
    claims: list[str] = []
    declined = 0
    for line in answer.replace("\r", "").split("\n"):
        line = _MD_RE.sub("", line)
        line = _LABEL_RE.sub("", line)
        line = re.sub(r"^\s*(?:\d+[.)]|[-•])\s*", "", line).strip()
        if not line:
            continue
        for sent in _SENT_SPLIT.split(line):
            sent = sent.strip()
            if len(sent.split()) < 4:
                continue
            if R._DECLINE_RE.search(sent.replace(",", ".")):
                declined += 1
                continue
            claims.append(sent)
    return claims[:MAX_CLAIMS], declined


def _minicheck_verdict(document: str, claim: str) -> bool | None:
    import requests  # noqa: PLC0415
    base = R.os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    resp = requests.post(f"{base}/api/chat", timeout=300, json={
        "model": MINICHECK_MODEL, "stream": False,
        "messages": [{"role": "user", "content": f"Document: {document}\nClaim: {claim}\n"}],
        "options": {"temperature": 0, "seed": SEED, "num_ctx": 8192, "num_predict": 4},
    })
    resp.raise_for_status()
    text = resp.json().get("message", {}).get("content", "").strip().lower()
    if text.startswith("yes"):
        return True
    if text.startswith("no"):
        return False
    return None


def cmd_minicheck(limit: int | None) -> None:
    if not KEY.exists():
        sys.exit("Rode 'export' antes (a checagem cruzada usa as mesmas amostras).")
    key = _read_csv(KEY)[:limit] if limit else _read_csv(KEY)
    done = {r["amostra"] for r in R._read_jsonl(MINICHECK_OUT)}
    for n, r in enumerate(key, start=1):
        if r["amostra"] in done:
            continue
        g = _gen_lookup(r["pasta"])[(r["id"], r["system"])]
        doc = "\n\n".join(g["retrieved_contexts"])
        claims, declined = split_claims(g["answer"])
        t0 = time.perf_counter()
        verdicts = [_minicheck_verdict(doc, c) for c in claims]
        valid = [v for v in verdicts if v is not None]
        score = (sum(valid) / len(valid)) if valid else None
        R._append_jsonl(MINICHECK_OUT, {
            "amostra": r["amostra"], "id": r["id"], "system": r["system"], "n_claims": len(claims),
            "n_declined": declined, "n_unparsed": len(verdicts) - len(valid), "minicheck_faithfulness": score,
            "claims": [{"texto": c, "suportada": v} for c, v in zip(claims, verdicts)]})
        logger.info("[%d/%d] %s %s/%s: %d afirmações, score=%s (%.0fs)", n, len(key), r["amostra"], r["id"],
                    r["system"], len(claims), None if score is None else round(score, 2), time.perf_counter() - t0)


# ── 4) concordância ─────────────────────────────────────────────────────────

SHEET_FULL = VAL_DIR / "planilha_rotulagem_completa.csv"
QUICK_GUIDE = """
PASSO A PASSO RÁPIDO (cerca de 2 minutos por linha; você NÃO precisa entender de fábrica)
-----------------------------------------------------------------------------------------
Para cada linha, pegue os números, estações, tarefas e sub-tarefas citados na RESPOSTA e procure cada um no
CONTEXTO (Ctrl+F no Excel funciona). Conte:
  - tudo o que a resposta afirma está no contexto            -> rótulo 2
  - a maior parte está, mas há algo inventado ou sem apoio   -> rótulo 1
  - a maior parte NÃO está no contexto ou o contradiz        -> rótulo 0
Lembre: "causa provável" e "ação corretiva" só valem se o contexto as trouxer ou implicar; se são sugestões gerais
(ex.: "verifique os sensores"), contam como afirmação sem apoio.
"""


def cmd_subset(n: int) -> None:
    """Reduz a planilha a N linhas espalhadas pela faixa de notas do juiz (rotulagem em ~30 min)."""
    if not (SHEET.exists() and KEY.exists()):
        sys.exit("Rode 'export' antes.")
    rows = _read_csv(SHEET)
    if any((r.get("rotulo") or "").strip() for r in rows):
        sys.exit("A planilha atual já tem rótulos; não vou sobrescrever. Rotule-a ou apague os rótulos antes.")
    if not SHEET_FULL.exists():
        _write_csv(SHEET_FULL, rows, list(rows[0].keys()))  # cópia da planilha completa
    key = {r["amostra"]: r for r in _read_csv(KEY)}
    order = sorted(rows, key=lambda r: float(key[r["amostra"]]["faithfulness_juiz"]))
    idx = sorted({int(round(i)) for i in np.linspace(0, len(order) - 1, min(n, len(order)))})
    chosen = [order[i] for i in idx]
    chosen.sort(key=lambda r: r["amostra"])  # ordem de leitura sem relação com a nota
    _write_csv(SHEET, chosen, list(rows[0].keys()))
    INSTR.write_text(INSTRUCTIONS + QUICK_GUIDE, encoding="utf-8")
    gens = sorted({key[r["amostra"]]["run"] for r in chosen})
    print(f"Planilha reduzida a {len(chosen)} linhas (completa guardada em {SHEET_FULL.name}).")
    print(f"Execuções representadas: {', '.join(gens)}")
    print(f"Faixa de notas do juiz coberta: {float(key[order[idx[0]]['amostra']]['faithfulness_juiz']):.2f} a "
          f"{float(key[order[idx[-1]]['amostra']]['faithfulness_juiz']):.2f}")
    print(f"Rotule: {SHEET}\nInstruções: {INSTR}")


def cmd_score(sheet: Path | None = None, annotator: str = "humano") -> None:
    sheet = sheet or SHEET
    if not (sheet.exists() and KEY.exists()):
        sys.exit("Rode 'export' e rotule a planilha antes.")
    print(f"Anotador: {annotator} | planilha: {sheet.name}")
    key = {r["amostra"]: r for r in _read_csv(KEY)}
    mini = {r["amostra"]: r for r in R._read_jsonl(MINICHECK_OUT)}
    labeled, refusals = [], []
    n_total = n_blank = n_invalid = 0
    for r in _read_csv(sheet):
        n_total += 1
        lab = (r.get("rotulo") or "").strip().upper()
        if not lab:
            n_blank += 1
            continue
        k = key[r["amostra"]]
        row = {"amostra": r["amostra"], "run": k["run"], "juiz": float(k["faithfulness_juiz"]),
               "mini": (mini.get(r["amostra"]) or {}).get("minicheck_faithfulness")}
        if lab == "R":
            refusals.append(row)
        elif lab in {"0", "1", "2"}:
            row["humano"] = int(lab)
            labeled.append(row)
        else:
            n_invalid += 1
    print(f"Rotuladas: {len(labeled) + len(refusals)}/{n_total} (em branco: {n_blank}, inválidas: {n_invalid})")
    if len(labeled) < 10:
        sys.exit("Poucas amostras rotuladas (<10) para calcular concordância.")

    h = np.array([r["humano"] for r in labeled])
    j = np.array([r["juiz"] for r in labeled])
    summary: dict = {"n_rotuladas": len(labeled), "n_recusas_R": len(refusals),
                     "criterio_primario": "Spearman(humano ordinal, faithfulness contínua do juiz)"}

    who = "humano" if annotator == "humano" else f"anotador {annotator} (NÃO é validação humana)"
    print(f"\n== Juiz ({R.JUDGE_MODEL}) x {who} — n = {len(labeled)} (recusas 'R' excluídas) ==")
    print(f"Nota média do juiz por rótulo do {annotator} (esperado: crescente):")
    for lab in (0, 1, 2):
        sel = j[h == lab]
        print(f"  rótulo {lab}: n={len(sel):2d}  faithfulness média = "
              f"{(sel.mean() if len(sel) else float('nan')):.2f}")
    sp = _boot(spearman, h, j)
    print(f"[PRIMÁRIO] Spearman (humano ordinal x juiz contínuo): {_fmt(sp)}")
    summary["juiz_vs_humano"] = {"spearman": sp}
    cuts = {"A_totalmente_suportada": ("humano==2 vs juiz>=0,999", h == 2, j >= 0.999),
            "B_maioria_suportada": ("humano>=1 vs juiz>=0,5", h >= 1, j >= 0.5)}
    for name, (desc, yh, yj) in cuts.items():
        tp, fn = int((yh & yj).sum()), int((yh & ~yj).sum())
        fp, tn = int((~yh & yj).sum()), int((~yh & ~yj).sum())
        print(f"\n[SECUNDÁRIO {name[0]}] {desc}: humano positivo={int(yh.sum())}, juiz positivo={int(yj.sum())}")
        if yh.sum() < 3 or yj.sum() < 3 or (~yh).sum() < 3 or (~yj).sum() < 3:
            print("  corte quase degenerado (menos de 3 casos em alguma classe): kappa não é confiável aqui.")
        acc = _boot(lambda a, b: float((a == b).mean()), yh, yj)
        ba = _boot(balanced_accuracy, yh, yj)
        ka = _boot(kappa_binary, yh, yj)
        print(f"  acurácia {_fmt(acc)} | acurácia balanceada {_fmt(ba)} | kappa de Cohen {_fmt(ka)}")
        print(f"  matriz (humano x juiz): ambos positivos={tp} | só humano={fn} | só juiz={fp} | ambos negativos={tn}")
        summary["juiz_vs_humano"][name] = {"corte": desc, "kappa": ka, "acuracia_balanceada": ba, "acuracia": acc,
                                           "matriz": {"ambos_pos": tp, "so_humano": fn, "so_juiz": fp, "ambos_neg": tn}}
    print("\nLeitura do kappa (Landis & Koch): <0,40 fraca | 0,40-0,60 moderada | 0,60-0,80 substancial | >0,80 quase perfeita")
    rj_path = VAL_DIR / "recusas_juiz.json"
    if rj_path.exists():
        ri = json.loads(rj_path.read_text(encoding="utf-8"))
        fj = ri["faithfulness_juiz"]
        print(f"\nRecusas puras (fora da planilha): n={ri['n_recusas_total']}; nota do juiz: {fj['zero']} com 0,0 | "
              f"{fj['intermediaria']} intermediárias | {fj['um']} com 1,0 (média {fj['media']:.2f}) — "
              "a faithfulness não é definida para respostas sem afirmações.")
        summary["recusas_juiz"] = ri

    if refusals:
        rj = np.array([r["juiz"] for r in refusals])
        print(f"\nRecusas puras ('R'): n={len(refusals)}, faithfulness média do juiz = {rj.mean():.2f} "
              "(RAGAS penaliza recusa: nota baixa aqui é artefato da métrica, não alucinação)")
        summary["recusas_R"] = {"n": len(refusals), "faithfulness_media_juiz": float(rj.mean())}

    mini_rows = [r for r in labeled if r["mini"] is not None]
    if len(mini_rows) >= 10:
        hm = np.array([r["humano"] for r in mini_rows])
        mm = np.array([r["mini"] for r in mini_rows])
        jm = np.array([r["juiz"] for r in mini_rows])
        print(f"\n== Checagem cruzada (Bespoke-MiniCheck) — n = {len(mini_rows)} ==")
        spm, spj = _boot(spearman, hm, mm), _boot(spearman, jm, mm)
        kam = _boot(kappa_binary, hm >= 1, mm >= 0.5)  # mesmo corte B do juiz (maioria suportada)
        print(f"MiniCheck x {annotator}: Spearman {_fmt(spm)} | kappa (corte B) {_fmt(kam)}")
        print(f"MiniCheck x juiz:   Spearman {_fmt(spj)}")
        summary["minicheck"] = {"n": len(mini_rows), "spearman_vs_humano": spm, "kappa_vs_humano": kam,
                                "spearman_vs_juiz": spj}
    elif MINICHECK_OUT.exists():
        print("\n(MiniCheck: menos de 10 amostras rotuladas com score; nada a comparar ainda.)")

    if NOISE_OUT.exists():
        nz = json.loads(NOISE_OUT.read_text(encoding="utf-8"))
        print(f"\nRuído do juiz (noise_check.json): idênticas {nz['identicas']}/{nz['n_validas']}, "
              f"amplitude média {nz['amplitude_media']}, máxima {nz['amplitude_max']}")
    summary["anotador"] = annotator
    out = SUMMARY_OUT if annotator == "humano" else SUMMARY_OUT.with_name(f"validation_summary_{annotator}.json")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSalvo em {out}")


# ── 4) perturbações com resposta conhecida (sem rótulo humano) ──────────────

PERTURB_OUT = VAL_DIR / "perturb_scores.jsonl"
PERTURB_SUMMARY = VAL_DIR / "perturb_summary.json"
_UNSUPPORTED_COND = {"numero": "troca de números citados por valores que NÃO estão no contexto",
                     "estacao": "troca da estação citada por outra que NÃO está no contexto",
                     "contexto": "mesma resposta, mas com episódios de OUTRA pergunta/estação como contexto"}
_NUM_FULL_RE = re.compile(r"(?<![\d.,])\d+[.,]\d+(?![\d])")


def _perturb_neutral(answer: str) -> str | None:
    """Controle: reordena as seções (parágrafos). Muda o texto sem mudar o conteúdo factual."""
    parts = [p for p in re.split(r"\n\s*\n", answer.strip()) if p.strip()]
    return "\n\n".join(reversed(parts)) if len(parts) >= 2 else None


def _perturb_numbers(answer: str, question: str, contexts: list[str]) -> str | None:
    ctx_vals = {float(x) for x in R._DECIMAL_RE.findall(" ".join(contexts).replace(",", "."))}
    allowed = ctx_vals | {float(x) for x in R._DECIMAL_RE.findall(question.replace(",", "."))}
    found = list(dict.fromkeys(_NUM_FULL_RE.findall(answer)))
    grounded = [s for s in found if float(s.replace(",", ".")) in ctx_vals]
    if not grounded:
        return None
    out = answer
    for k, s in enumerate(grounded[:2]):
        v = float(s.replace(",", "."))
        nv = v
        for delta in (7.3, 11.9, 4.6, 23.1):
            nv = round(v + delta + k, 1)
            if nv not in allowed:
                break
        new = f"{nv:.1f}".replace(".", ",") if "," in s else f"{nv:.1f}"
        out = re.sub(rf"(?<![\d.,]){re.escape(s)}(?!\d)", new, out)
    return out if out != answer else None


def _perturb_station(answer: str, question: str, contexts: list[str], universe: set[str]) -> str | None:
    ctx_st = set(R._STATION_RE.findall(" ".join(contexts) + " " + question))
    present = list(dict.fromkeys(R._STATION_RE.findall(answer)))
    targets = [s for s in present if s in ctx_st]
    alts = sorted(u for u in universe if u not in ctx_st and u not in present)
    if not targets or not alts:
        return None
    return re.sub(rf"\b{re.escape(targets[0])}\b", alts[len(answer) % len(alts)], answer)


def _foreign_contexts(g: dict, pool: list[dict]) -> list[str] | None:
    """Contexto de outra pergunta, sem nenhum episódio em comum e sem as estações da resposta."""
    mine = set(g.get("retrieved_ids", []))
    my_st = set(R._STATION_RE.findall(g["answer"]))
    cands = sorted((o for o in pool if o["id"] != g["id"] and o.get("retrieved_contexts")
                    and not (mine & set(o.get("retrieved_ids", [])))
                    and not (my_st & set(R._STATION_RE.findall(" ".join(o["retrieved_contexts"]))))),
                   key=lambda o: o["id"])
    return cands[len(g["answer"]) % len(cands)]["retrieved_contexts"] if cands else None


def cmd_perturb(n: int, n_control: int, summary_only: bool) -> None:
    if not KEY.exists():
        sys.exit("Rode 'export' antes (a amostra de respostas vem do gabarito).")
    done = {r["key"]: r for r in R._read_jsonl(PERTURB_OUT)}
    if not summary_only:
        key = _read_csv(KEY)
        lookups: dict[str, dict] = {}
        universe: set[str] = set()
        for _lbl, d in _run_dirs():
            for g in R._read_jsonl(d / "generations.jsonl"):
                for c in g.get("retrieved_contexts", []):
                    universe |= set(R._STATION_RE.findall(c))
        cands = []
        for r in sorted(key, key=lambda r: float(r["faithfulness_juiz"])):
            lk = lookups.setdefault(r["pasta"], _gen_lookup(r["pasta"]))
            g = lk[(r["id"], r["system"])]
            ctxs, q, a = g["retrieved_contexts"], g["question"], g["answer"]
            if _perturb_numbers(a, q, ctxs) and _perturb_station(a, q, ctxs, universe):
                cands.append((r, g, lk))
        pick = sorted({int(round(i)) for i in np.linspace(0, len(cands) - 1, min(n, len(cands)))}) if cands else []
        jobs = []  # (chave, amostra, condição, original, g_modificado)
        for i in pick:
            r, g, lk = cands[i]
            pool = [o for o in lk.values() if o["system"].startswith("rag_")]
            ctxs, q, a = g["retrieved_contexts"], g["question"], g["answer"]
            variants = {"neutra": _perturb_neutral(a), "numero": _perturb_numbers(a, q, ctxs),
                        "estacao": _perturb_station(a, q, ctxs, universe)}
            variants = {c: dict(g, answer=t) for c, t in variants.items() if t}
            fc = _foreign_contexts(g, pool)
            if fc:
                variants["contexto"] = dict(g, retrieved_contexts=fc)
            for cond, gm in variants.items():
                jobs.append((f"{r['amostra']}|{cond}", r, cond, float(r["faithfulness_juiz"]), gm))
        # controle positivo: respostas de REFERÊNCIA do golden set (corretas por construção) com o contexto
        # recuperado, quando todos os fatos da referência estão no contexto (conferido deterministicamente)
        ctrl = []
        for g in sorted(R._read_jsonl(R.BASE_RESULTS / "generations.jsonl"), key=lambda o: (o["id"], o["system"])):
            if not g["system"].startswith("rag_") or not g.get("retrieved_contexts"):
                continue
            if R.key_fact_recall(" ".join(g["retrieved_contexts"]), g["reference"], g["question"]) == 1.0:
                ctrl.append(g)
        for g in ctrl[:n_control]:
            jobs.append((f"REF|{g['id']}|{g['system']}", {"amostra": f"REF:{g['id']}", "id": g["id"],
                                                          "system": g["system"]},
                         "referencia", None, dict(g, answer=g["reference"])))
        todo = [j for j in jobs if j[0] not in done]
        logger.info("%d medições previstas, %d já feitas, %d a fazer (~1 min cada)", len(jobs), len(jobs) - len(todo),
                    len(todo))
        if todo:
            llm, emb = R._build_judge()
            for n_done, (k, r, cond, orig, gm) in enumerate(todo, start=1):
                t0 = time.perf_counter()
                v = _faithfulness_once(llm, emb, gm)
                row = {"key": k, "amostra": r["amostra"], "id": r["id"], "system": r["system"], "cond": cond,
                       "original": orig, "perturbada": v}
                R._append_jsonl(PERTURB_OUT, row)
                done[k] = row
                logger.info("[%d/%d] %s %s: original=%s perturbada=%s (%.0fs)", n_done, len(todo), r["amostra"], cond,
                            orig if orig is None else round(orig, 3), None if v is None else round(v, 3),
                            time.perf_counter() - t0)
    _perturb_report(list(done.values()))


def _perturb_report(rows: list[dict]) -> None:
    summary: dict = {"judge": R.JUDGE_MODEL}
    by = {c: [r for r in rows if r["cond"] == c and r["perturbada"] is not None and r["original"] is not None]
          for c in ("neutra", "numero", "estacao", "contexto")}
    neutral = np.array([r["perturbada"] - r["original"] for r in by["neutra"]])
    tol = max(0.05, float(np.percentile(np.abs(neutral), 90))) if len(neutral) else 0.05
    print(f"\n== Perturbações com resposta conhecida (juiz {R.JUDGE_MODEL}) ==")
    print(f"Tolerância de 'mudança real' = p90 do |Δ| do controle neutro (mín. 0,05) = {tol:.2f}")
    summary["tolerancia"] = tol
    for cond in ("neutra", "numero", "estacao", "contexto"):
        rs = by[cond]
        if not rs:
            continue
        o = np.array([r["original"] for r in rs])
        p = np.array([r["perturbada"] for r in rs])
        d = p - o
        mean_d = _boot(lambda a, b: float(np.mean(b - a)), o, p)
        detect = _boot(lambda a, b: float(np.mean((b - a) < -tol)), o, p)
        label = "CONTROLE neutro (reordenar seções; NÃO deve mudar)" if cond == "neutra" else _UNSUPPORTED_COND[cond]
        print(f"\n[{cond}] {label}  (n={len(rs)})")
        print(f"  nota original média {o.mean():.2f} -> perturbada {p.mean():.2f} | Δ médio {_fmt(mean_d)}")
        print(f"  caiu: {int((d < -tol).sum())} | ficou igual (±{tol:.2f}): {int((np.abs(d) <= tol).sum())} | "
              f"subiu: {int((d > tol).sum())}"
              + (f" | taxa de detecção {_fmt(detect)}" if cond != "neutra" else ""))
        summary[cond] = {"n": len(rs), "media_original": float(o.mean()), "media_perturbada": float(p.mean()),
                         "delta_medio": mean_d, "caiu": int((d < -tol).sum()), "igual": int((np.abs(d) <= tol).sum()),
                         "subiu": int((d > tol).sum()), "taxa_deteccao": detect if cond != "neutra" else None}
    refs = [r["perturbada"] for r in rows if r["cond"] == "referencia" and r["perturbada"] is not None]
    if refs:
        arr = np.array(refs)
        print(f"\n[referencia] controle POSITIVO: respostas de referência (corretas) com o contexto recuperado "
              f"(n={len(refs)})")
        print(f"  faithfulness média {arr.mean():.2f} | ≥0,8: {int((arr >= 0.8).sum())}/{len(arr)} | "
              f"mín {arr.min():.2f} | máx {arr.max():.2f}")
        summary["referencia"] = {"n": len(refs), "media": float(arr.mean()), "pct_ge_0_8": float((arr >= 0.8).mean()),
                                 "min": float(arr.min()), "max": float(arr.max())}
    PERTURB_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSalvo em {PERTURB_SUMMARY}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    for noisy in ("httpx", "urllib3", "pymilvus", "rag.retriever"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["export", "subset", "noise", "minicheck", "perturb", "score"])
    ap.add_argument("--n-label", type=int, default=15, help="subset: nº de linhas a rotular")
    ap.add_argument("--sheet", type=Path, default=None, help="score: planilha rotulada (padrão: a planilha humana)")
    ap.add_argument("--annotator", default="humano", help="score: quem rotulou (ex.: 'ia_claude'); NÃO use 'humano' para IA")
    ap.add_argument("--n", type=int, default=16, help="perturb: nº de respostas a perturbar")
    ap.add_argument("--n-control", type=int, default=12, help="perturb: nº de referências (controle positivo)")
    ap.add_argument("--summary-only", action="store_true", help="perturb: só reimprime o relatório")
    ap.add_argument("--judge", default=R.DEFAULT_JUDGE, help="juiz do RAGAS cujas notas são validadas")
    ap.add_argument("--n-per-cell", type=int, default=8, help="export: amostras por (gerador x sistema)")
    ap.add_argument("--force", action="store_true", help="export: sobrescreve a planilha (APAGA rótulos)")
    ap.add_argument("--k", type=int, default=10, help="noise: nº de respostas a reavaliar")
    ap.add_argument("--repeats", type=int, default=2, help="noise: repetições por resposta")
    ap.add_argument("--limit", type=int, default=None, help="minicheck: só as N primeiras amostras (teste)")
    args = ap.parse_args()
    R._configure(R.LEGACY_MODEL, args.judge)
    {"export": lambda: cmd_export(args.n_per_cell, args.force),
     "noise": lambda: cmd_noise(args.k, args.repeats),
     "subset": lambda: cmd_subset(args.n_label),
     "minicheck": lambda: cmd_minicheck(args.limit),
     "perturb": lambda: cmd_perturb(args.n, args.n_control, args.summary_only),
     "score": lambda: cmd_score(args.sheet, args.annotator)}[args.stage]()


if __name__ == "__main__":
    main()
