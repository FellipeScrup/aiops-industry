"""Golden set objetivo para avaliar o RAG sobre os episódios da Smart Factory.

Diferente de um golden set gerado por LLM, aqui TODA referência e TODA evidência
vêm de dados verificáveis:

  - fatos de episódio (estação, tarefa, sub-tarefa, duração, início)  → tabela
    smartfactory_episodes (Silver → Gold);
  - diagnóstico de anomalia (duração, mediana, limiar)               → mesma tabela
    (regra mediana + 3·MAD calculada em parse_episodes.py);
  - impacto no processo (atividade BPM, próxima estação)             → tabela BPM
    do projeto (ingestion/bpm_context.py).

Três tiers, TIER_SIZE perguntas cada, seed fixa (reprodutível):

  factual        recuperar UM episódio: metade por estação+horário, metade por
                 descrição semântica (sem horário);
  cross_station  recuperar DOIS episódios de estações diferentes quase simultâneos;
  causal         diagnosticar se uma calibração foi anormalmente longa (casos
                 positivos E negativos) e identificar o impacto no processo (BPM).

Cada item registra `evidence_episode_ids`: os episódios que o retriever precisa
trazer para a pergunta ser respondível (usado nas métricas Hit@k / MRR).
`evidence_mode` = "all" (todos são necessários) ou "any" (qualquer um basta).

Uso:
  python evaluation/golden_set.py
"""

import json
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ingestion.bpm_context import get_bpm_context  # noqa: E402

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEED: int = 42
TIER_SIZE: int = 8
OUTPUT_PATH: Path = Path(__file__).resolve().parent / "golden_set.json"

# Meia-janela do filtro temporal do retriever híbrido (rag/query_parser.py).
WINDOW_S: float = 5.0

POSTGRES_USER: str = os.getenv("POSTGRES_USER", "aiops")
POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "aiops")
POSTGRES_DB: str = os.getenv("POSTGRES_DB", "aiops_industry")
POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")

# Próximas estações que não são "estação" (sem informação ou sem continuidade).
_NO_NEXT = {"-", "idle", ""}


def _load_episodes() -> pd.DataFrame:
    engine = create_engine(
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    df = pd.read_sql(text("""
        SELECT episode_id, station, current_task, current_sub_task, start_ts,
               duration_s, is_anomaly, subtask_median_s, subtask_threshold_s
        FROM smartfactory_episodes
        WHERE current_state = 'not ready'
        ORDER BY start_ts
    """), engine)
    df["ts_sec"] = df["start_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    logger.info("%d episódios not-ready carregados (indexados no Milvus).", len(df))
    return df


def _fact(row: pd.Series) -> str:
    return (
        f"A estação {row.station} ficou not ready por {row.duration_s:.1f}s "
        f"executando '{row.current_task}', sub-tarefa '{row.current_sub_task}'."
    )


def _unique_in_second(df: pd.DataFrame) -> pd.DataFrame:
    """Episódios sem outro da mesma estação iniciando no mesmo segundo (pergunta não ambígua)."""
    counts = df.groupby(["station", "ts_sec"])["episode_id"].transform("size")
    return df[counts == 1]


def _unique_in_window(df: pd.DataFrame, station: str, center: pd.Timestamp) -> bool:
    """Exatamente um episódio da estação inicia dentro da janela do filtro temporal."""
    near = df[(df.station == station)
              & ((df.start_ts - center).abs() <= pd.Timedelta(seconds=WINDOW_S))]
    return len(near) == 1


# ── Tier factual ────────────────────────────────────────────────────────────

def _factual(df: pd.DataFrame, rng: random.Random) -> list[dict]:
    items: list[dict] = []
    # A pergunta "estação X ficou not ready em T" só é inequívoca se nenhum outro episódio da
    # mesma estação começar na janela ±WINDOW_S de T (senão há mais de uma resposta correta).
    base = df[~df.current_task.str.contains("calibrat")]
    candidates = base[[_unique_in_window(df, r.station, r.start_ts) for r in base.itertuples()]]

    # (a) estação + horário — metade do tier
    by_station = {s: g for s, g in candidates.groupby("station")}
    stations = [s for s in ("HBW_1", "VGR_1", "OV_1", "MM_1", "SM_1") if s in by_station]
    for i in range(TIER_SIZE // 2):
        st = stations[i % len(stations)]
        row = by_station[st].sample(1, random_state=rng.randint(0, 10**6)).iloc[0]
        items.append({
            "subtype": "estacao_horario",
            "question": (
                f"A estação {row.station} ficou not ready em {row.ts_sec}. "
                f"Qual tarefa e sub-tarefa ela estava executando e quanto tempo durou?"
            ),
            "reference": _fact(row),
            "evidence_episode_ids": [row.episode_id],
            "evidence_mode": "all",
            "meta": {"stations": [row.station], "timestamp": row.ts_sec},
        })
        by_station[st] = by_station[st].drop(row.name)

    # (b) descrição semântica, sem horário — combinação (estação, tarefa, sub-tarefa) única
    combo_n = df.groupby(["station", "current_task", "current_sub_task"])["episode_id"].transform("size")
    already = {i for it in items for i in it["evidence_episode_ids"]}  # não reutilizar episódios do item (a)
    unique_combo = df[(combo_n == 1) & ~df.current_task.str.contains("calibrat")
                      & ~df.episode_id.isin(already)]
    by_station_u = {s: g for s, g in unique_combo.groupby("station")}
    order = [s for s in ("HBW_1", "OV_1", "VGR_1", "MM_1", "SM_1") if s in by_station_u]
    for i in range(TIER_SIZE - len(items)):
        st = order[i % len(order)]
        row = by_station_u[st].sample(1, random_state=rng.randint(0, 10**6)).iloc[0]
        items.append({
            "subtype": "descricao_semantica",
            "question": (
                f"Qual foi a duração do episódio em que a estação {row.station} executou "
                f"'{row.current_task}' na sub-tarefa '{row.current_sub_task}'?"
            ),
            "reference": _fact(row),
            "evidence_episode_ids": [row.episode_id],
            "evidence_mode": "all",
            "meta": {"stations": [row.station], "timestamp": None},
        })
        by_station_u[st] = by_station_u[st].drop(row.name)
    return items


# ── Tier cross_station ──────────────────────────────────────────────────────

def _cross_station(df: pd.DataFrame, rng: random.Random) -> list[dict]:
    pairs: dict[tuple[str, str], list[tuple[pd.Series, pd.Series]]] = defaultdict(list)
    uniq = _unique_in_second(df)
    for _, a in uniq.iterrows():
        near = uniq[(uniq.station != a.station)
                    & (uniq.start_ts >= a.start_ts)
                    & ((uniq.start_ts - a.start_ts) <= pd.Timedelta(seconds=WINDOW_S - 1))]
        for _, b in near.iterrows():
            # ambos os episódios precisam ser os únicos da respectiva estação na janela
            if _unique_in_window(df, a.station, a.start_ts) and _unique_in_window(df, b.station, a.start_ts):
                pairs[tuple(sorted((a.station, b.station)))].append((a, b))

    quota = [("HBW_1", "VGR_1")] * 4 + [("MM_1", "SM_1")] * 2 + [("SM_1", "VGR_1"), ("HBW_1", "OV_1")]
    items: list[dict] = []
    used: set[str] = set()
    fallback = sorted(pairs, key=lambda k: -len(pairs[k]))  # demais pares, mais frequentes primeiro
    for key in quota + fallback * TIER_SIZE:
        if len(items) >= TIER_SIZE:
            break
        pool = [p for p in pairs.get(key, []) if p[0].episode_id not in used and p[1].episode_id not in used]
        if not pool:
            continue
        a, b = pool[rng.randrange(len(pool))]
        used.update({a.episode_id, b.episode_id})
        items.append({
            "subtype": "duas_estacoes",
            "question": (
                f"Por volta de {a.ts_sec}, as estações {a.station} e {b.station} ficaram "
                f"not ready quase ao mesmo tempo. O que cada uma estava fazendo e por quanto tempo?"
            ),
            "reference": f"{_fact(a)} {_fact(b)}",
            "evidence_episode_ids": [a.episode_id, b.episode_id],
            "evidence_mode": "all",
            "meta": {"stations": [a.station, b.station], "timestamp": a.ts_sec},
        })
    return items


# ── Tier causal ─────────────────────────────────────────────────────────────

def _causal(df: pd.DataFrame, rng: random.Random) -> list[dict]:
    items: list[dict] = []

    # (a) diagnóstico de anomalia de duração (3 positivos + 2 negativos)
    # Apenas "calibrating motor 4": sub-tarefa exclusiva da HBW_1. As demais (motor 2, por ex.) são
    # compartilhadas entre estações com durações típicas diferentes e a regra de anomalia agrupa por
    # sub-tarefa (não por estação+sub-tarefa), o que contaminaria o rótulo — ver ameaças à validade.
    calib = _unique_in_second(df[df.current_sub_task == "calibrating motor 4"])
    positives = calib[calib.is_anomaly].sample(3, random_state=rng.randint(0, 10**6))
    # negativos "limpos": duração próxima da mediana (evita casos 4x mais curtos que o típico)
    near_median = calib[~calib.is_anomaly
                        & (calib.duration_s >= 0.8 * calib.subtask_median_s)
                        & (calib.duration_s <= 1.25 * calib.subtask_median_s)]
    negatives = near_median.sample(2, random_state=rng.randint(0, 10**6))
    for _, row in pd.concat([positives, negatives]).iterrows():
        motor = row.current_sub_task.replace("calibrating ", "")
        if row.is_anomaly:
            excess = row.duration_s - row.subtask_median_s
            ref = (
                f"Sim. A calibração do {motor} durou {row.duration_s:.1f}s, {excess:.1f}s acima da "
                f"mediana de {row.subtask_median_s:.1f}s para essa sub-tarefa (limiar {row.subtask_threshold_s:.1f}s): "
                f"duração anômala."
            )
        else:
            ref = (
                f"Não. A calibração do {motor} durou {row.duration_s:.1f}s, dentro do normal "
                f"(mediana de {row.subtask_median_s:.1f}s para essa sub-tarefa)."
            )
        items.append({
            "subtype": "anomalia_sim" if row.is_anomaly else "anomalia_nao",
            "question": (
                f"A calibração do {motor} da estação {row.station} iniciada em {row.ts_sec} "
                f"foi anormalmente longa em relação ao tempo típico?"
            ),
            "reference": ref,
            "evidence_episode_ids": [row.episode_id],
            "evidence_mode": "all",
            "meta": {"stations": [row.station], "timestamp": row.ts_sec, "is_anomaly": bool(row.is_anomaly)},
        })

    # (b) impacto no processo (BPM) — só atividades com próxima estação concreta
    rows = []
    for (st, task), g in df.groupby(["station", "current_task"]):
        ctx = get_bpm_context(st, task)
        if not ctx or ctx["next_station"] in _NO_NEXT or ctx["next_station"] == st:
            continue
        rows.append((st, task, ctx, list(g.episode_id)))
    rng.shuffle(rows)
    seen_stations: set[str] = set()
    for st, task, ctx, ids in rows:
        if st in seen_stations:
            continue
        seen_stations.add(st)
        items.append({
            "subtype": "impacto_bpm",
            "question": (
                f"Se a estação {st} ficar parada enquanto executa '{task}', qual atividade do "
                f"processo é interrompida e qual é a próxima estação da linha?"
            ),
            "reference": (
                f"A atividade interrompida é '{ctx['activity']}' (processo {ctx['process']}); "
                f"a próxima estação do processo é {ctx['next_station']}."
            ),
            "evidence_episode_ids": ids,
            "evidence_mode": "any",
            "meta": {"stations": [st], "timestamp": None},
        })
        if len(items) >= TIER_SIZE:
            break
    return items


def build() -> list[dict]:
    df = _load_episodes()
    rng = random.Random(SEED)
    golden: list[dict] = []
    for tier, fn in (("factual", _factual), ("cross_station", _cross_station), ("causal", _causal)):
        items = fn(df, rng)
        if len(items) != TIER_SIZE:
            logger.warning("Tier %s gerou %d/%d itens.", tier, len(items), TIER_SIZE)
        for i, item in enumerate(items, start=1):
            golden.append({"id": f"{tier[:3]}-{i:02d}", "tier": tier, **item})
    return golden


def main() -> None:
    golden = build()
    OUTPUT_PATH.write_text(json.dumps(golden, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = pd.Series([g["tier"] + "/" + g["subtype"] for g in golden]).value_counts().sort_index()
    logger.info("Golden set com %d perguntas salvo em %s", len(golden), OUTPUT_PATH)
    for k, v in counts.items():
        logger.info("  %-34s %d", k, v)


if __name__ == "__main__":
    main()
