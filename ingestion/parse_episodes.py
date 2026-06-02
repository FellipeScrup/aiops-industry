"""Silver → Gold: aggregates raw 10 Hz events into meaningful episodes.

The Silver table (smartfactory_logs) holds one row per 100 ms sample. A single
machine action lasting a few seconds therefore spans dozens of near-identical
rows, which is useless for diagnostic retrieval ("why did the station stop?").

This step collapses consecutive rows of the same
(station, current_task, current_sub_task, current_state) into a single episode
with a start, an end, a duration and a sensor delta, and flags duration
anomalies per sub-task. The result is persisted to the Gold table
smartfactory_episodes — the unit that actually carries meaning.

Episode break rule: a new episode starts whenever station, current_task,
current_sub_task OR current_state changes versus the previous row (rows read in
station, event_timestamp order). current_state is part of the key so every
episode has a single well-defined state.

Anomaly rule (per sub-task, computed over not-ready episodes only):
    duration > median + ANOMALY_MAD_K * MAD   AND   (duration - median) >= ANOMALY_MIN_EXCESS_S
The absolute floor avoids flagging millisecond deviations on very consistent
sub-tasks where MAD ≈ 0.
"""

import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────

POSTGRES_USER: str = os.getenv("POSTGRES_USER", "aiops")
POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "aiops")
POSTGRES_DB: str = os.getenv("POSTGRES_DB", "aiops_industry")
POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")

SOURCE_SPLIT: str = os.getenv("EPISODE_SPLIT", "train")

# Anomaly detection (per sub-task, over not-ready episodes)
ANOMALY_MAD_K: float = 3.0
ANOMALY_MIN_EXCESS_S: float = 2.0

# Episode of a single 10 Hz event (n_events=1) has start_ts==end_ts and duration 0s.
# These are transient samples without semantics — filter them out.
MIN_EPISODE_EVENTS: int = 2

EPISODES_TABLE: str = "smartfactory_episodes"

_DDL = text(f"""
CREATE TABLE IF NOT EXISTS {EPISODES_TABLE} (
    episode_id          VARCHAR(64) PRIMARY KEY,
    station             VARCHAR(20)  NOT NULL,
    current_state       VARCHAR(20)  NOT NULL,
    current_task        TEXT,
    current_sub_task    TEXT,
    start_ts            TIMESTAMP    NOT NULL,
    end_ts              TIMESTAMP    NOT NULL,
    duration_s          DOUBLE PRECISION NOT NULL,
    n_events            INTEGER      NOT NULL,
    is_anomaly          BOOLEAN      NOT NULL,
    subtask_median_s    DOUBLE PRECISION,
    subtask_threshold_s DOUBLE PRECISION,
    sensors_start       JSONB,
    sensors_end         JSONB,
    sensors_changed     JSONB,
    text                TEXT         NOT NULL
);
""")

_INSERT = text(f"""
INSERT INTO {EPISODES_TABLE}
    (episode_id, station, current_state, current_task, current_sub_task,
     start_ts, end_ts, duration_s, n_events, is_anomaly,
     subtask_median_s, subtask_threshold_s,
     sensors_start, sensors_end, sensors_changed, text)
VALUES
    (:episode_id, :station, :current_state, :current_task, :current_sub_task,
     :start_ts, :end_ts, :duration_s, :n_events, :is_anomaly,
     :subtask_median_s, :subtask_threshold_s,
     CAST(:sensors_start AS JSONB), CAST(:sensors_end AS JSONB),
     CAST(:sensors_changed AS JSONB), :text)
ON CONFLICT (episode_id) DO NOTHING
""")


# ── DB ──────────────────────────────────────────────────────────────────────

def _build_engine() -> Engine:
    url = (
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    return create_engine(url, pool_pre_ping=True)


def _load_silver(engine: Engine) -> pd.DataFrame:
    logger.info("Carregando Silver (split=%s) ordenado por station, timestamp...", SOURCE_SPLIT)
    query = text("""
        SELECT station, event_timestamp, current_state,
               current_task, current_sub_task, sensors
        FROM smartfactory_logs
        WHERE split = :split
        ORDER BY station, event_timestamp
    """)
    df = pd.read_sql(query, engine, params={"split": SOURCE_SPLIT})
    logger.info("  %d eventos carregados.", len(df))
    return df


# ── Helpers ─────────────────────────────────────────────────────────────────

def _as_dict(raw: Any) -> dict:
    """sensors column may come back as dict (psycopg2 JSONB) or str."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def _sensor_delta(start: dict, end: dict) -> dict[str, list]:
    """Keys whose value changed between the first and last event of the episode."""
    changed: dict[str, list] = {}
    for k in start.keys() | end.keys():
        a, b = start.get(k), end.get(k)
        if a != b:
            changed[k] = [a, b]
    return changed


def _episode_id(station: str, start: pd.Timestamp) -> str:
    return f"{station}__{start.strftime('%Y%m%d%H%M%S%f')}"


def _build_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """Run-length encode consecutive rows into episodes."""
    key_cols = ["station", "current_task", "current_sub_task", "current_state"]
    key = df[key_cols].fillna("")
    boundary = (key != key.shift()).any(axis=1)
    df = df.copy()
    df["_epi"] = boundary.cumsum()

    rows: list[dict] = []
    for _, grp in df.groupby("_epi", sort=True):
        first = grp.iloc[0]
        last = grp.iloc[-1]
        start = first["event_timestamp"]
        end = last["event_timestamp"]
        duration = (end - start).total_seconds()

        sensors_start = _as_dict(first["sensors"])
        sensors_end = _as_dict(last["sensors"])
        changed = _sensor_delta(sensors_start, sensors_end)

        rows.append({
            "episode_id":       _episode_id(str(first["station"]), start),
            "station":          first["station"],
            "current_state":    first["current_state"],
            "current_task":     first["current_task"] or "",
            "current_sub_task": first["current_sub_task"] or "",
            "start_ts":         start,
            "end_ts":           end,
            "duration_s":       round(float(duration), 3),
            "n_events":         int(len(grp)),
            "sensors_start":    sensors_start,
            "sensors_end":      sensors_end,
            "sensors_changed":  changed,
        })

    episodes = pd.DataFrame(rows)

    before = len(episodes)
    episodes = episodes[episodes["n_events"] >= MIN_EPISODE_EVENTS].reset_index(drop=True)
    dropped = before - len(episodes)
    if dropped:
        logger.info("Descartados %d episódios degenerados (n_events < %d).", dropped, MIN_EPISODE_EVENTS)

    logger.info(
        "%d episódios (not ready=%d, ready=%d).",
        len(episodes),
        int((episodes["current_state"] == "not ready").sum()),
        int((episodes["current_state"] == "ready").sum()),
    )
    return episodes


def _flag_anomalies(episodes: pd.DataFrame) -> pd.DataFrame:
    """Per sub-task duration anomaly over not-ready episodes (median + k·MAD, floor)."""
    episodes = episodes.copy()
    episodes["is_anomaly"] = False
    episodes["subtask_median_s"] = np.nan
    episodes["subtask_threshold_s"] = np.nan

    nr = episodes["current_state"] == "not ready"
    for subtask, grp in episodes[nr].groupby("current_sub_task"):
        durations = grp["duration_s"].to_numpy()
        median = float(np.median(durations))
        mad = float(np.median(np.abs(durations - median)))
        threshold = median + ANOMALY_MAD_K * mad

        episodes.loc[grp.index, "subtask_median_s"] = round(median, 3)
        episodes.loc[grp.index, "subtask_threshold_s"] = round(threshold, 3)

        is_anom = (grp["duration_s"] > threshold) & (
            grp["duration_s"] - median >= ANOMALY_MIN_EXCESS_S
        )
        episodes.loc[grp.index[is_anom], "is_anomaly"] = True

    logger.info("Anomalias de duração marcadas: %d", int(episodes["is_anomaly"].sum()))
    return episodes


def _build_text(ep: pd.Series) -> str:
    """Human-readable Portuguese description embedded into the vector store."""
    station = ep["station"]
    state = ep["current_state"]
    dur = ep["duration_s"]
    task = ep["current_task"] or "sem tarefa (ociosa)"
    subtask = ep["current_sub_task"] or "-"
    changed = list(ep["sensors_changed"].keys())
    changed_str = ", ".join(sorted(changed)) if changed else "nenhum"

    if state == "not ready":
        parts = [
            f"Estação {station} ficou not ready por {dur:.1f}s "
            f"executando '{task}', sub-tarefa '{subtask}'."
        ]
        median = ep["subtask_median_s"]
        if ep["is_anomaly"]:
            excess = dur - (median if pd.notna(median) else 0.0)
            parts.append(
                f"Duração ANÔMALA: {excess:.1f}s acima do normal para esta "
                f"sub-tarefa (mediana {median:.1f}s)."
            )
        elif pd.notna(median):
            parts.append(f"Duração dentro do normal (mediana da sub-tarefa: {median:.1f}s).")
        parts.append(f"Sensores que mudaram durante o episódio: {changed_str}.")
        return " ".join(parts)

    # ready / idle
    return (
        f"Estação {station} permaneceu {state} (ociosa) por {dur:.1f}s. "
        f"Sensores que mudaram durante o episódio: {changed_str}."
    )


# ── Persist ─────────────────────────────────────────────────────────────────

def _persist(engine: Engine, episodes: pd.DataFrame) -> None:
    logger.info("Criando tabela %s (se necessário) e regravando episódios...", EPISODES_TABLE)
    with engine.begin() as conn:
        conn.execute(_DDL)
        conn.execute(text(f"TRUNCATE TABLE {EPISODES_TABLE}"))

    records: list[dict] = []
    for _, ep in episodes.iterrows():
        records.append({
            "episode_id":          ep["episode_id"],
            "station":             ep["station"],
            "current_state":       ep["current_state"],
            "current_task":        ep["current_task"],
            "current_sub_task":    ep["current_sub_task"],
            "start_ts":            ep["start_ts"].to_pydatetime(),
            "end_ts":              ep["end_ts"].to_pydatetime(),
            "duration_s":          ep["duration_s"],
            "n_events":            ep["n_events"],
            "is_anomaly":          bool(ep["is_anomaly"]),
            "subtask_median_s":    None if pd.isna(ep["subtask_median_s"]) else ep["subtask_median_s"],
            "subtask_threshold_s": None if pd.isna(ep["subtask_threshold_s"]) else ep["subtask_threshold_s"],
            "sensors_start":       json.dumps(ep["sensors_start"]),
            "sensors_end":         json.dumps(ep["sensors_end"]),
            "sensors_changed":     json.dumps(ep["sensors_changed"]),
            "text":                ep["text"],
        })

    with engine.begin() as conn:
        conn.execute(_INSERT, records)

    logger.info("Persistidos %d episódios em %s.", len(records), EPISODES_TABLE)


# ── Entrypoint ──────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("Iniciando agregação Silver → Gold (episódios).")
    engine = _build_engine()

    df = _load_silver(engine)
    if df.empty:
        logger.warning("Nenhum evento no split '%s'. Abortando.", SOURCE_SPLIT)
        return

    episodes = _build_episodes(df)
    episodes = _flag_anomalies(episodes)
    episodes["text"] = episodes.apply(_build_text, axis=1)

    _persist(engine, episodes)

    n_anom = int(episodes["is_anomaly"].sum())
    n_nr = int((episodes["current_state"] == "not ready").sum())
    logger.info(
        "Concluído. Episódios=%d | not ready=%d | anomalias=%d",
        len(episodes), n_nr, n_anom,
    )


if __name__ == "__main__":
    main()
