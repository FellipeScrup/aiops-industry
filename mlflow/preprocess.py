"""Silver → ML: preprocessing pipeline for Smart Factory log events.

Reads events from PostgreSQL (smartfactory_logs, split='train'), derives lag
features (t-1 → t) per station for anomaly detection, and saves a
ready-to-train Parquet to data/silver/.

Lag design prevents data leakage: all input features come from the previous
event in the same station, while the label is the *current* state.
"""

import logging
import os
from pathlib import Path

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

# ── Config ────────────────────────────────────────────────────────────────────

POSTGRES_USER: str = os.getenv("POSTGRES_USER", "aiops")
POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "aiops")
POSTGRES_DB: str = os.getenv("POSTGRES_DB", "aiops_industry")
POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")

OUTPUT_PARQUET = Path("data/silver/processed_smartfactory.parquet")
REPORT_PATH = Path("mlflow/reports/preprocessing_report.txt")

STATION_MAP: dict[str, int] = {
    "MM_1": 0, "EC_1": 1, "SM_1": 2,
    "HBW_1": 3, "OV_1": 4, "VGR_1": 5, "WT_1": 6,
}

# Features usadas no treinamento (todas de t-1, sem leakage)
FEATURES: list[str] = [
    "prev_task_duration",
    "prev_is_moving",
    "prev_is_calibrating",
    "prev_has_task",
    "prev_was_not_ready",
    "time_delta_s",
    "station_num",
]

LABEL: str = "anomaly"


def _build_engine() -> Engine:
    url = (
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    return create_engine(url)


# ── Etapa 1 — Carregar dados ──────────────────────────────────────────────────

def load_data(engine: Engine) -> pd.DataFrame:
    logger.info("Carregando dados de smartfactory_logs (split='train')...")
    query = text("""
        SELECT
            station,
            event_timestamp,
            current_state,
            current_task,
            current_task_duration,
            current_sub_task
        FROM smartfactory_logs
        WHERE split = 'train'
        ORDER BY station, event_timestamp
    """)
    df = pd.read_sql(query, engine, parse_dates=["event_timestamp"])
    logger.info("  %d eventos carregados.", len(df))
    return df


# ── Etapa 2 — Feature engineering com lag ────────────────────────────────────

def _compute_instant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features no instante t (usadas como lag em t+1)."""
    task_str = df["current_task"].fillna("")
    subtask_str = df["current_sub_task"].fillna("")

    df["task_duration"] = pd.to_numeric(df["current_task_duration"], errors="coerce").fillna(0.0)
    df["is_moving"] = task_str.str.contains("transporting|moving", case=False, regex=True).astype(float)
    df["is_calibrating"] = subtask_str.str.contains("calibrating", case=False).astype(float)
    df["has_task"] = (task_str.str.strip() != "").astype(float)
    df["is_not_ready"] = (df["current_state"] == "not ready").astype(float)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Construindo features com lag-1 por estação...")
    df = df.copy()
    df = _compute_instant_features(df)

    df["station_num"] = df["station"].map(STATION_MAP).fillna(-1).astype(int)

    instant_cols = ["task_duration", "is_moving", "is_calibrating", "has_task", "is_not_ready"]

    lagged_frames: list[pd.DataFrame] = []

    for station, grp in df.groupby("station", sort=False):
        grp = grp.sort_values("event_timestamp").copy()

        # Lag-1 das features de estado
        for col in instant_cols:
            grp[f"prev_{col.replace('is_not_ready', 'was_not_ready')}"] = grp[col].shift(1)

        # Renomear para manter consistência
        grp = grp.rename(columns={
            "prev_task_duration":  "prev_task_duration",
            "prev_is_moving":      "prev_is_moving",
            "prev_is_calibrating": "prev_is_calibrating",
            "prev_has_task":       "prev_has_task",
        })

        # Diferença temporal em segundos
        grp["time_delta_s"] = (
            grp["event_timestamp"].diff().dt.total_seconds().fillna(0.0)
        )

        # Remover primeira linha de cada estação (sem t-1 disponível)
        grp = grp.iloc[1:]

        lagged_frames.append(grp)

    df_lag = pd.concat(lagged_frames, ignore_index=True)

    # Label no instante t
    df_lag[LABEL] = df_lag["is_not_ready"].astype(int)

    dist = df_lag[LABEL].value_counts().sort_index().to_dict()
    logger.info(
        "  Distribuição anomaly: normal=%d | anomalous=%d",
        dist.get(0, 0), dist.get(1, 0),
    )
    logger.info("  Dataset final: %d eventos (após drop da primeira linha por estação).", len(df_lag))
    return df_lag


# ── Etapa 3 — Relatório ───────────────────────────────────────────────────────

def compute_report(df: pd.DataFrame) -> str:
    total = len(df)
    lines: list[str] = []

    lines.append("=" * 60)
    lines.append("  PREPROCESSING REPORT — AIOps Industry (Smart Factory Logs)")
    lines.append("  Estratégia: Lag-1 features (t-1 → prediz t)")
    lines.append("=" * 60)
    lines.append(f"\nTotal de eventos (train, após lag): {total:,}")

    lines.append("\n--- Distribuição de Labels ---")
    label_counts = df[LABEL].value_counts().sort_index()
    for num, count in label_counts.items():
        name = "ready" if num == 0 else "not ready"
        pct = 100.0 * count / total
        lines.append(f"  {name:<12} (anomaly={num}):  {count:>7,}  ({pct:5.1f}%)")

    lines.append("\n--- Estatísticas por Feature (lag) ---")
    header = f"  {'Feature':<22} {'min':>8} {'max':>10} {'mean':>10} {'nulls':>8}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for feat in FEATURES:
        col = df[feat]
        null_count = int(col.isna().sum())
        lines.append(
            f"  {feat:<22} {col.min():>8.4f} {col.max():>10.4f}"
            f" {col.mean():>10.4f} {null_count:>8}"
        )

    lines.append("")
    return "\n".join(lines)


def save_report(report: str) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    logger.info("Relatório salvo em %s", REPORT_PATH)


# ── Etapa 4 — Salvar Parquet ──────────────────────────────────────────────────

def save_parquet(df: pd.DataFrame) -> None:
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    cols = FEATURES + [LABEL]
    df[cols].to_parquet(OUTPUT_PARQUET, index=False)
    size_mb = OUTPUT_PARQUET.stat().st_size / 1_048_576
    logger.info(
        "Dataset salvo em %s (%d eventos, %.1f MB)",
        OUTPUT_PARQUET, len(df), size_mb,
    )


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        engine = _build_engine()
        df = load_data(engine)
        df = build_features(df)
        report = compute_report(df)
        save_report(report)
        save_parquet(df)
    except Exception:
        logger.exception("Pré-processamento falhou")
        raise

    print("Pré-processamento concluído.")
    print(f"  Parquet: {OUTPUT_PARQUET}")
    print(f"  Relatório: {REPORT_PATH}")


if __name__ == "__main__":
    main()
