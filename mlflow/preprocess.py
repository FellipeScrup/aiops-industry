"""Sprint 4 — Silver → ML: preprocessing pipeline.

Reads raw logs from PostgreSQL, applies severity labeling and feature
engineering, saves a ready-to-train Parquet to data/silver/ and writes
a human-readable preprocessing report.
"""

import logging
import os
import re
from pathlib import Path

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

# ── Config ────────────────────────────────────────────────────────────────────

POSTGRES_USER: str = os.getenv("POSTGRES_USER", "aiops")
POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "aiops")
POSTGRES_DB: str = os.getenv("POSTGRES_DB", "aiops_industry")
POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")

OUTPUT_PARQUET = Path("data/silver/processed_logs.parquet")
REPORT_PATH = Path("mlflow/reports/preprocessing_report.txt")

LABEL_MAP: dict[str, int] = {"info": 0, "warning": 1, "critical": 2}
LABEL_NAMES: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}

FEATURES: list[str] = [
    "alarm_code_num",
    "machine_num",
    "source_num",
    "hour_of_day",
    "day_of_week",
    "alarm_frequency",
    "duration_ms",
]

_NUM_RE = re.compile(r"\d+")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_num(value: str) -> int:
    m = _NUM_RE.search(str(value))
    return int(m.group()) if m else 0


def _build_engine() -> Engine:
    url = (
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    return create_engine(url)


# ── Etapa 1 — Carregar dados ──────────────────────────────────────────────────

def load_data(engine: Engine) -> pd.DataFrame:
    logger.info("Carregando dados do PostgreSQL...")
    query = text("""
        SELECT
            source,
            machine_id,
            alarm_code,
            event_type,
            duration_ms,
            EXTRACT(HOUR FROM raw_timestamp) AS hour_of_day,
            EXTRACT(DOW  FROM raw_timestamp) AS day_of_week
        FROM logs
    """)
    df = pd.read_sql(query, engine)
    logger.info("  %d linhas carregadas.", len(df))
    return df


# ── Etapa 2 — Labels de severidade ───────────────────────────────────────────

def assign_severity(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculando frequência de alarm_code e atribuindo labels...")

    freq = df["alarm_code"].value_counts()
    df = df.copy()
    df["alarm_frequency"] = df["alarm_code"].map(freq)

    # np.select avalia as condições em ordem — a primeira que bate vence.
    conditions = [
        (df["source"] == "PIADE") & (df["event_type"] == "downtime"),
        (df["source"] == "PIADE") & (df["event_type"] == "performance_loss"),
        (df["source"] == "PIADE") & (df["event_type"] == "scheduled_downtime"),
        (df["source"] == "PIADE") & (df["event_type"] == "idle"),
        (df["source"] == "PIADE") & (df["event_type"] == "production"),
        df["alarm_frequency"] < 500,
        df["alarm_frequency"] > 10_000,
    ]
    choices = ["critical", "warning", "info", "info", "info", "critical", "warning"]

    df["severity"] = np.select(conditions, choices, default="info")
    df["severity_num"] = df["severity"].map(LABEL_MAP)

    dist = df["severity"].value_counts().to_dict()
    logger.info("  Distribuição de labels: %s", dist)
    return df


# ── Etapa 3 — Feature engineering ────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Construindo features...")
    df = df.copy()
    df["alarm_code_num"] = df["alarm_code"].apply(_extract_num)
    df["machine_num"] = df["machine_id"].apply(_extract_num)
    df["source_num"] = df["source"].map({"ALPI": 0, "PIADE": 1}).fillna(0).astype(int)
    df["duration_ms"] = df["duration_ms"].fillna(0).astype(float)
    df["hour_of_day"] = df["hour_of_day"].astype(int)
    df["day_of_week"] = df["day_of_week"].astype(int)
    logger.info("  Features prontas: %s", FEATURES)
    return df


# ── Etapa 4 — Estatísticas e relatório ───────────────────────────────────────

def compute_report(df: pd.DataFrame) -> str:
    total = len(df)
    lines: list[str] = []

    lines.append("=" * 60)
    lines.append("  PREPROCESSING REPORT — AIOps Industry (Sprint 4)")
    lines.append("=" * 60)
    lines.append(f"\nTotal de linhas: {total:,}")

    lines.append("\n--- Distribuição de Labels ---")
    label_counts = df["severity_num"].value_counts().sort_index()
    for num, count in label_counts.items():
        name = LABEL_NAMES.get(int(num), str(num))
        pct = 100.0 * count / total
        lines.append(f"  {name:<10} (label={num}):  {count:>7,}  ({pct:5.1f}%)")

    lines.append("\n--- Estatísticas por Feature ---")
    header = f"  {'Feature':<18} {'min':>10} {'max':>12} {'mean':>12} {'nulls':>8}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for feat in FEATURES:
        col = df[feat]
        null_count = int(col.isna().sum())
        lines.append(
            f"  {feat:<18} {col.min():>10.2f} {col.max():>12.2f}"
            f" {col.mean():>12.4f} {null_count:>8}"
        )

    lines.append("\n--- Fontes ---")
    for src, cnt in df["source"].value_counts().items():
        lines.append(f"  {src}: {cnt:,} linhas")

    lines.append("")
    return "\n".join(lines)


def save_report(report: str) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    logger.info("Relatório salvo em %s", REPORT_PATH)


# ── Etapa 5 — Salvar Parquet ──────────────────────────────────────────────────

def save_parquet(df: pd.DataFrame) -> None:
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    cols = FEATURES + ["severity", "severity_num"]
    df[cols].to_parquet(OUTPUT_PARQUET, index=False)
    size_mb = OUTPUT_PARQUET.stat().st_size / 1_048_576
    logger.info(
        "Dataset salvo em %s (%d linhas, %.1f MB)",
        OUTPUT_PARQUET, len(df), size_mb,
    )


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        engine = _build_engine()
        df = load_data(engine)
        df = assign_severity(df)
        df = engineer_features(df)
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
