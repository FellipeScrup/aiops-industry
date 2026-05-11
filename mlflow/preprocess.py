"""Silver → ML: preprocessing pipeline for PIADE telemetry windows.

Reads 1h windows from PostgreSQL (piade_telemetry), derives degradation labels
from telemetry thresholds, and saves a ready-to-train Parquet to data/silver/.
"""

import logging
import os
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

OUTPUT_PARQUET = Path("data/silver/processed_telemetry.parquet")
REPORT_PATH = Path("mlflow/reports/preprocessing_report.txt")

LABEL_MAP: dict[str, int] = {"normal": 0, "degraded": 1, "critical": 2}
LABEL_NAMES: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}

FEATURES: list[str] = [
    "pct_idle",
    "pct_production",
    "pct_downtime",
    "pct_perf_loss",
    "pct_sched_downtime",
    "count_sum",
    "num_changes",
]


def _build_engine() -> Engine:
    url = (
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    return create_engine(url)


# ── Etapa 1 — Carregar dados ──────────────────────────────────────────────────

def load_data(engine: Engine) -> pd.DataFrame:
    logger.info("Carregando dados de piade_telemetry...")
    query = text("""
        SELECT
            equipment_id,
            interval_start,
            count_sum,
            num_changes,
            pct_idle,
            pct_production,
            pct_downtime,
            pct_perf_loss,
            pct_sched_downtime
        FROM piade_telemetry
    """)
    df = pd.read_sql(query, engine)
    logger.info("  %d janelas carregadas.", len(df))
    return df


# ── Etapa 2 — Labels de degradação ───────────────────────────────────────────

def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Derive a 3-class degradation label from telemetry thresholds.

    critical : downtime > 15% OR perf_loss > 20%
    degraded : downtime > 5%  OR perf_loss > 10%
    normal   : otherwise
    """
    logger.info("Atribuindo labels de degradação...")
    df = df.copy()

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    conditions = [
        (df["pct_downtime"] > 0.15) | (df["pct_perf_loss"] > 0.20),
        (df["pct_downtime"] > 0.05) | (df["pct_perf_loss"] > 0.10),
    ]
    choices = ["critical", "degraded"]
    df["label"] = np.select(conditions, choices, default="normal")
    df["label_num"] = df["label"].map(LABEL_MAP)

    dist = df["label"].value_counts().to_dict()
    logger.info("  Distribuição de labels: %s", dist)
    return df


# ── Etapa 3 — Relatório ───────────────────────────────────────────────────────

def compute_report(df: pd.DataFrame) -> str:
    total = len(df)
    lines: list[str] = []

    lines.append("=" * 60)
    lines.append("  PREPROCESSING REPORT — AIOps Industry (PIADE Telemetry)")
    lines.append("=" * 60)
    lines.append(f"\nTotal de janelas: {total:,}")

    lines.append("\n--- Distribuição de Labels ---")
    label_counts = df["label_num"].value_counts().sort_index()
    for num, count in label_counts.items():
        name = LABEL_NAMES.get(int(num), str(num))
        pct = 100.0 * count / total
        lines.append(f"  {name:<10} (label={num}):  {count:>7,}  ({pct:5.1f}%)")

    lines.append("\n--- Estatísticas por Feature ---")
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
    cols = FEATURES + ["label", "label_num"]
    df[cols].to_parquet(OUTPUT_PARQUET, index=False)
    size_mb = OUTPUT_PARQUET.stat().st_size / 1_048_576
    logger.info(
        "Dataset salvo em %s (%d janelas, %.1f MB)",
        OUTPUT_PARQUET, len(df), size_mb,
    )


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        engine = _build_engine()
        df = load_data(engine)
        df = assign_labels(df)
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
