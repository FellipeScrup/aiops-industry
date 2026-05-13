"""Anomaly Detection training on Smart Factory logs — aligned with paper metrics.

Binary XGBoost classifier:  anomalous (not ready) = 1  /  normal (ready) = 0

Metrics logged to MLflow:
  average_precision    area under Precision-Recall curve (AP score)
  roc_auc              area under ROC curve
  fit_time_s           training wall-clock time in seconds
  predict_time_ms      mean inference latency per sample in milliseconds

Artifacts logged to MLflow (artifact_path="plots"):
  pr_curve.png           Precision-Recall curve
  roc_curve.png          ROC curve
  feature_importance.png Global Feature Importance — horizontal bar chart (GFI)

Model saved locally: mlflow/models/xgboost_ad.json

Usage:
  python mlflow/train_ad.py
  make train-ad
"""

import logging
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

MLFLOW_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME: str = "smartfactory-anomaly-detection"
RUN_NAME: str = "xgboost-binary-ad"

PARQUET_PATH = Path("data/silver/processed_smartfactory.parquet")

FEATURES: list[str] = [
    "prev_task_duration",
    "prev_is_moving",
    "prev_is_calibrating",
    "prev_has_task",
    "prev_was_not_ready",
    "time_delta_s",
    "station_num",
]

FEATURE_LABELS: dict[str, str] = {
    "prev_task_duration":   "Prev Task Duration (s)",
    "prev_is_moving":       "Prev Is Moving",
    "prev_is_calibrating":  "Prev Is Calibrating",
    "prev_has_task":        "Prev Has Task",
    "prev_was_not_ready":   "Prev Was Not Ready",
    "time_delta_s":         "Time Delta (s)",
    "station_num":          "Station",
}

XGB_PARAMS: dict = {
    "n_estimators":  100,
    "max_depth":     6,
    "learning_rate": 0.1,
    "random_state":  42,
    "n_jobs":        -1,
    "eval_metric":   "logloss",
}

MODELS_DIR = Path(__file__).parent / "models"
PLOTS_DIR  = Path("/tmp/aiops_ad_plots")


# ── 1. Data ───────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load preprocessed smartfactory data from parquet."""
    logger.info("Carregando %s...", PARQUET_PATH)
    df = pd.read_parquet(PARQUET_PATH)
    logger.info("  %d eventos carregados.", len(df))

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    dist  = df["anomaly"].value_counts().sort_index().to_dict()
    n_tot = len(df)
    logger.info(
        "  Anomaly distribution: normal=%d (%.1f%%) | anomalous=%d (%.1f%%)",
        dist.get(0, 0), 100 * dist.get(0, 0) / n_tot,
        dist.get(1, 0), 100 * dist.get(1, 0) / n_tot,
    )
    return df


# ── 2. Train ──────────────────────────────────────────────────────────────────

def fit_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[XGBClassifier, float]:
    """Train XGBoost and return (model, fit_time_s)."""
    model = XGBClassifier(**XGB_PARAMS)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time_s = time.perf_counter() - t0
    logger.info("Fit time: %.4f s", fit_time_s)
    return model, fit_time_s


def predict_with_latency(
    model: XGBClassifier,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    """Run predict_proba and return (y_proba_positive, predict_time_ms_per_sample)."""
    t0 = time.perf_counter()
    y_proba = model.predict_proba(X_test)[:, 1]
    elapsed = time.perf_counter() - t0
    predict_time_ms = (elapsed / len(X_test)) * 1000
    logger.info("Predict time: %.6f ms/sample (%d samples)", predict_time_ms, len(X_test))
    return y_proba, predict_time_ms


# ── 3. Plots ──────────────────────────────────────────────────────────────────

def _pr_curve_plot(y_true: np.ndarray, y_proba: np.ndarray, path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    baseline = float(y_true.mean())

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, lw=2, color="#2980b9", label=f"XGBoost  AP={ap:.4f}")
    ax.axhline(baseline, color="#e74c3c", linestyle="--", lw=1.5,
               label=f"Baseline (random)  AP={baseline:.4f}")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — Smart Factory Anomaly Detection", fontsize=13)
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("PR Curve salva em %s", path)


def _roc_curve_plot(y_true: np.ndarray, y_proba: np.ndarray, path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, color="#27ae60", label=f"XGBoost  AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random (AUC=0.5000)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Smart Factory Anomaly Detection", fontsize=13)
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("ROC Curve salva em %s", path)


def _feature_importance_plot(model: XGBClassifier, path: Path) -> None:
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)

    labels = [FEATURE_LABELS[FEATURES[i]] for i in sorted_idx]
    values = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, values, color="#8e44ad", edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
    ax.set_xlabel("Importance (F-score weight)", fontsize=12)
    ax.set_title("Global Feature Importance — XGBoost AD (GFI)", fontsize=13)
    ax.set_xlim([0, max(values) * 1.2])
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Feature Importance salva em %s", path)


# ── 4. Save model locally ─────────────────────────────────────────────────────

def save_model_local(model: XGBClassifier) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / "xgboost_ad.json"
    model.save_model(str(dest))
    logger.info("Modelo salvo localmente em %s", dest)


# ── 5. MLflow ─────────────────────────────────────────────────────────────────

def log_to_mlflow(
    model: XGBClassifier,
    metrics: dict,
    plot_paths: list[Path],
) -> str:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.log_params({**XGB_PARAMS, "features": ",".join(FEATURES)})
        mlflow.log_metrics(metrics)

        for p in plot_paths:
            mlflow.log_artifact(str(p), artifact_path="plots")

        mlflow.xgboost.log_model(model, name="xgboost-ad-model")
        run_id = run.info.run_id

    logger.info("Run registrado: %s", run_id)
    return run_id


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Data
    df = load_data()
    X = df[FEATURES]
    y = df["anomaly"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("Split — Train: %d | Test: %d", len(X_train), len(X_test))

    # 2. Train + latency
    model, fit_time_s = fit_model(X_train, y_train)
    y_proba, predict_time_ms = predict_with_latency(model, X_test)

    # 3. Metrics (paper protocol)
    avg_precision = float(average_precision_score(y_test, y_proba))
    roc_auc       = float(roc_auc_score(y_test, y_proba))
    logger.info("Average Precision: %.4f | ROC AUC: %.4f", avg_precision, roc_auc)

    metrics = {
        "average_precision":  round(avg_precision, 6),
        "roc_auc":            round(roc_auc, 6),
        "fit_time_s":         round(fit_time_s, 6),
        "predict_time_ms":    round(predict_time_ms, 6),
        "n_train":            int(len(X_train)),
        "n_test":             int(len(X_test)),
        "anomaly_rate_test":  round(float(y_test.mean()), 4),
    }

    # 4. Plots
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    pr_path = PLOTS_DIR / "pr_curve.png"
    roc_path = PLOTS_DIR / "roc_curve.png"
    fi_path  = PLOTS_DIR / "feature_importance.png"

    y_test_arr = y_test.to_numpy()
    _pr_curve_plot(y_test_arr, y_proba, pr_path)
    _roc_curve_plot(y_test_arr, y_proba, roc_path)
    _feature_importance_plot(model, fi_path)

    # 5. Persist + MLflow
    save_model_local(model)
    run_id = log_to_mlflow(model, metrics, [pr_path, roc_path, fi_path])

    # 6. Summary
    print("\n" + "=" * 52)
    print("  ANOMALY DETECTION — RESULTADOS (paper metrics)")
    print("=" * 52)
    print(f"  Average Precision : {avg_precision:.4f}")
    print(f"  ROC AUC           : {roc_auc:.4f}")
    print(f"  Fit Time          : {fit_time_s:.4f} s")
    print(f"  Predict Time      : {predict_time_ms:.6f} ms/sample")
    print(f"  Anomaly rate (test): {y_test.mean():.1%}")
    print(f"\n  MLflow Run ID : {run_id}")
    print(f"  MLflow UI     : {MLFLOW_URI}")
    print(f"  Modelo local  : {MODELS_DIR}/xgboost_ad.json")


if __name__ == "__main__":
    main()
