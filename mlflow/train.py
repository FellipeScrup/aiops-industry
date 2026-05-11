"""Sprint 4 — XGBoost severity classifier with MLflow tracking.

Reads the preprocessed Parquet produced by preprocess.py and runs the
full training + MLflow tracking pipeline.
"""

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import classification_report, f1_score, accuracy_score
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
EXPERIMENT_NAME: str = "piade-degradation-classification"
RUN_NAME: str = "xgboost-baseline"

XGB_PARAMS: dict = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "eval_metric": "mlogloss",
}

FEATURES: list[str] = [
    "pct_idle",
    "pct_production",
    "pct_downtime",
    "pct_perf_loss",
    "pct_sched_downtime",
    "count_sum",
    "num_changes",
]

PARQUET_PATH = Path("data/silver/processed_telemetry.parquet")
MODELS_DIR = Path(__file__).parent / "models"


# ── Etapa 1 — Carregar dados ──────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    logger.info("Carregando dados de %s...", PARQUET_PATH)
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"{PARQUET_PATH} não encontrado. Execute 'make preprocess' primeiro."
        )
    df = pd.read_parquet(PARQUET_PATH)
    logger.info("  %d linhas carregadas.", len(df))
    return df


# ── Etapa 2 — Split e treino ──────────────────────────────────────────────────

def train(df: pd.DataFrame) -> tuple[XGBClassifier, dict, str]:
    logger.info("Dividindo dataset (80/20, stratified)...")
    X = df[FEATURES]
    y = df["label_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("  Train: %d | Test: %d", len(X_train), len(X_test))

    logger.info("Treinando XGBoost...")
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)

    logger.info("Calculando métricas...")
    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted"))
    report_str = classification_report(
        y_test, y_pred, target_names=["normal", "degraded", "critical"]
    )
    report_df = pd.DataFrame(
        classification_report(
            y_test,
            y_pred,
            target_names=["normal", "degraded", "critical"],
            output_dict=True,
        )
    ).T

    metrics: dict = {"accuracy": accuracy, "f1_weighted": f1_weighted}
    for label in ["normal", "degraded", "critical"]:
        for metric in ["precision", "recall", "f1-score"]:
            key = f"{label}_{metric.replace('-', '_')}"
            metrics[key] = float(report_df.loc[label, metric])

    logger.info("  Accuracy: %.4f | F1 (weighted): %.4f", accuracy, f1_weighted)
    logger.info("\n%s", report_str)

    return model, metrics, report_str


# ── Etapa 3 — MLflow tracking ─────────────────────────────────────────────────

def _save_feature_importance_plot(model: XGBClassifier, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(FEATURES, model.feature_importances_)
    ax.set_xlabel("Importance")
    ax.set_title("XGBoost Feature Importance")
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def log_to_mlflow(
    model: XGBClassifier,
    metrics: dict,
    report_str: str,
) -> str:
    logger.info("Registrando no MLflow (%s)...", MLFLOW_URI)
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.log_params(XGB_PARAMS)
        mlflow.log_metrics(metrics)

        report_path = "/tmp/classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report_str)
        mlflow.log_artifact(report_path, artifact_path="reports")

        importance_path = "/tmp/feature_importance.png"
        _save_feature_importance_plot(model, importance_path)
        mlflow.log_artifact(importance_path, artifact_path="plots")

        mlflow.xgboost.log_model(model, "xgboost-model")

        run_id: str = run.info.run_id

    logger.info("  Run registrado com ID: %s", run_id)
    return run_id


# ── Etapa 4 — Salvar modelo localmente ───────────────────────────────────────

def save_model_local(model: XGBClassifier) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / "xgboost_severity.json"
    model.save_model(str(dest))
    logger.info("Modelo salvo localmente em %s", dest)


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        df = load_data()
        model, metrics, report_str = train(df)
        run_id = log_to_mlflow(model, metrics, report_str)
        save_model_local(model)
    except Exception:
        logger.exception("Pipeline de treino falhou")
        raise

    print(f"Run ID: {run_id}")
    print(f"Acurácia: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
