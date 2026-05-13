"""RAG evaluation: Smart Factory log golden set + baseline vs RAG comparison.

Condições avaliadas:
  baseline — LLM puro, sem nenhum contexto recuperado
  rag      — Pipeline completo: embed query → Milvus smartfactory_logs → generate

Métricas por condição (média sobre o golden set):
  rouge_l              — F1 da LCS entre resposta e referência (pure Python)
  semantic_similarity  — cosseno entre embeddings nomic da resposta e da referência
  avg_retrieval_score  — média dos scores COSINE retornados pelo Milvus (0.0 para baseline)

Uso:
  python mlflow/evaluate_rag.py
  python mlflow/evaluate_rag.py --force-gen
  python mlflow/evaluate_rag.py --skip-eval
  python mlflow/evaluate_rag.py --conditions baseline,rag
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Windows console default é cp1252, que não codifica emojis (MLflow imprime "🏃").
# Força UTF-8 no stdout/stderr para evitar UnicodeEncodeError no end-of-run logging.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

import time

import mlflow
import numpy as np
import psycopg2
from dotenv import load_dotenv
from fastembed import TextEmbedding

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.generator import call_raw, generate
from rag.retriever import retrieve

load_dotenv()

# O .env aponta MLFLOW_S3_ENDPOINT_URL para http://minio:9000 (hostname interno
# da rede Docker). Quando este script roda no host (fora da rede), o DNS falha
# no upload de artifact. Reescreve para localhost antes do mlflow inicializar.
_s3_endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "")
if "://minio:" in _s3_endpoint:
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = _s3_endpoint.replace("://minio:", "://localhost:")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

POSTGRES_HOST     = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT     = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB       = os.getenv("POSTGRES_DB", "aiops_industry")
POSTGRES_USER     = os.getenv("POSTGRES_USER", "aiops")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

GEN_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
# Delay mínimo entre chamadas ao LLM durante geração do golden set.
# Gemini free tier: 15 RPM → 4s mínimo. Ajuste para 0 com Ollama local.
INTER_REQUEST_DELAY: float = float(os.getenv("LLM_REQUEST_DELAY", "5"))

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT = "rag-evaluation"

TOP_N_SCENARIOS = 25
GOLDEN_SET_PATH = Path("mlflow/golden_set.json")
HTML_REPORT_TMP = Path("/tmp/rag_eval_report.html")

TIERED_FACTUAL_PER_STATION = 2  # 2 × 7 estações = 14 cenários factuais
TIERED_CROSS_STATION_N     = 10
TIERED_CAUSAL_N            = 10
CROSS_STATION_WINDOW_S     = 5  # janela temporal ±5s para correlação

EMBED_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

_embed_model: TextEmbedding | None = None


# ── Embedding ─────────────────────────────────────────────────────────────────

def _get_embed_model() -> TextEmbedding:
    global _embed_model
    if _embed_model is None:
        logger.info("Carregando modelo de embedding para avaliação...")
        _embed_model = TextEmbedding(model_name=EMBED_MODEL_NAME)
    return _embed_model


def _embed(text: str) -> np.ndarray:
    return list(_get_embed_model().embed([text]))[0]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── ROUGE-L puro ──────────────────────────────────────────────────────────────

def _lcs(x: list[str], y: list[str]) -> int:
    m, n = len(x), len(y)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            curr[j] = prev[j - 1] + 1 if x[i - 1] == y[j - 1] else max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def rouge_l(hypothesis: str, reference: str) -> float:
    h = hypothesis.lower().split()
    r = reference.lower().split()
    if not h or not r:
        return 0.0
    lcs = _lcs(h, r)
    p = lcs / len(h)
    rec = lcs / len(r)
    if p + rec == 0:
        return 0.0
    return 2 * p * rec / (p + rec)


# ── PostgreSQL ────────────────────────────────────────────────────────────────

def _pg_connect():
    return psycopg2.connect(
        host=POSTGRES_HOST, port=int(POSTGRES_PORT),
        dbname=POSTGRES_DB, user=POSTGRES_USER, password=POSTGRES_PASSWORD,
    )


def extract_log_scenarios(n: int = TOP_N_SCENARIOS) -> list[dict]:
    """Extract the N longest-duration anomalous events from smartfactory_logs."""
    logger.info("Extraindo top-%d cenários de log anômalo do PostgreSQL...", n)
    conn = _pg_connect()
    try:
        with conn, conn.cursor() as cur:
            cur.execute("""
                SELECT
                    station,
                    event_timestamp,
                    current_task,
                    current_sub_task,
                    current_task_duration
                FROM smartfactory_logs
                WHERE current_state = 'not ready'
                  AND current_task IS NOT NULL
                  AND current_task != ''
                ORDER BY current_task_duration DESC
                LIMIT %s
            """, (n,))
            rows = cur.fetchall()
    finally:
        conn.close()

    scenarios = [
        {
            "station":               r[0],
            "event_timestamp":       str(r[1]),
            "current_task":          str(r[2]),
            "current_sub_task":      str(r[3] or ""),
            "current_task_duration": float(r[4] or 0),
        }
        for r in rows
    ]
    logger.info("  %d cenários extraídos.", len(scenarios))
    return scenarios


# ── Golden set via Ollama ─────────────────────────────────────────────────────

_QA_PROMPT = """\
Você é um especialista em manutenção de fábricas inteligentes com automação industrial.
Com base nos dados de log abaixo de uma estação em estado 'not ready', gere:
  1. Uma PERGUNTA realista que um técnico de manutenção faria ao analisar este padrão.
  2. Uma RESPOSTA IDEAL completa: o que o padrão indica, causa provável, ação corretiva recomendada.

Dados do evento:
  Estação: {station}
  Timestamp: {event_timestamp}
  Tarefa: {current_task}
  Sub-tarefa: {current_sub_task}
  Duração: {duration:.3f}s

Responda SOMENTE com JSON válido, sem markdown:
{{"pergunta": "...", "resposta_ideal": "..."}}"""


def _generate_qa_pair(scenario: dict) -> dict | None:
    prompt = _QA_PROMPT.format(
        station=scenario["station"],
        event_timestamp=scenario["event_timestamp"],
        current_task=scenario["current_task"],
        current_sub_task=scenario["current_sub_task"],
        duration=scenario["current_task_duration"],
    )
    try:
        raw = call_raw(prompt, model=GEN_MODEL).strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
        return {
            "station":               scenario["station"],
            "event_timestamp":       scenario["event_timestamp"],
            "current_task":          scenario["current_task"],
            "current_sub_task":      scenario["current_sub_task"],
            "current_task_duration": scenario["current_task_duration"],
            "question":              data["pergunta"],
            "reference_answer":      data["resposta_ideal"],
        }
    except Exception as exc:
        logger.warning(
            "Falha ao gerar Q&A para estação %s timestamp %s: %s",
            scenario["station"], scenario["event_timestamp"], exc,
        )
        return None


def build_golden_set(scenarios: list[dict], force: bool = False) -> list[dict]:
    if not force and GOLDEN_SET_PATH.exists():
        data = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
        if data:
            logger.info("Golden set em cache: %s (%d pares).", GOLDEN_SET_PATH, len(data))
            return data
        logger.info("Golden set em cache vazio — regenerando...")

    logger.info("Gerando golden set via %s (delay=%.0fs entre chamadas)...", GEN_MODEL, INTER_REQUEST_DELAY)
    golden: list[dict] = []
    for i, scenario in enumerate(scenarios, 1):
        logger.info(
            "  [%d/%d] estação %s | %s",
            i, len(scenarios), scenario["station"], scenario["event_timestamp"],
        )
        pair = _generate_qa_pair(scenario)
        if pair:
            golden.append(pair)
        if i < len(scenarios) and INTER_REQUEST_DELAY > 0:
            time.sleep(INTER_REQUEST_DELAY)

    GOLDEN_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_SET_PATH.write_text(
        json.dumps(golden, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Golden set salvo: %s (%d pares).", GOLDEN_SET_PATH, len(golden))
    return golden


# ── Extractors tiered (Passo 0) ───────────────────────────────────────────────

def extract_factual_scenarios(per_station: int = TIERED_FACTUAL_PER_STATION) -> list[dict]:
    """Amostra estratificada por estação (pseudo-aleatória determinística via MD5).

    Garante cobertura de todas as 7 estações no tier factual — inclui
    estações sem `current_task` (EC_1 telemetria, WT_1) para perguntas
    do tipo "qual o estado de X às T?".
    """
    logger.info("Extraindo cenários factuais (%d por estação)...", per_station)
    conn = _pg_connect()
    try:
        with conn, conn.cursor() as cur:
            cur.execute("""
                WITH ranked AS (
                    SELECT id, station, event_timestamp, current_state, current_task,
                           current_sub_task, current_task_duration,
                           ROW_NUMBER() OVER (
                               PARTITION BY station
                               ORDER BY MD5(id::text)
                           ) AS rn
                    FROM smartfactory_logs
                    WHERE split = 'train'
                )
                SELECT id, station, event_timestamp, current_state, current_task,
                       current_sub_task, current_task_duration
                FROM ranked
                WHERE rn <= %s
                ORDER BY station, rn
            """, (per_station,))
            rows = cur.fetchall()
    finally:
        conn.close()

    scenarios = [
        {
            "tier":                  "factual",
            "stations":              [r[1]],
            "event_timestamp":       str(r[2]),
            "evidence_event_ids":    [r[0]],
            "scenario_context": {
                "current_state":         r[3],
                "current_task":          r[4] or "",
                "current_sub_task":      r[5] or "",
                "current_task_duration": float(r[6] or 0),
            },
        }
        for r in rows
    ]
    logger.info("  %d cenários factuais extraídos.", len(scenarios))
    return scenarios


def extract_cross_station_scenarios(n: int = TIERED_CROSS_STATION_N) -> list[dict]:
    """Pares de eventos em estações distintas na mesma janela temporal (±5s).

    Para cada top-anomalia, retorna o evento mais próximo no tempo em
    outra estação. Útil para perguntas de correlação cross-station.
    """
    logger.info("Extraindo cenários cross-station (n=%d, janela ±%ds)...", n, CROSS_STATION_WINDOW_S)
    conn = _pg_connect()
    try:
        with conn, conn.cursor() as cur:
            cur.execute("""
                WITH per_station AS (
                    SELECT id, station, event_timestamp, current_state,
                           current_task, current_sub_task, current_task_duration,
                           ROW_NUMBER() OVER (
                               PARTITION BY station
                               ORDER BY current_task_duration DESC
                           ) AS station_rn
                    FROM smartfactory_logs
                    WHERE split = 'train'
                      AND current_state = 'not ready'
                      AND current_task IS NOT NULL AND current_task != ''
                ),
                anomalies AS (
                    SELECT id, station, event_timestamp, current_state,
                           current_task, current_sub_task, current_task_duration,
                           ROW_NUMBER() OVER (
                               ORDER BY station_rn, current_task_duration DESC
                           ) AS rn
                    FROM per_station
                    WHERE station_rn <= 30
                ),
                paired AS (
                    SELECT a.id AS id_a, a.station AS station_a, a.event_timestamp AS ts_a,
                           a.current_state AS state_a, a.current_task AS task_a,
                           a.current_sub_task AS sub_a, a.current_task_duration AS dur_a,
                           a.rn,
                           b.id AS id_b, b.station AS station_b, b.event_timestamp AS ts_b,
                           b.current_state AS state_b, b.current_task AS task_b,
                           b.current_sub_task AS sub_b,
                           ROW_NUMBER() OVER (
                               PARTITION BY a.id
                               ORDER BY ABS(EXTRACT(EPOCH FROM (b.event_timestamp - a.event_timestamp)))
                           ) AS pair_rn
                    FROM anomalies a
                    JOIN smartfactory_logs b
                      ON b.station != a.station
                     AND b.split = 'train'
                     AND b.event_timestamp BETWEEN a.event_timestamp - INTERVAL %s
                                               AND a.event_timestamp + INTERVAL %s
                )
                SELECT id_a, station_a, ts_a, state_a, task_a, sub_a, dur_a,
                       id_b, station_b, ts_b, state_b, task_b, sub_b
                FROM paired
                WHERE pair_rn = 1
                ORDER BY rn
                LIMIT %s
            """, (f"{CROSS_STATION_WINDOW_S} seconds", f"{CROSS_STATION_WINDOW_S} seconds", n))
            rows = cur.fetchall()
    finally:
        conn.close()

    scenarios = [
        {
            "tier":                  "cross_station",
            "stations":              [r[1], r[8]],
            "event_timestamp":       str(r[2]),
            "evidence_event_ids":    [r[0], r[7]],
            "scenario_context": {
                "station_a":          r[1],
                "event_a_ts":         str(r[2]),
                "event_a_state":      r[3],
                "event_a_task":       r[4] or "",
                "event_a_sub_task":   r[5] or "",
                "event_a_duration":   float(r[6] or 0),
                "station_b":          r[8],
                "event_b_ts":         str(r[9]),
                "event_b_state":      r[10],
                "event_b_task":       r[11] or "",
                "event_b_sub_task":   r[12] or "",
            },
        }
        for r in rows
    ]
    logger.info("  %d cenários cross-station extraídos.", len(scenarios))
    return scenarios


def extract_causal_scenarios(n: int = TIERED_CAUSAL_N) -> list[dict]:
    """Anomalias de maior duração + sub-task imediatamente anterior na mesma estação.

    A sub-task anterior dá pista de causa raiz: 'estava em Heating, agora não-pronto'.
    """
    logger.info("Extraindo cenários causais (n=%d)...", n)
    conn = _pg_connect()
    try:
        with conn, conn.cursor() as cur:
            cur.execute("""
                WITH ranked AS (
                    SELECT id, station, event_timestamp, current_state, current_task,
                           current_sub_task, current_task_duration,
                           LAG(id)               OVER (PARTITION BY station ORDER BY event_timestamp) AS prev_id,
                           LAG(current_task)     OVER (PARTITION BY station ORDER BY event_timestamp) AS prev_task,
                           LAG(current_sub_task) OVER (PARTITION BY station ORDER BY event_timestamp) AS prev_sub_task,
                           LAG(event_timestamp)  OVER (PARTITION BY station ORDER BY event_timestamp) AS prev_ts
                    FROM smartfactory_logs
                    WHERE split = 'train'
                ),
                anomalies AS (
                    SELECT id, station, event_timestamp, current_state, current_task,
                           current_sub_task, current_task_duration,
                           prev_id, prev_task, prev_sub_task, prev_ts,
                           ROW_NUMBER() OVER (
                               PARTITION BY station
                               ORDER BY current_task_duration DESC
                           ) AS station_rn
                    FROM ranked
                    WHERE current_state = 'not ready'
                      AND current_task IS NOT NULL AND current_task != ''
                      AND prev_id IS NOT NULL
                )
                SELECT id, station, event_timestamp, current_state, current_task,
                       current_sub_task, current_task_duration,
                       prev_id, prev_task, prev_sub_task, prev_ts
                FROM anomalies
                ORDER BY station_rn, current_task_duration DESC
                LIMIT %s
            """, (n,))
            rows = cur.fetchall()
    finally:
        conn.close()

    scenarios = [
        {
            "tier":                  "causal",
            "stations":              [r[1]],
            "event_timestamp":       str(r[2]),
            "evidence_event_ids":    [r[0], r[7]],
            "scenario_context": {
                "current_state":         r[3],
                "current_task":          r[4] or "",
                "current_sub_task":      r[5] or "",
                "current_task_duration": float(r[6] or 0),
                "prev_task":             r[8] or "",
                "prev_sub_task":         r[9] or "",
                "prev_event_ts":         str(r[10]),
            },
        }
        for r in rows
    ]
    logger.info("  %d cenários causais extraídos.", len(scenarios))
    return scenarios


# ── Prompts por tier ──────────────────────────────────────────────────────────

_QA_PROMPT_FACTUAL = """\
Você é um técnico de manutenção de uma fábrica inteligente com 7 estações \
(MM_1, EC_1, SM_1, HBW_1, OV_1, VGR_1, WT_1). Olhando o evento abaixo, gere:
  1. Uma PERGUNTA factual e direta, de um único hop (sobre estado, tarefa ou \
duração nesse instante).
  2. Uma RESPOSTA IDEAL curta e objetiva (1-3 frases) usando exatamente os dados \
fornecidos. Cite a estação e o timestamp.

Evento:
  Estação: {station}
  Timestamp: {event_timestamp}
  Estado: {current_state}
  Tarefa: {current_task}
  Sub-tarefa: {current_sub_task}
  Duração na sub-tarefa: {current_task_duration:.3f}s

Responda SOMENTE com JSON válido, sem markdown:
{{"pergunta": "...", "resposta_ideal": "..."}}"""


_QA_PROMPT_CROSS_STATION = """\
Você é um técnico de manutenção de uma fábrica inteligente com 7 estações \
cooperando em fluxo. Dado o par de eventos abaixo, observados na MESMA \
janela temporal (±5s) em estações DIFERENTES, gere:
  1. Uma PERGUNTA de correlação cross-station: o que estava acontecendo \
em '{station_b}' enquanto '{station_a}' apresentava o comportamento descrito? \
A pergunta deve exigir olhar as DUAS estações.
  2. Uma RESPOSTA IDEAL que cite EXPLICITAMENTE ambas as estações, seus \
estados e tarefas no instante, e proponha uma hipótese de relação causal \
ou temporal entre elas.

Evento A:
  Estação: {station_a}
  Timestamp: {event_a_ts}
  Estado: {event_a_state}
  Tarefa: {event_a_task} | Sub-tarefa: {event_a_sub_task}
  Duração: {event_a_duration:.3f}s

Evento B (mesma janela ±5s):
  Estação: {station_b}
  Timestamp: {event_b_ts}
  Estado: {event_b_state}
  Tarefa: {event_b_task} | Sub-tarefa: {event_b_sub_task}

Responda SOMENTE com JSON válido, sem markdown:
{{"pergunta": "...", "resposta_ideal": "..."}}"""


_QA_PROMPT_CAUSAL = """\
Você é um engenheiro de manutenção investigando causa raiz em uma fábrica \
inteligente. Dado o evento atual (em estado 'not ready' com duração alta) e \
a sub-tarefa imediatamente anterior na MESMA estação, gere:
  1. Uma PERGUNTA causal/diagnóstica do tipo 'por que' ou 'qual sub-tarefa \
antecedeu' que peça inferência de causa raiz.
  2. Uma RESPOSTA IDEAL com: (a) o que a sequência sub-tarefa anterior → \
estado atual indica; (b) hipótese de causa provável; (c) ação corretiva \
recomendada. Cite a sub-tarefa anterior nominalmente.

Evento atual:
  Estação: {station}
  Timestamp: {event_timestamp}
  Estado: {current_state}
  Tarefa: {current_task} | Sub-tarefa: {current_sub_task}
  Duração na sub-tarefa: {current_task_duration:.3f}s

Sub-tarefa imediatamente anterior na mesma estação:
  Tarefa anterior: {prev_task}
  Sub-tarefa anterior: {prev_sub_task}
  Timestamp anterior: {prev_event_ts}

Responda SOMENTE com JSON válido, sem markdown:
{{"pergunta": "...", "resposta_ideal": "..."}}"""


# ── Dispatcher tiered ─────────────────────────────────────────────────────────

def _parse_qa_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def _generate_qa_pair_tiered(scenario: dict, idx: int) -> dict | None:
    tier = scenario["tier"]
    ctx = scenario["scenario_context"]

    if tier == "factual":
        prompt = _QA_PROMPT_FACTUAL.format(
            station=scenario["stations"][0],
            event_timestamp=scenario["event_timestamp"],
            current_state=ctx["current_state"],
            current_task=ctx["current_task"],
            current_sub_task=ctx["current_sub_task"],
            current_task_duration=ctx["current_task_duration"],
        )
    elif tier == "cross_station":
        prompt = _QA_PROMPT_CROSS_STATION.format(**ctx)
    elif tier == "causal":
        prompt = _QA_PROMPT_CAUSAL.format(
            station=scenario["stations"][0],
            event_timestamp=scenario["event_timestamp"],
            current_state=ctx["current_state"],
            current_task=ctx["current_task"],
            current_sub_task=ctx["current_sub_task"],
            current_task_duration=ctx["current_task_duration"],
            prev_task=ctx["prev_task"],
            prev_sub_task=ctx["prev_sub_task"],
            prev_event_ts=ctx["prev_event_ts"],
        )
    else:
        logger.warning("Tier desconhecido: %s — pulando.", tier)
        return None

    try:
        raw = call_raw(prompt, model=GEN_MODEL)
        data = _parse_qa_json(raw)
    except Exception as exc:
        logger.warning(
            "Falha ao gerar Q&A (tier=%s, stations=%s): %s",
            tier, scenario["stations"], exc,
        )
        return None

    # qwen2.5 às vezes retorna resposta_ideal como dict {"a":..., "b":..., "c":...}
    # (interpretou "(a) ... (b) ... (c) ..." do prompt como structured output).
    # Achata pra string única concatenando os values com \n\n.
    answer = data.get("resposta_ideal")
    if isinstance(answer, dict):
        answer = "\n\n".join(str(v) for v in answer.values())
    elif isinstance(answer, list):
        answer = "\n\n".join(str(v) for v in answer)
    elif not isinstance(answer, str):
        answer = str(answer) if answer is not None else ""
    data["resposta_ideal"] = answer

    # Campos de compatibilidade com _evaluate_condition / report HTML.
    compat_station = ", ".join(scenario["stations"])
    compat_task    = ctx.get("current_task") or ctx.get("event_a_task", "")
    compat_sub     = ctx.get("current_sub_task") or ctx.get("event_a_sub_task", "")
    compat_dur     = ctx.get("current_task_duration") or ctx.get("event_a_duration", 0.0)

    return {
        "id":                  f"{tier}-{idx:04d}",
        "tier":                tier,
        "stations":            scenario["stations"],
        "event_timestamp":     scenario["event_timestamp"],
        "evidence_event_ids":  scenario["evidence_event_ids"],
        "scenario_context":    scenario["scenario_context"],
        "question":            data["pergunta"],
        "reference_answer":    data["resposta_ideal"],
        # Compat (mantém _evaluate_condition / HTML report funcionando):
        "station":               compat_station,
        "current_task":          compat_task,
        "current_sub_task":      compat_sub,
        "current_task_duration": float(compat_dur),
    }


def build_golden_set_tiered(force: bool = False) -> list[dict]:
    """Gera o golden set com os 3 tiers (factual + cross_station + causal)."""
    if not force and GOLDEN_SET_PATH.exists():
        data = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
        if data:
            logger.info("Golden set tiered em cache: %s (%d pares).", GOLDEN_SET_PATH, len(data))
            return data
        logger.info("Golden set em cache vazio — regenerando tiered...")

    factual = extract_factual_scenarios()
    cross   = extract_cross_station_scenarios()
    causal  = extract_causal_scenarios()

    all_scenarios = factual + cross + causal
    logger.info(
        "Gerando golden set tiered via %s (factual=%d, cross=%d, causal=%d, delay=%.0fs)...",
        GEN_MODEL, len(factual), len(cross), len(causal), INTER_REQUEST_DELAY,
    )

    golden: list[dict] = []
    for i, scenario in enumerate(all_scenarios, 1):
        logger.info(
            "  [%d/%d] tier=%s | stations=%s | %s",
            i, len(all_scenarios), scenario["tier"],
            scenario["stations"], scenario["event_timestamp"],
        )
        pair = _generate_qa_pair_tiered(scenario, i)
        if pair:
            golden.append(pair)
        if i < len(all_scenarios) and INTER_REQUEST_DELAY > 0:
            time.sleep(INTER_REQUEST_DELAY)

    GOLDEN_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_SET_PATH.write_text(
        json.dumps(golden, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Golden set tiered salvo: %s (%d pares).", GOLDEN_SET_PATH, len(golden))
    return golden


# ── Condições de avaliação ────────────────────────────────────────────────────

_BASELINE_PROMPT = """\
Você é um engenheiro de manutenção especializado em fábricas inteligentes com automação industrial.
Responda em português, de forma clara e objetiva.

Com base no seu conhecimento geral, analise a situação descrita e inclua:
1. O que o padrão descrito indica sobre o estado do sistema
2. Causa provável do problema
3. Ação corretiva recomendada

Se não tiver informações suficientes, diga isso claramente.

PERGUNTA: {query}"""


def _run_baseline(question: str) -> tuple[str, list[dict]]:
    answer = call_raw(_BASELINE_PROMPT.format(query=question), model=GEN_MODEL)
    return answer, []


def _run_rag(question: str) -> tuple[str, list[dict]]:
    hits = retrieve(question, top_k=5)
    answer = generate(question, hits, model=GEN_MODEL)
    return answer, hits


_RUN_FN = {
    "baseline": _run_baseline,
    "rag":      _run_rag,
}


# ── Métricas ──────────────────────────────────────────────────────────────────

def _compute_metrics(answer: str, reference: str, hits: list[dict]) -> dict:
    rl      = rouge_l(answer, reference)
    sem_sim = _cosine(_embed(answer), _embed(reference))
    avg_score = float(np.mean([h["score"] for h in hits])) if hits else 0.0
    return {
        "rouge_l":             round(rl, 4),
        "semantic_similarity": round(float(sem_sim), 4),
        "avg_retrieval_score": round(avg_score, 4),
    }


def _evaluate_condition(golden: list[dict], condition: str) -> tuple[dict, list[dict]]:
    run_fn = _RUN_FN[condition]
    buckets: dict[str, list[float]] = {
        "rouge_l": [], "semantic_similarity": [], "avg_retrieval_score": [],
    }
    per_q: list[dict] = []

    for i, pair in enumerate(golden, 1):
        logger.info("  [%s] Q%d/%d — %s", condition, i, len(golden), pair["question"][:70])
        answer, hits = run_fn(pair["question"])
        m = _compute_metrics(answer, pair["reference_answer"], hits)
        for k in buckets:
            buckets[k].append(m[k])
        per_q.append({
            "station":       pair["station"],
            "current_task":  pair["current_task"],
            "question":      pair["question"],
            "reference_answer": pair["reference_answer"],
            "answer":        answer,
            **m,
        })

    aggregated = {k: round(float(np.mean(v)), 4) for k, v in buckets.items()}
    aggregated["n_questions"] = len(golden)
    return aggregated, per_q


# ── HTML report ───────────────────────────────────────────────────────────────

def _build_html_report(results: dict[str, tuple[dict, list[dict]]]) -> str:
    conditions = list(results.keys())
    METRICS = ["rouge_l", "semantic_similarity", "avg_retrieval_score"]
    LABELS  = {
        "rouge_l":             "ROUGE-L",
        "semantic_similarity": "Sem. Similarity",
        "avg_retrieval_score": "Avg Retrieval Score",
    }

    header_cells = "".join(f"<th>{c}</th>" for c in conditions)
    summary_rows = ""
    for m in METRICS:
        vals = [results[c][0].get(m, 0.0) for c in conditions]
        best = max(vals)
        cells = ""
        for v in vals:
            style = "background:#c8f7c5;font-weight:bold" if v == best and best > 0 else ""
            cells += f'<td style="{style}">{v:.4f}</td>'
        summary_rows += f"<tr><td><b>{LABELS[m]}</b></td>{cells}</tr>\n"

    summary_table = f"""
<table border="1" cellpadding="8" cellspacing="0"
       style="border-collapse:collapse;font-family:monospace;margin-bottom:30px">
  <thead style="background:#2c3e50;color:#fff">
    <tr><th>Métrica</th>{header_cells}</tr>
  </thead>
  <tbody>{summary_rows}</tbody>
</table>"""

    _, rag_qs = results.get("rag", ({}, []))
    detail_rows = ""
    for q in rag_qs:
        detail_rows += f"""
<tr>
  <td>{q['station']}<br/><small style="color:#888">{q.get('current_task','')[:40]}</small></td>
  <td style="max-width:280px;font-size:13px">{q['question']}</td>
  <td style="max-width:300px;font-size:12px;color:#555">{q['answer'][:300]}…</td>
  <td>{q['rouge_l']:.3f}</td>
  <td>{q['semantic_similarity']:.3f}</td>
</tr>"""

    detail_table = f"""
<table border="1" cellpadding="6" cellspacing="0"
       style="border-collapse:collapse;font-family:monospace;width:100%">
  <thead style="background:#34495e;color:#fff">
    <tr><th>Estação</th><th>Pergunta</th><th>Resposta (trecho)</th>
        <th>ROUGE-L</th><th>Sem.Sim</th></tr>
  </thead>
  <tbody>{detail_rows}</tbody>
</table>"""

    baseline_agg = results.get("baseline", ({}, []))[0]
    delta_rows = ""
    for m in ["rouge_l", "semantic_similarity"]:
        b_val = baseline_agg.get(m, 0.0)
        for cond in [c for c in conditions if c != "baseline"]:
            c_val = results.get(cond, ({}, []))[0].get(m, 0.0)
            delta = c_val - b_val
            color = "green" if delta > 0 else "red"
            if b_val != 0:
                delta_rows += (
                    f"<tr><td>{LABELS[m]}</td><td>{cond}</td>"
                    f'<td style="color:{color}">{delta:+.4f} ({delta/b_val*100:+.1f}%)</td></tr>\n'
                )
            else:
                delta_rows += f"<tr><td>{LABELS[m]}</td><td>{cond}</td><td>N/A</td></tr>\n"

    delta_table = f"""
<table border="1" cellpadding="8" cellspacing="0"
       style="border-collapse:collapse;font-family:monospace">
  <thead style="background:#1a5276;color:#fff">
    <tr><th>Métrica</th><th>Condição</th><th>Delta vs baseline</th></tr>
  </thead>
  <tbody>{delta_rows}</tbody>
</table>"""

    n_qs = len(rag_qs) if rag_qs else len(results.get("baseline", ({}, []))[1])
    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <title>RAG Evaluation — AIOps Industry</title>
  <style>
    body {{ font-family: sans-serif; margin: 40px; color: #222; }}
    h1   {{ color: #1a252f; }}
    h2   {{ color: #2c3e50; border-bottom: 2px solid #ccc; padding-bottom: 6px; margin-top: 40px; }}
    td, th {{ vertical-align: top; padding: 6px 10px; }}
  </style>
</head>
<body>
  <h1>RAG Evaluation — AIOps Industry</h1>
  <p>Modelo LLM: <code>{GEN_MODEL}</code> &nbsp;|&nbsp;
     Embedding: <code>{EMBED_MODEL_NAME}</code> &nbsp;|&nbsp;
     Dataset: Smart Factory Logs (10Hz, 7 estações) &nbsp;|&nbsp;
     Golden set: {n_qs} perguntas</p>

  <h2>1. Comparação entre condições (média)</h2>
  {summary_table}

  <h2>2. Delta vs baseline (ganho do RAG)</h2>
  {delta_table}

  <h2>3. Detalhe por pergunta — condição <em>rag</em></h2>
  {detail_table}
</body>
</html>"""


# ── MLflow logging ────────────────────────────────────────────────────────────

def log_to_mlflow(
    results: dict[str, tuple[dict, list[dict]]],
    golden: list[dict],
) -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="rag-eval-comparison") as parent:
        mlflow.log_param("llm_model",   GEN_MODEL)
        mlflow.log_param("embed_model", EMBED_MODEL_NAME)
        mlflow.log_param("n_questions", len(golden))
        mlflow.log_param("top_k_milvus", 5)
        mlflow.log_param("conditions",  ",".join(results.keys()))
        mlflow.log_param("dataset",     "smartfactory_logs")

        mlflow.log_artifact(str(GOLDEN_SET_PATH), artifact_path="golden_set")

        for condition, (aggregated, per_q) in results.items():
            with mlflow.start_run(run_name=condition, nested=True):
                mlflow.log_param("condition",   condition)
                mlflow.log_param("model",       GEN_MODEL)
                mlflow.log_param("n_questions", aggregated["n_questions"])

                numeric = {k: v for k, v in aggregated.items() if k != "n_questions"}
                mlflow.log_metrics(numeric)

                per_q_path = f"/tmp/per_q_{condition}.json"
                Path(per_q_path).write_text(
                    json.dumps(per_q, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                mlflow.log_artifact(per_q_path, artifact_path="per_question")

        html = _build_html_report(results)
        HTML_REPORT_TMP.write_text(html, encoding="utf-8")
        mlflow.log_artifact(str(HTML_REPORT_TMP), artifact_path="reports")

        logger.info(
            "MLflow: experimento '%s' | run ID: %s",
            EXPERIMENT, parent.info.run_id,
        )
        logger.info("Abra: %s/#/experiments", MLFLOW_URI)


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Avalia o RAG em condições baseline vs rag e loga no MLflow."
    )
    parser.add_argument("--force-gen",  action="store_true",
                        help="Regenera o golden set mesmo que exista.")
    parser.add_argument("--skip-eval",  action="store_true",
                        help="Só gera/exibe o golden set, sem avaliar.")
    parser.add_argument("--conditions", default="baseline,rag",
                        help="Condições a avaliar (csv). Default: baseline,rag")
    parser.add_argument("--tiered",     action="store_true",
                        help="Usa golden set tiered (factual+cross_station+causal). "
                             "Sem a flag, mantém comportamento legado (top-25 anomalias).")
    args = parser.parse_args()

    if args.tiered:
        golden = build_golden_set_tiered(force=args.force_gen)
    else:
        scenarios = extract_log_scenarios(TOP_N_SCENARIOS)
        if not scenarios:
            logger.error(
                "Nenhum cenário de log encontrado. "
                "Verifique se smartfactory_logs está populada (make ingest-all)."
            )
            sys.exit(1)
        golden = build_golden_set(scenarios, force=args.force_gen)
    if not golden:
        logger.error("Golden set vazio. Verifique o LLM configurado (LLM_MODEL=%s).", GEN_MODEL)
        sys.exit(1)

    if args.skip_eval:
        print(f"\nGolden set ({len(golden)} pares) salvo em {GOLDEN_SET_PATH}")
        for p in golden:
            print(f"  [{p['station']} | {p['event_timestamp']}] {p['question'][:80]}")
        return

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip() in _RUN_FN]
    results: dict[str, tuple[dict, list[dict]]] = {}

    for condition in conditions:
        logger.info("=== Avaliando: %s (%d perguntas) ===", condition, len(golden))
        agg, per_q = _evaluate_condition(golden, condition)
        results[condition] = (agg, per_q)
        logger.info(
            "  %s — ROUGE-L: %.4f | Sem.Sim: %.4f | AvgScore: %.4f",
            condition, agg["rouge_l"], agg["semantic_similarity"], agg["avg_retrieval_score"],
        )

    log_to_mlflow(results, golden)

    print("\n" + "=" * 60)
    print("RESULTADO FINAL")
    print("=" * 60)
    header = f"{'Condição':<12} {'ROUGE-L':>8} {'Sem.Sim':>10} {'AvgScore':>10}"
    print(header)
    print("-" * len(header))
    for cond, (agg, _) in results.items():
        print(
            f"{cond:<12} {agg['rouge_l']:>8.4f}"
            f" {agg['semantic_similarity']:>10.4f}"
            f" {agg['avg_retrieval_score']:>10.4f}"
        )
    print(f"\nMLflow UI → {MLFLOW_URI}/#/experiments")
    print(f"Relatório  → abra o artefato reports/rag_eval_report.html no MLflow")


if __name__ == "__main__":
    main()
