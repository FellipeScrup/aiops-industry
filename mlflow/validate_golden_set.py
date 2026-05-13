"""Validador do golden set tiered (Passo 0 do TCC).

Checa, em mlflow/golden_set.json:
  1. Schema — todos os campos obrigatórios + tier ∈ {factual, cross_station, causal}
  2. Distribuição — ≥ 8 pares por tier; ≥ 1 par por estação no tier factual
  3. Evidência existe — cada evidence_event_ids[i] está em smartfactory_logs (PostgreSQL)
  4. Tamanho mínimo — question ≥ 30 chars; reference_answer ≥ 80 chars

Saída: tabela ASCII no stdout. Exit 0 se tudo passa, 1 caso contrário.
"""

import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

POSTGRES_HOST     = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT     = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB       = os.getenv("POSTGRES_DB", "aiops_industry")
POSTGRES_USER     = os.getenv("POSTGRES_USER", "aiops")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

GOLDEN_SET_PATH = Path("mlflow/golden_set.json")
EXPECTED_STATIONS = {"MM_1", "EC_1", "SM_1", "HBW_1", "OV_1", "VGR_1", "WT_1"}
VALID_TIERS = {"factual", "cross_station", "causal"}
MIN_PER_TIER = 8
MIN_QUESTION_CHARS = 30
# Thresholds de tamanho da reference_answer por tier:
# factual = 1-3 frases, OK ser curto.
# cross_station / causal = devem citar ambas estações ou hipótese + ação corretiva → mais texto.
MIN_ANSWER_CHARS_BY_TIER = {
    # factual: pode ser frase única quando contexto é ready+idle
    # ("O estado de X é ready." ~ 35-45 chars), mas filtra
    # respostas degeneradas tipo só "ready." ou "X.".
    "factual":       40,
    "cross_station": 100,
    "causal":        120,
}

REQUIRED_FIELDS = {
    "id", "tier", "stations", "event_timestamp", "evidence_event_ids",
    "scenario_context", "question", "reference_answer",
}


def _pg_connect():
    return psycopg2.connect(
        host=POSTGRES_HOST, port=int(POSTGRES_PORT),
        dbname=POSTGRES_DB, user=POSTGRES_USER, password=POSTGRES_PASSWORD,
    )


def check_schema(pairs: list[dict]) -> list[str]:
    errors: list[str] = []
    for i, p in enumerate(pairs):
        missing = REQUIRED_FIELDS - set(p.keys())
        if missing:
            errors.append(f"par #{i} ({p.get('id', '?')}): campos faltando {missing}")
            continue
        if p["tier"] not in VALID_TIERS:
            errors.append(f"par {p['id']}: tier inválido '{p['tier']}'")
        if not isinstance(p["stations"], list) or not p["stations"]:
            errors.append(f"par {p['id']}: 'stations' deve ser lista não-vazia")
        if not isinstance(p["evidence_event_ids"], list) or not p["evidence_event_ids"]:
            errors.append(f"par {p['id']}: 'evidence_event_ids' deve ser lista não-vazia")
        if not isinstance(p["scenario_context"], dict):
            errors.append(f"par {p['id']}: 'scenario_context' deve ser dict")
    return errors


def check_distribution(pairs: list[dict]) -> list[str]:
    errors: list[str] = []
    by_tier = Counter(p["tier"] for p in pairs if "tier" in p)
    for tier in VALID_TIERS:
        if by_tier[tier] < MIN_PER_TIER:
            errors.append(
                f"tier '{tier}': {by_tier[tier]} pares (mínimo {MIN_PER_TIER})"
            )

    factual_stations: set[str] = set()
    for p in pairs:
        if p.get("tier") == "factual":
            factual_stations.update(p.get("stations", []))
    missing = EXPECTED_STATIONS - factual_stations
    if missing:
        errors.append(
            f"tier 'factual' não cobre todas as estações — faltam: {sorted(missing)}"
        )
    return errors


def check_evidence_exists(pairs: list[dict]) -> list[str]:
    all_ids: set[str] = set()
    for p in pairs:
        for eid in p.get("evidence_event_ids", []):
            all_ids.add(str(eid))
    if not all_ids:
        return ["nenhum evidence_event_id encontrado em todo o golden set"]

    try:
        conn = _pg_connect()
    except Exception as exc:
        return [f"não foi possível conectar ao PostgreSQL para validar evidências: {exc}"]

    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM smartfactory_logs WHERE id = ANY(%s)",
                (list(all_ids),),
            )
            found = {row[0] for row in cur.fetchall()}
    finally:
        conn.close()

    missing = all_ids - found
    if not missing:
        return []
    errors: list[str] = []
    for p in pairs:
        bad = [eid for eid in p.get("evidence_event_ids", []) if str(eid) in missing]
        if bad:
            errors.append(f"par {p.get('id', '?')}: evidence_event_ids ausentes no banco: {bad}")
    return errors


def check_text_lengths(pairs: list[dict]) -> list[str]:
    errors: list[str] = []
    for p in pairs:
        q = p.get("question", "")
        a = p.get("reference_answer", "")
        tier = p.get("tier", "factual")
        min_answer = MIN_ANSWER_CHARS_BY_TIER.get(tier, 80)
        if len(q) < MIN_QUESTION_CHARS:
            errors.append(
                f"par {p.get('id', '?')}: question com {len(q)} chars (mínimo {MIN_QUESTION_CHARS})"
            )
        if len(a) < min_answer:
            errors.append(
                f"par {p.get('id', '?')} [{tier}]: reference_answer com {len(a)} chars (mínimo {min_answer})"
            )
    return errors


def _print_summary(pairs: list[dict]) -> None:
    by_tier = Counter(p.get("tier", "?") for p in pairs)
    by_station: Counter = Counter()
    for p in pairs:
        for s in p.get("stations", []):
            by_station[s] += 1

    print()
    print("=" * 60)
    print(f"GOLDEN SET — {len(pairs)} pares totais")
    print("=" * 60)
    print("\nPor tier:")
    for tier in ("factual", "cross_station", "causal"):
        flag = "OK " if by_tier.get(tier, 0) >= MIN_PER_TIER else "!! "
        print(f"  {flag}{tier:<16} {by_tier.get(tier, 0):>3}")
    print("\nPor estação (todas ocorrências):")
    for st in sorted(EXPECTED_STATIONS):
        print(f"  {st:<8} {by_station.get(st, 0):>3}")
    extras = set(by_station) - EXPECTED_STATIONS
    if extras:
        print(f"  estações fora do conjunto esperado: {extras}")


def main() -> None:
    if not GOLDEN_SET_PATH.exists():
        logger.error("Golden set não encontrado em %s. Rode `make golden-set` primeiro.", GOLDEN_SET_PATH)
        sys.exit(1)

    pairs = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    if not isinstance(pairs, list) or not pairs:
        logger.error("Golden set vazio ou em formato inválido (esperado: lista JSON não-vazia).")
        sys.exit(1)

    _print_summary(pairs)

    schema_errors       = check_schema(pairs)
    distribution_errors = check_distribution(pairs)
    evidence_errors     = check_evidence_exists(pairs) if not schema_errors else []
    length_errors       = check_text_lengths(pairs)

    all_errors = {
        "schema":       schema_errors,
        "distribuição": distribution_errors,
        "evidência":    evidence_errors,
        "tamanhos":     length_errors,
    }

    print("\n" + "=" * 60)
    print("VALIDAÇÃO")
    print("=" * 60)
    any_failure = False
    for category, errs in all_errors.items():
        if errs:
            any_failure = True
            print(f"\n[FAIL] {category} ({len(errs)} problema(s)):")
            for e in errs:
                print(f"  - {e}")
        else:
            print(f"[ OK ] {category}")

    print()
    if any_failure:
        print("Validação falhou. Corrija os problemas acima e rode de novo.")
        sys.exit(1)
    print("Golden set válido. Pronto para `make eval-rag`.")


if __name__ == "__main__":
    main()
