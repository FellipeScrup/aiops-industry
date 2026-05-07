"""Seed alarm_dictionary with LLM-generated descriptions for top-50 alarm codes."""

import json
import logging
import os
import re

import psycopg2
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

POSTGRES_USER: str = os.getenv("POSTGRES_USER", "aiops")
POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "aiops")
POSTGRES_DB: str = os.getenv("POSTGRES_DB", "aiops_industry")
POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")

OLLAMA_BASE: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL: str = "llama3.2:3b"
LIMIT: int = 50

# {{ and }} are escaped curly braces that become { } in the final prompt.
_PROMPT_TEMPLATE = """\
Você é um especialista em manutenção de máquinas de embalagem industrial.
O código de alarme '{code}' ocorreu {count} vezes em máquinas industriais
de embalagem. A fonte do dado é {source} (ALPI=máquinas Galdi, PIADE=equipamentos de embalagem).

Responda APENAS com JSON válido, sem texto antes ou depois, sem markdown:
{{
  "title": "título curto da falha em português (máx 60 chars)",
  "category": "pressure|temperature|sensor|motor|seal|pneumatic|electrical|communication|mechanical|other",
  "severity": "info|warning|critical",
  "description": "descrição técnica em 1-2 frases em português",
  "probable_causes": ["causa 1", "causa 2", "causa 3"],
  "corrective_actions": ["ação 1", "ação 2", "ação 3"]
}}"""

_SELECT_TOP_CODES = """
    SELECT alarm_code,
           COUNT(*) AS total,
           MODE() WITHIN GROUP (ORDER BY source) AS source
    FROM logs
    GROUP BY alarm_code
    ORDER BY total DESC
    LIMIT %s
"""

_UPSERT_SQL = """
    INSERT INTO alarm_dictionary
        (alarm_code, source_dataset, category, severity, title, description,
         probable_causes, corrective_actions)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (alarm_code) DO UPDATE SET
        title              = EXCLUDED.title,
        description        = EXCLUDED.description,
        category           = EXCLUDED.category,
        severity           = EXCLUDED.severity,
        probable_causes    = EXCLUDED.probable_causes,
        corrective_actions = EXCLUDED.corrective_actions,
        updated_at         = NOW()
"""


def _db_conn() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=int(POSTGRES_PORT),
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


def _call_ollama(code: str, count: int, source: str) -> str:
    prompt = _PROMPT_TEMPLATE.format(code=code, count=count, source=source)
    resp = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def _parse_json(raw: str) -> dict | None:
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _coerce_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def main() -> None:
    conn = _db_conn()
    cur = conn.cursor()

    cur.execute(_SELECT_TOP_CODES, (LIMIT,))
    rows = cur.fetchall()
    total = len(rows)
    logger.info("Top %d alarm_codes carregados do PostgreSQL.", total)

    inserted = 0
    for idx, (code, count, source) in enumerate(rows, start=1):
        try:
            raw = _call_ollama(str(code), int(count), str(source))
            data = _parse_json(raw)
            if data is None:
                raise ValueError("JSON não encontrado na resposta do modelo.")

            title = str(data.get("title", ""))[:60]
            category = str(data.get("category", "other"))
            severity = str(data.get("severity", "info"))
            description = str(data.get("description", ""))
            causes = _coerce_list(data.get("probable_causes", []))
            actions = _coerce_list(data.get("corrective_actions", []))

            cur.execute(
                _UPSERT_SQL,
                (code, source, category, severity, title, description, causes, actions),
            )
            conn.commit()
            inserted += 1
            logger.info(
                "✓ [%d/%d] Alarme %s (%d ocorrências): %s",
                idx, total, code, count, title,
            )

        except Exception as exc:
            conn.rollback()
            logger.error("✗ [%d/%d] Alarme %s: %s", idx, total, code, exc)

    cur.close()
    conn.close()
    logger.info(
        "Dicionário populado: %d/%d alarmes inseridos com sucesso.",
        inserted, total,
    )


if __name__ == "__main__":
    main()
