"""Extrai metadados estruturados da pergunta do técnico para hybrid retrieval.

Determinístico, sem LLM. Reconhece:
- estações citadas (lista 7 estações fixas da Smart Factory)
- timestamp no formato ISO 'YYYY-MM-DD HH:MM:SS[.ffffff]'

Saída usada pelo retriever híbrido para montar `expr` do Milvus
(filtro escalar antes da vector search).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

STATIONS: tuple[str, ...] = (
    "MM_1", "EC_1", "SM_1", "HBW_1", "OV_1", "VGR_1", "WT_1",
)

_STATION_RE = re.compile(r"\b(" + "|".join(STATIONS) + r")\b")
_TIMESTAMP_RE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)\b"
)

# Mesmo formato que pandas usa ao serializar pandas.Timestamp → str:
# sempre 6 dígitos de microssegundos. Bate com o que está no Milvus
# (event_timestamp VARCHAR) e no golden set (evidence_event_ids).
_TS_FMT = "%Y-%m-%d %H:%M:%S.%f"


@dataclass
class QueryMetadata:
    stations: list[str] = field(default_factory=list)
    timestamp: str | None = None
    window_lo: str | None = None
    window_hi: str | None = None


def _normalize_ts(raw: str) -> str:
    """Garante 6 dígitos de microssegundos (formato do Milvus)."""
    if "." not in raw:
        raw = raw + ".000000"
    else:
        head, frac = raw.split(".", 1)
        frac = (frac + "000000")[:6]
        raw = f"{head}.{frac}"
    return raw


def parse_query(query: str, window_seconds: int = 5) -> QueryMetadata:
    """Extrai estações + timestamp da pergunta.

    Args:
        query: texto livre da pergunta.
        window_seconds: meia-janela ao redor do timestamp (default ±5s).

    Returns:
        QueryMetadata. Campos None/[] quando nada é detectado.
    """
    stations: list[str] = []
    seen: set[str] = set()
    for m in _STATION_RE.finditer(query):
        s = m.group(1)
        if s not in seen:
            stations.append(s)
            seen.add(s)

    ts_match = _TIMESTAMP_RE.search(query)
    if not ts_match:
        return QueryMetadata(stations=stations)

    ts_norm = _normalize_ts(ts_match.group(1))
    try:
        dt = datetime.strptime(ts_norm, _TS_FMT)
    except ValueError:
        return QueryMetadata(stations=stations)

    delta = timedelta(seconds=window_seconds)
    lo = (dt - delta).strftime(_TS_FMT)
    hi = (dt + delta).strftime(_TS_FMT)

    return QueryMetadata(
        stations=stations,
        timestamp=ts_norm,
        window_lo=lo,
        window_hi=hi,
    )
