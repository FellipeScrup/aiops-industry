"""Componente lexical (BM25) + fusão RRF para a recuperação híbrida denso+esparso.

Motivação (análise de erro do golden set): embeddings densos não distinguem sufixos numéricos exatos
("ejecting the workpiece to the sink 1" vs "... sink 2"), e é justamente esse tipo de termo literal
que o BM25 captura. A fusão é feita por Reciprocal Rank Fusion (Cormack et al., 2009) com k=60, o
valor padrão da literatura — fixado a priori, sem ajuste sobre o golden set (evita overfitting).

Sem dependências externas: o corpus tem poucas centenas de episódios, então um índice em memória
é suficiente. O IDF é calculado sobre o corpus INTEIRO (não só sobre os candidatos).
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

BM25_K1 = 1.5   # saturação de frequência de termo (valores usuais: 1.2–2.0)
BM25_B = 0.75   # normalização por tamanho do documento
RRF_K = 60      # constante padrão do RRF

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Minúsculas, sem acentos, tokens alfanuméricos (mantém 'hbw_1', 'sink', '1', ...)."""
    norm = unicodedata.normalize("NFKD", text.lower())
    norm = "".join(c for c in norm if not unicodedata.combining(c))
    return _TOKEN_RE.findall(norm)


def hit_text(hit: dict) -> str:
    """Texto indexado de um episódio (mesmos campos que o gerador enxerga)."""
    return " ".join(str(hit.get(k) or "") for k in ("station", "current_task", "current_sub_task", "log_text"))


class BM25Index:
    """Estatísticas de corpus (df, tamanho médio) para pontuar documentos arbitrários."""

    def __init__(self, corpus_texts: list[str]) -> None:
        docs = [tokenize(t) for t in corpus_texts]
        self.n_docs = max(len(docs), 1)
        self.avg_len = (sum(len(d) for d in docs) / len(docs)) if docs else 1.0
        self.df: Counter[str] = Counter()
        for d in docs:
            self.df.update(set(d))

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1.0 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not doc_tokens:
            return 0.0
        tf = Counter(doc_tokens)
        norm = BM25_K1 * (1.0 - BM25_B + BM25_B * len(doc_tokens) / self.avg_len)
        total = 0.0
        for term in set(query_tokens):
            f = tf.get(term, 0)
            if f:
                total += self.idf(term) * f * (BM25_K1 + 1.0) / (f + norm)
        return total


def rerank_bm25(query: str, hits: list[dict], index: BM25Index) -> list[dict]:
    """Reordena os candidatos fundindo o ranking denso (ordem de entrada) com o ranking BM25 por RRF.

    `score` (similaridade cosseno) é preservado; `bm25` e `rrf_score` são acrescentados.
    Empates no RRF mantêm a ordem densa original (sort estável).
    """
    if len(hits) < 2:
        return hits
    q_tokens = tokenize(query)
    bm25 = [index.score(q_tokens, tokenize(hit_text(h))) for h in hits]
    bm25_order = sorted(range(len(hits)), key=lambda i: bm25[i], reverse=True)
    bm25_rank = {i: r for r, i in enumerate(bm25_order, start=1)}
    fused = []
    for dense_rank, h in enumerate(hits, start=1):
        i = dense_rank - 1
        rrf = 1.0 / (RRF_K + dense_rank) + 1.0 / (RRF_K + bm25_rank[i])
        fused.append({**h, "bm25": round(bm25[i], 4), "rrf_score": rrf})
    return sorted(fused, key=lambda h: h["rrf_score"], reverse=True)
