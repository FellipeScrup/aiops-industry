"""Sprint 7 — FastAPI: REST interface for the RAG pipeline."""

import logging
import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.pipeline import query as rag_query

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AIOps Industry — RAG API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    question: str
    answer: str
    context: list[dict]
    retrieval_scores: list[float]
    processing_time_s: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
def query_endpoint(body: QueryRequest) -> QueryResponse:
    logger.info("POST /query — question=%r top_k=%d", body.question, body.top_k)
    t0 = time.perf_counter()

    try:
        result = rag_query(body.question, top_k=body.top_k)
    except ConnectionError as exc:
        logger.error("Serviço indisponível: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))

    elapsed = round(time.perf_counter() - t0, 3)
    logger.info("POST /query concluído em %.2fs", elapsed)

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        context=result["context"],
        retrieval_scores=result["retrieval_scores"],
        processing_time_s=elapsed,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "1.0.0"}


@app.get("/metadata")
def metadata() -> dict:
    return {
        "model": os.getenv("LLM_MODEL", "llama3.2:3b"),
        "embed_model": "nomic-embed-text",
        "vector_db": "milvus",
        "total_vectors": 50501,
        "datasets": ["ALPI", "PIADE"],
    }
