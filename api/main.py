"""Sprint 7 — FastAPI: REST interface for the RAG pipeline."""

import logging
import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.pipeline import query as rag_query
from rag.retriever import COLLECTION_NAME, EMBED_MODEL_NAME

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AIOps Industry — RAG API",
    version="1.1.0",
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
    model: str = Field(default="qwen2.5:7b")


class QueryResponse(BaseModel):
    question: str
    answer: str
    context: list[dict]
    retrieval_scores: list[float]
    processing_time_s: float
    model_used: str
    token_usage: dict


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
def query_endpoint(body: QueryRequest) -> QueryResponse:
    logger.info(
        "POST /query — question=%r top_k=%d model=%s",
        body.question, body.top_k, body.model,
    )
    t0 = time.perf_counter()

    try:
        # use_hybrid=True: quando a pergunta cita estação (ex.: "SM_1"), filtra
        # no Milvus por station antes da busca vetorial. Quando não cita,
        # cai automaticamente para busca vetorial pura — estritamente melhor.
        result = rag_query(
            body.question, top_k=body.top_k, model=body.model, use_hybrid=True,
        )
    except ConnectionError as exc:
        logger.error("Serviço indisponível: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except (ValueError, ImportError) as exc:
        logger.error("Erro de configuração/modelo: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        logger.error("Erro no LLM: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc))

    elapsed = round(time.perf_counter() - t0, 3)
    logger.info("POST /query concluído em %.2fs", elapsed)

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        context=result["context"],
        retrieval_scores=result["retrieval_scores"],
        processing_time_s=elapsed,
        model_used=result["model_used"],
        token_usage=result.get("token_usage", {}),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "1.1.0"}


@app.get("/metadata")
def metadata() -> dict:
    return {
        "model": os.getenv("LLM_MODEL", "qwen2.5:7b"),
        "available_models": ["qwen2.5:7b", "llama3.2:3b", "gemini"],
        "embed_model": EMBED_MODEL_NAME,
        "vector_db": "milvus",
        "collection": COLLECTION_NAME,
        "datasets": ["Smart Factory Logs (14441997)"],
    }
