COMPOSE := docker compose -f infra/docker/docker-compose.yml --env-file .env

.DEFAULT_GOAL := help

.PHONY: help up down logs ps restart clean status \
        create-tables ingest-bronze ingest-silver ingest-all \
        ingest-episodes export-silver medallion \
        embed test-retrieval \
        rag-query \
        golden-set eval-generate eval-ragas eval-report eval-compare eval \
        api ui serve

help: ## Exibe esta mensagem de ajuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

up: ## Sobe todos os serviços em modo detached
	$(COMPOSE) up -d

down: ## Para e remove os containers (volumes preservados)
	$(COMPOSE) down

logs: ## Acompanha os logs de todos os serviços (últimas 100 linhas)
	$(COMPOSE) logs -f --tail=100

ps: ## Lista os containers e seus estados
	$(COMPOSE) ps

restart: ## Reinicia todos os serviços (down + up)
	$(MAKE) down
	$(MAKE) up

clean: ## Para os containers E remove todos os volumes de dados (destrutivo)
	@read -p "ATENÇÃO: isso apaga todos os volumes de dados. Confirma? [y/N] " r; \
	if [ "$$r" = "y" ] || [ "$$r" = "Y" ]; then \
		$(COMPOSE) down -v; \
		echo "Volumes removidos."; \
	else \
		echo "Operação cancelada."; \
	fi

status: ## Exibe status dos serviços com portas expostas
	@$(COMPOSE) ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

# ── Ingestão Medallion ──────────────────────────────────────────────────────

create-tables: ## Cria tabela smartfactory_logs no PostgreSQL
	@set -a && . ./.env && set +a && \
	docker exec -i aiops-postgres psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB" \
	  < ingestion/create_tables.sql && \
	echo "  Tabelas criadas."

ingest-bronze: ## Sobe CSVs brutos para o MinIO (Bronze layer)
	python ingestion/upload_bronze.py

ingest-silver: ## Processa Bronze → Silver (normaliza e persiste no PostgreSQL)
	python ingestion/parse_silver.py

ingest-all: create-tables ingest-bronze ingest-silver ## Pipeline completo: DDL → Bronze (MinIO) → Silver (PostgreSQL)

ingest-episodes: ## Silver → Gold: agrega eventos 10Hz em episódios (tabela smartfactory_episodes)
	python ingestion/parse_episodes.py

export-silver: ## Exporta Silver: PostgreSQL → Parquet → s3://silver/smartfactory/
	python ingestion/export_silver.py

medallion: ingest-all ingest-episodes export-silver embed ## Pipeline Medallion completo: Bronze → Silver → Gold (MinIO + Milvus)

# ── Embeddings Gold ──────────────────────────────────────────────────────────

embed: ## Embeds episódios (smartfactory_episodes) e indexa no Milvus (sempre reconstrói a collection)
	python ingestion/embed_gold.py

test-retrieval: ## Testa busca vetorial no Milvus (uso: make test-retrieval QUERY="falha motor")
	python ingestion/test_retrieval.py "$(QUERY)"

# ── RAG Core ─────────────────────────────────────────────────────────────────

rag-query: ## Consulta o RAG (uso: make rag-query QUERY="..." [FLAGS="--hybrid"])
	PYTHONPATH=. python rag/pipeline.py "$(QUERY)" $(FLAGS)

# ── Avaliação (RAGAS + métricas de recuperação) ──────────────────────────────
# Requer o venv isolado: python -m venv .venv-eval && .venv-eval/Scripts/pip install -r requirements.txt -r evaluation/requirements.txt
EVAL_PY ?= .venv-eval/Scripts/python
# MODEL = gerador avaliado (resultados em evaluation/results/<modelo>/); JUDGE = juiz do RAGAS
# (use OUTRA família que o gerador). Ex.: make eval-generate MODEL=qwen3.5:9b
MODEL ?= qwen2.5:7b
JUDGE ?= phi4

golden-set: ## Gera o golden set objetivo (24 perguntas) a partir do PostgreSQL
	PYTHONPATH=. $(EVAL_PY) evaluation/golden_set.py

eval-generate: ## Executa baseline, RAG vetorial e RAG com filtragem sobre o golden set (retomável; MODEL=...)
	PYTHONPATH=. $(EVAL_PY) evaluation/run_eval.py generate --model $(MODEL)

eval-ragas: ## Calcula as métricas RAGAS com juiz LLM local de outra família (retomável; MODEL=... JUDGE=...)
	PYTHONPATH=. $(EVAL_PY) evaluation/run_eval.py ragas --model $(MODEL) --judge $(JUDGE)

eval-report: ## Agrega resultados (IC bootstrap), gera tabelas e figuras (MODEL=... JUDGE=...)
	PYTHONPATH=. $(EVAL_PY) evaluation/run_eval.py report --model $(MODEL) --judge $(JUDGE)

eval-compare: ## Compara lado a lado todos os geradores já avaliados (com diferença pareada)
	PYTHONPATH=. $(EVAL_PY) evaluation/run_eval.py compare

eval: golden-set eval-generate eval-ragas eval-report ## Avaliação completa de um gerador (MODEL=... JUDGE=...)

# ── API + Interface ───────────────────────────────────────────────────────────

api: ## Sobe a FastAPI em localhost:8001
	PYTHONPATH=. python -m uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload

ui: ## Sobe a interface Gradio em localhost:7860
	PYTHONPATH=. python interface/app.py

serve: ## Sobe API + UI juntos (2 processos em background)
	make api & make ui
