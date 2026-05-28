COMPOSE := docker compose -f infra/docker/docker-compose.yml --env-file .env

.DEFAULT_GOAL := help

.PHONY: help up down logs ps restart clean status \
        create-tables ingest-bronze ingest-silver ingest-all \
        ingest-episodes export-silver medallion \
        embed test-retrieval \
        rag-query \
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

# ── API + Interface ───────────────────────────────────────────────────────────

api: ## Sobe a FastAPI em localhost:8001
	PYTHONPATH=. python -m uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload

ui: ## Sobe a interface Gradio em localhost:7860
	PYTHONPATH=. python interface/app.py

serve: ## Sobe API + UI juntos (2 processos em background)
	make api & make ui
