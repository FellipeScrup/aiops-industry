COMPOSE := docker compose -f infra/docker/docker-compose.yml --env-file .env

.DEFAULT_GOAL := help

.PHONY: help up down logs ps restart clean status \
        create-tables ingest-bronze ingest-silver ingest-all \
        preprocess train-ad eval-rag eval-rag-gen \
        golden-set validate-golden \
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

# ── Modelagem ML ─────────────────────────────────────────────────────────────

preprocess: ## Pré-processa dados Silver para ML (gera data/silver/processed_smartfactory.parquet)
	python -m pip install -q -r mlflow/requirements.txt && python mlflow/preprocess.py

train-ad: ## Treina XGBoost AD binário e registra métricas do paper (AP, ROC AUC, fit/predict time)
	MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 python mlflow/train_ad.py

eval-rag: ## Avalia RAG (baseline vs rag) e loga no MLflow (uso: make eval-rag)
	python -m pip install -q -r mlflow/requirements.txt && \
	python mlflow/evaluate_rag.py

eval-rag-gen: ## Só (re)gera o golden set legado (top-25 anomalias), sem avaliar
	python mlflow/evaluate_rag.py --force-gen --skip-eval

golden-set: ## Gera golden set tiered (factual + cross_station + causal) — Passo 0
	python mlflow/evaluate_rag.py --tiered --force-gen --skip-eval

validate-golden: ## Valida schema, distribuição e evidências do golden_set.json
	python mlflow/validate_golden_set.py

# ── Embeddings Gold ──────────────────────────────────────────────────────────

embed: ## Gera embeddings e indexa no Milvus (Gold layer)
	EMBED_LIMIT=50000 python ingestion/embed_gold.py

test-retrieval: ## Testa busca vetorial no Milvus (uso: make test-retrieval QUERY="falha motor")
	python ingestion/test_retrieval.py "$(QUERY)"

# ── RAG Core ─────────────────────────────────────────────────────────────────

rag-query: ## Consulta o RAG (uso: make rag-query QUERY="máquina s_1 com alto downtime")
	PYTHONPATH=. python rag/pipeline.py "$(QUERY)"

# ── API + Interface ───────────────────────────────────────────────────────────

api: ## Sobe a FastAPI em localhost:8001
	PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload

ui: ## Sobe a interface Gradio em localhost:7860
	PYTHONPATH=. python interface/app.py

serve: ## Sobe API + UI juntos (2 processos em background)
	make api & make ui
