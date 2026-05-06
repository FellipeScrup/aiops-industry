COMPOSE := docker compose -f infra/docker/docker-compose.yml --env-file .env

.DEFAULT_GOAL := help

.PHONY: help up down logs ps restart clean status \
        create-tables ingest-bronze ingest-silver ingest-all \
        preprocess train

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

create-tables: ## Cria tabelas logs e alarm_dictionary no PostgreSQL
	@set -a && . ./.env && set +a && \
	docker exec -i aiops-postgres psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB" \
	  < ingestion/create_tables.sql && \
	echo "  Tabelas criadas."

ingest-bronze: ## Sobe CSVs brutos para o MinIO (Bronze layer)
	python ingestion/upload_bronze.py

ingest-silver: ## Processa Bronze → Silver (normaliza e persiste no PostgreSQL)
	python ingestion/parse_silver.py

ingest-all: create-tables ingest-bronze ingest-silver ## Pipeline completo: tabelas → Bronze → Silver

# ── Modelagem ML ─────────────────────────────────────────────────────────────

preprocess: ## Pré-processa os dados Silver para ML (gera data/silver/processed_logs.parquet)
	pip install -q -r mlflow/requirements.txt && python mlflow/preprocess.py

train: preprocess ## Treina XGBoost após pré-processamento e registra no MLflow
	MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 python mlflow/train.py
