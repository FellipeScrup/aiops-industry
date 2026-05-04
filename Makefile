COMPOSE := docker compose -f infra/docker/docker-compose.yml --env-file .env

.DEFAULT_GOAL := help

.PHONY: help up down logs ps restart clean status

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
