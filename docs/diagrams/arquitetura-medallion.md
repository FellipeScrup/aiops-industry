# Arquitetura — AIOps Industry RAG Industrial

## Visão Geral

A plataforma segue o padrão **Lakehouse + Medallion** (Armbrust et al., 2021) com três camadas de qualidade progressiva de dados, alimentadas exclusivamente pelo dataset **PIADE sequences_1h_data**.

## Camadas Medallion

### Bronze — MinIO (Data Lake)
- Dados brutos sem transformação
- Arquivo: `piade/sequences_1h_data.csv`
- Janelas de 1 hora por equipamento com percentuais de tempo de operação, contagens de alarme e mudanças de estado

### Silver — PostgreSQL (Banco Relacional)
- Dados normalizados e persistidos por `parse_silver.py`
- Tabela única: `piade_telemetry`
  - `equipment_id`, `interval_start` (chave composta única)
  - `count_sum`, `num_changes`
  - `pct_idle`, `pct_production`, `pct_downtime`, `pct_perf_loss`, `pct_sched_downtime`

### Gold — Milvus (Banco Vetorial)
- Embeddings de 768 dimensões via `nomic-ai/nomic-embed-text-v1.5` (fastembed)
- Collection: `piade_telemetry`
- Campos armazenados: `machine_id`, `interval_start`, `pct_idle`, `pct_downtime`, `pct_perf_loss`, `count_sum`, `log_text`
- Busca por similaridade de cosseno (IVF_FLAT, nlist=128, nprobe=16)
- Texto de embedding: descrição narrativa da janela de 1h (downtime%, idle%, perf_loss%, alarmes, mudanças de estado)

## Fluxo RAG

1. Técnico insere descrição do comportamento da máquina na interface Gradio ou via `POST /query`
2. FastAPI repassa para o pipeline RAG
3. Retriever embeda a query com prefixo `search_query:` e busca as top-k janelas mais similares no Milvus
4. Generator monta o prompt com o contexto de telemetria e envia ao LLM (Ollama local ou Gemini API)
5. O LLM atua como engenheiro de confiabilidade: diagnostica o comportamento, aponta causa provável e recomenda ação corretiva

## Classificador ML (XGBoost + MLflow)

- Features: `pct_idle`, `pct_production`, `pct_downtime`, `pct_perf_loss`, `pct_sched_downtime`, `count_sum`, `num_changes`
- Target: nível de degradação derivado de thresholds de telemetria
  - `normal` (0): downtime ≤ 5% e perf_loss ≤ 10%
  - `degraded` (1): downtime > 5% ou perf_loss > 10%
  - `critical` (2): downtime > 15% ou perf_loss > 20%
- Experimento: `piade-degradation-classification`
- Modelo salvo localmente em `mlflow/models/xgboost_severity.json`

## Infraestrutura

- Tudo containerizado via `docker compose up`
- Tracking de experimentos: MLflow (backend PostgreSQL, artefatos MinIO)
- LLMs locais via Ollama (Llama 3.2:3b, Qwen 2.5)
- LLM remoto opcional via Gemini API (`model=gemini`)
