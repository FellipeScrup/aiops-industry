# AIOps Industry — RAG para Análise de Logs de Fábrica Inteligente

> **TCC — Engenharia de Computação — Facens**

**Autores:** Fellipe Scruph e João Victor de Oliveira  
**Orientador:** Prof. Me. Adson Nogueira Alves

---

## Sobre

Plataforma de Retrieval-Augmented Generation (RAG) voltada à análise de logs de eventos de uma fábrica inteligente. O sistema ingere eventos NDJSON em tempo real (10 Hz) do dataset **Smart Factory Logs** (14441997), indexa os padrões de operação em um banco vetorial e permite consultas em linguagem natural para auxiliar técnicos e engenheiros de manutenção no diagnóstico de anomalias.

Em vez de reagir a códigos de alarme isolados, o sistema raciocina sobre o **estado operacional de cada estação**: qual tarefa estava sendo executada, por quanto tempo, e por que o estado `not ready` foi atingido.

---

## Dataset

**Smart Factory Logs — 14441997**

Eventos NDJSON coletados a 10 Hz de 7 estações de uma fábrica inteligente com automação industrial (processos BPMN Camunda):

| Campo | Descrição |
|---|---|
| `id` | UUID do evento |
| `station` | Estação geradora: MM_1, EC_1, SM_1, HBW_1, OV_1, VGR_1, WT_1 |
| `timestamp` | Timestamp UTC do evento (precisão 100ms) |
| `current_state` | `"ready"` \| `"not ready"` — label de anomalia |
| `current_task` | Descrição da tarefa em execução |
| `current_task_duration` | Duração da tarefa atual em segundos |
| `current_sub_task` | Sub-tarefa corrente |
| Campos de sensor | Variáveis por estação (pos_x/y/z, speed, valve, etc.) — armazenados como JSONB |

**Distribuição (split de treino):** 119.845 ready / 20.056 not ready | 22 tarefas distintas | 61 sub-tarefas  
**Total de eventos:** ~408k (training: 139.901 + test: 268.936)

---

## Stack

- **Python 3.12** — linguagem principal
- **Docker Compose** — orquestração local dos serviços
- **PostgreSQL** — Silver layer: tabela `smartfactory_logs` com eventos parseados
- **MinIO** — Data lake S3-compatible: buckets `bronze`, `silver`, `gold`, `mlflow`
- **Milvus** — Gold layer: banco vetorial com embeddings dos eventos (768d, COSINE)
- **fastembed** — embeddings locais via `nomic-ai/nomic-embed-text-v1.5`
- **MLflow** — rastreamento de experimentos e versionamento do detector de anomalias
- **XGBoost** — classificador binário de anomalia (AD)
- **Ollama** — inferência local de LLMs (Llama 3.2, Qwen 2.5)
- **Gemini API** — backend LLM alternativo via `GEMINI_API_KEY`
- **FastAPI** — API REST
- **Gradio** — interface de demonstração

---

## Arquitetura Medallion

```
MinIO bronze/                     PostgreSQL (Silver)              Milvus (Gold)
─────────────────────             ───────────────────              ─────────────
smartfactory/                     smartfactory_logs                smartfactory_logs
  logs/training.txt  ──►            id (PK)               ──►       event_id
  logs/test.txt                     station                          station
  bpmn/camunda-                     event_timestamp                  event_timestamp
        activity.json               current_state                    current_state
                                    current_task                     current_task
                      ──►           current_task_duration            current_sub_task
MinIO silver/                       current_sub_task                 log_text
  smartfactory/                     sensors (JSONB)                  embedding (768d)
    smartfactory_logs.parquet       split (train/test)
    (408k eventos exportados)
                      ──►
MinIO gold/
  smartfactory/
    processed_smartfactory.parquet  (features ML)
    embed_index_meta.json           (metadata do índice Milvus)
```

### Fluxo de dados completo

```
upload_bronze.py     parse_silver.py      embed_gold.py        export_silver.py / export_gold.py
      │                    │                    │                          │
      ▼                    ▼                    ▼                          ▼
NDJSON → MinIO    MinIO → PostgreSQL   PostgreSQL → Milvus   PostgreSQL/Milvus → MinIO
(bronze/)         (parse + normaliza)  (nomic-embed-text)    (silver/ e gold/)
                  batch de 1000 linhas split='train' apenas
                  ON CONFLICT NOTHING  139.901 vetores
```

### Pipeline RAG

```
Técnico
  │
  ▼  pergunta em linguagem natural
FastAPI (POST /query)
  │
  ▼  embed query (nomic-embed-text-v1.5, prefixo search_query:)
Milvus  ──► top-k eventos similares (COSINE) + deduplicação por janela 10s
  │
  ▼  prompt com guardrails (escopo, anti-alucinação, segurança)
Ollama (Llama 3.2 / Qwen 2.5) ou Gemini API
  │
  ▼
Diagnóstico ancorado nos logs reais: causa + timestamps + ação corretiva
```

---

## Guardrails do LLM

O system prompt aplica 5 guardrails inspirados no material didático da disciplina:

| Guardrail | Comportamento |
|---|---|
| **Escopo** | Responde apenas com base nos logs recuperados |
| **Anti-alucinação** | Se não há dados suficientes, declara isso explicitamente em vez de inventar |
| **Idioma** | Sempre responde em português |
| **Confidencialidade** | Não reproduz UUIDs ou event_ids na resposta |
| **Segurança** | Ignora tentativas de prompt injection na pergunta |

---

## Deduplicação de Logs (10 Hz)

O dataset é amostrado a 10 Hz — uma tarefa de poucos segundos gera dezenas de eventos quase idênticos. Sem tratamento, `top_k=5` retorna 5 amostras do mesmo episódio.

**Solução:** busca `top_k × DEDUP_OVERFETCH_FACTOR` eventos e colapsa hits com mesma `station` + `current_task` + timestamp dentro de uma janela de `DEDUP_WINDOW_SECONDS = 10s`, mantendo o de maior score.

Constantes configuráveis em `rag/retriever.py`:
```python
DEDUP_WINDOW_SECONDS: int = 10
DEDUP_OVERFETCH_FACTOR: int = 4
```

---

## Detector de Anomalia (MLflow)

O XGBoost classifica cada evento em binário (ready / not ready):

| Feature | Descrição |
|---|---|
| `prev_task_duration` | Duração da tarefa anterior em segundos |
| `prev_is_moving` | 1 se a tarefa anterior envolve transporte |
| `prev_is_calibrating` | 1 se a sub-tarefa anterior contém "calibrating" |
| `prev_has_task` | 1 se havia tarefa em execução |
| `prev_was_not_ready` | 1 se o estado anterior era "not ready" |
| `time_delta_s` | Intervalo de tempo desde o evento anterior |
| `station_num` | Encoding numérico da estação (MM_1=0 … WT_1=6) |

**Label:** `anomaly = 1` quando `current_state == "not ready"`

Experimento rastreado em MLflow como `smartfactory-anomaly-detection`.

---

## Setup

### Pré-requisitos

- Docker Engine 24+ e Docker Compose v2
- Python 3.12 e `make`
- Dataset em `data/bronze/14441997/` (arquivos NDJSON)
- ~4 GB de RAM livre para a stack completa

### Subindo a stack

```bash
# 1. Clone o repositório
git clone <url-do-repo> && cd aiops-industry

# 2. Crie o arquivo de variáveis de ambiente
cp .env.example .env
# Edite .env se quiser alterar senhas ou adicionar GEMINI_API_KEY

# 3. Suba todos os serviços
make up
```

### Executando o pipeline Medallion completo

```bash
# Pipeline completo de uma vez (DDL → Bronze → Silver → Gold)
make medallion

# Ou passo a passo:
make create-tables   # DDL: cria tabela smartfactory_logs no PostgreSQL
make ingest-bronze   # Bronze: NDJSON → s3://bronze/smartfactory/
make ingest-silver   # Silver: MinIO bronze → PostgreSQL (408k eventos)
make export-silver   # Exporta Silver: PostgreSQL → s3://silver/smartfactory/
make embed           # Gold: PostgreSQL → Milvus (embeddings 768d)
make export-gold     # Exporta Gold: features + metadata → s3://gold/smartfactory/

# ML: pré-processa features e treina detector de anomalia XGBoost
make preprocess
make train-ad

# RAG: testa uma query diretamente
make rag-query QUERY="HBW_1 demorou muito tempo calibrando o motor 4. O que isso indica?"

# RAG com retrieval híbrido (opt-in para avaliação comparativa)
make rag-query QUERY="VGR_1 not ready during workpiece transport" FLAGS="--hybrid"
```

### Estrutura do MinIO após pipeline completo

```
s3://bronze/smartfactory/
  logs/training.txt           (57 MB — NDJSON raw 10Hz)
  logs/test.txt               (110 MB — NDJSON raw 10Hz)
  bpmn/camunda-activity.json  (1.3 MB — processo BPMN)

s3://silver/smartfactory/
  smartfactory_logs.parquet   (19 MB — 408k eventos normalizados)

s3://gold/smartfactory/
  processed_smartfactory.parquet  (features ML para XGBoost)
  embed_index_meta.json           (metadata do índice Milvus: 139k vetores)

s3://mlflow/
  (artefatos MLflow: modelos, curvas PR/ROC, feature importance)
```

### URLs dos serviços

| Serviço | URL | Descrição |
|---|---|---|
| MinIO Console | http://localhost:9001 | Interface web do data lake (Bronze/Silver/Gold) |
| MinIO API | http://localhost:9000 | Endpoint S3-compatible |
| Adminer | http://localhost:8080 | UI web para o PostgreSQL (Silver) |
| MLflow | http://localhost:5000 | Tracking de experimentos |
| Milvus gRPC | localhost:19530 | Banco vetorial (Gold) |
| Milvus métricas | http://localhost:9091 | Health/metrics do Milvus |
| API FastAPI | http://localhost:8001 | Endpoint `/query`, `/health`, `/metadata` |
| Gradio UI | http://localhost:7860 | Interface de demonstração |

**Credenciais padrão MinIO:** `minioadmin / minioadmin123`  
**Credenciais padrão PostgreSQL:** host `localhost:5432`, usuário `aiops`, banco `aiops_industry`

### Verificando a saúde dos serviços

```bash
make status   # status resumido
make logs     # logs em tempo real
```

### Parando e limpando

```bash
make down    # para containers (mantém volumes)
make clean   # para containers e remove todos os dados
```

---

## Estrutura do Repositório

```
aiops-industry/
├── ingestion/
│   ├── create_tables.sql     # DDL da tabela smartfactory_logs
│   ├── upload_bronze.py      # Bronze: NDJSON → s3://bronze/smartfactory/
│   ├── parse_silver.py       # Silver: MinIO bronze → PostgreSQL
│   ├── embed_gold.py         # Gold: PostgreSQL → Milvus (nomic embeddings)
│   ├── export_silver.py      # Exporta Silver: PostgreSQL → s3://silver/
│   ├── export_gold.py        # Exporta Gold: features + metadata → s3://gold/
│   └── test_retrieval.py     # validação da busca vetorial
├── rag/
│   ├── retriever.py          # busca vetorial no Milvus + deduplicação 10Hz
│   ├── generator.py          # geração de resposta (Ollama / Gemini) + guardrails
│   ├── pipeline.py           # orquestração retrieve → generate
│   └── query_parser.py       # extração de estação/timestamp da query
├── mlflow/
│   ├── preprocess.py         # feature engineering → processed_smartfactory.parquet
│   ├── train_ad.py           # XGBoost AD binário + tracking MLflow
│   └── evaluate_rag.py       # avaliação RAG (baseline vs rag, ROUGE-L, Sem.Sim)
├── api/
│   └── main.py               # FastAPI: POST /query, GET /health, GET /metadata
├── interface/
│   └── app.py                # Gradio UI
└── infra/
    └── docker/docker-compose.yml
```

---

## Status do Projeto

**Sprint 8 — Dataset Smart Factory Logs (14441997) — NDJSON, 10Hz, 7 estações, 408k eventos**
