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
- **MinIO** — Bronze layer: armazenamento dos arquivos NDJSON brutos
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
MinIO (Bronze)                    PostgreSQL (Silver)              Milvus (Gold)
──────────────────                ───────────────────              ─────────────
smartfactory/                     smartfactory_logs                smartfactory_logs
  logs/training.txt  ──►            id (PK)               ──►       event_id
  logs/test.txt                     station                          station
  bpmn/camunda-      
        activity.json               event_timestamp                  event_timestamp
                                    current_state                    current_state
                                    current_task                     current_task
                                    current_task_duration            current_sub_task
                                    current_sub_task                 log_text
                                    sensors (JSONB)                  embedding (768d)
                                    split (train/test)
```

### Fluxo de dados

```
upload_bronze.py          parse_silver.py           embed_gold.py
     │                         │                         │
     ▼                         ▼                         ▼
NDJSON → MinIO       MinIO → PostgreSQL         PostgreSQL → Milvus
(bronze)             (parse + normaliza)         (nomic-embed-text-v1.5)
                     batch de 1000 linhas        split='train' apenas
                     ON CONFLICT DO NOTHING      EMBED_LIMIT=50000
```

### Pipeline RAG

```
Técnico
  │
  ▼  POST /query
FastAPI
  │
  ▼  embed query (nomic-embed-text-v1.5, prefixo search_query:)
Milvus  ──► top-k eventos de log similares (COSINE)
  │
  ▼  prompt template (engenheiro de fábrica inteligente)
Ollama / Gemini
  │
  ▼
Diagnóstico: causa do estado not ready + ação corretiva
```

---

## Detector de Anomalia (MLflow)

O XGBoost classifica cada evento em binário (ready / not ready):

| Feature | Descrição |
|---|---|
| `task_duration` | Duração da tarefa atual em segundos |
| `is_moving` | 1 se a tarefa envolve "transporting" ou "moving" |
| `is_calibrating` | 1 se a sub-tarefa contém "calibrating" |
| `station_num` | Encoding numérico da estação (MM_1=0 … WT_1=6) |
| `has_task` | 1 se `current_task` não está vazio |

**Label:** `anomaly = 1` quando `current_state == "not ready"`

Experimento rastreado em MLflow como `smartfactory-anomaly-detection`.

---

## Setup

### Pré-requisitos

- Docker Engine 24+ e Docker Compose v2
- Python 3.12 e `make`
- Dataset em `../14441997/` (pasta irmã do projeto)
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

### Executando o pipeline completo

```bash
# DDL: cria tabela smartfactory_logs no PostgreSQL
make create-tables

# Bronze: envia NDJSON brutos para o MinIO (bucket smartfactory)
make ingest-bronze

# Silver: parseia NDJSON e persiste no PostgreSQL (408k eventos)
make ingest-silver

# Gold: gera embeddings e indexa 50k eventos no Milvus
make embed

# ML: pré-processa features e treina detector de anomalia XGBoost
make preprocess
make train-ad

# RAG: testa uma query diretamente
make rag-query QUERY="VGR_1 not ready during workpiece transport"
```

### URLs dos serviços

| Serviço | URL | Descrição |
|---|---|---|
| MinIO Console | http://localhost:9001 | Interface web do data lake (Bronze) |
| MinIO API | http://localhost:9000 | Endpoint S3-compatible |
| Adminer | http://localhost:8080 | UI web para o PostgreSQL (Silver) |
| MLflow | http://localhost:5000 | Tracking de experimentos |
| Milvus gRPC | localhost:19530 | Banco vetorial (Gold) |
| Milvus métricas | http://localhost:9091 | Health/metrics do Milvus |
| API FastAPI | http://localhost:8001 | Endpoint `/query` e `/health` |
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
│   ├── upload_bronze.py      # Bronze: NDJSON → MinIO (bucket smartfactory)
│   ├── parse_silver.py       # Silver: MinIO → PostgreSQL (smartfactory_logs)
│   ├── embed_gold.py         # Gold: PostgreSQL → Milvus (nomic embeddings)
│   └── test_retrieval.py     # validação da busca vetorial
├── rag/
│   ├── retriever.py          # busca vetorial no Milvus
│   ├── generator.py          # geração de resposta (Ollama / Gemini)
│   └── pipeline.py           # orquestração retrieve → generate
├── mlflow/
│   ├── preprocess.py         # feature engineering → processed_smartfactory.parquet
│   ├── train_ad.py           # XGBoost AD binário + tracking MLflow
│   └── evaluate_rag.py       # avaliação RAG (baseline vs rag, ROUGE-L, Sem.Sim)
├── api/
│   └── main.py               # FastAPI: POST /query
├── interface/
│   └── app.py                # Gradio UI
└── infra/
    └── docker/docker-compose.yml
```

---

## Status do Projeto

**Sprint 8 — Dataset Smart Factory Logs (14441997) — NDJSON, 10Hz, 7 estações, 408k eventos**
