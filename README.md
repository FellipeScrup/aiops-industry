# AIOps Industry — RAG para Análise de Telemetria Industrial

> **TCC — Engenharia de Computação — Facens**

**Autores:** Fellipe Scruph e João Victor de Oliveira  
**Orientador:** Prof. Me. Adson Nogueira Alves

---

## Sobre

Plataforma de Retrieval-Augmented Generation (RAG) voltada à análise do comportamento de equipamentos industriais de embalagem. O sistema ingere janelas de telemetria de 1 hora do dataset **PIADE** (`sequences_1h_data`), indexa os padrões de operação em um banco vetorial e permite consultas em linguagem natural para auxiliar técnicos e engenheiros de confiabilidade no diagnóstico de degradações de desempenho.

Em vez de reagir a códigos de alarme isolados, o sistema raciocina sobre o **estado agregado da máquina**: quanto tempo ela ficou parada, quanto perdeu de performance, quantos eventos ocorreram — e por quê isso pode ter acontecido.

---

## Dataset

**PIADE — sequences_1h_data.csv**

Janelas de 1 hora por equipamento com as seguintes features principais:

| Coluna | Descrição |
|---|---|
| `equipment_ID` | Identificador da máquina (ex: `s_1`) |
| `interval_start` | Início da janela temporal |
| `count_sum` | Total de ocorrências de alarme na janela |
| `#changes` | Número de mudanças de estado operacional |
| `%idle` | Proporção de tempo em idle |
| `%production` | Proporção de tempo em produção |
| `%downtime` | Proporção de tempo em downtime não planejado |
| `%performance_loss` | Proporção de tempo com perda de performance |
| `%scheduled_downtime` | Proporção de tempo em parada planejada |

---

## Stack

- **Python** — linguagem principal
- **Docker Compose** — orquestração local dos serviços
- **PostgreSQL** — Silver layer: tabela `piade_telemetry` com janelas de 1h
- **MinIO** — Bronze layer: armazenamento do CSV bruto
- **Milvus** — Gold layer: banco vetorial com embeddings das janelas de telemetria
- **MLflow** — rastreamento de experimentos e versionamento do classificador
- **Ollama** — inferência local de LLMs (Llama 3.2, Qwen 2.5)
- **Gemini API** — backend LLM alternativo via `GEMINI_API_KEY`
- **FastAPI** — API REST
- **Gradio** — interface de demonstração

---

## Arquitetura Medallion

```
MinIO (Bronze)                PostgreSQL (Silver)           Milvus (Gold)
─────────────────             ───────────────────           ─────────────
piade/                        piade_telemetry               piade_telemetry
  sequences_1h_data.csv  ──►    equipment_id           ──►   machine_id
                                interval_start                interval_start
                                count_sum                     pct_idle
                                num_changes                   pct_downtime
                                pct_idle                      pct_perf_loss
                                pct_production                count_sum
                                pct_downtime                  log_text
                                pct_perf_loss                 embedding (768d)
                                pct_sched_downtime
```

### Fluxo de dados

```
upload_bronze.py         parse_silver.py          embed_gold.py
     │                        │                        │
     ▼                        ▼                        ▼
CSV → MinIO         MinIO → PostgreSQL        PostgreSQL → Milvus
(bronze)            (janelas 1h normalizadas)  (nomic-embed-text-v1.5)
```

### Pipeline RAG

```
Técnico
  │
  ▼  POST /query
FastAPI
  │
  ▼  embed query (nomic-embed-text-v1.5)
Milvus  ──► top-k janelas de telemetria similares
  │
  ▼  prompt template (engenheiro de confiabilidade)
Ollama / Gemini
  │
  ▼
Diagnóstico: causa da degradação + ação corretiva
```

---

## Classificador ML (MLflow)

O XGBoost classifica cada janela de 1h em três níveis de degradação:

| Label | Condição |
|---|---|
| `normal` | downtime ≤ 5% e perf_loss ≤ 10% |
| `degraded` | downtime > 5% ou perf_loss > 10% |
| `critical` | downtime > 15% ou perf_loss > 20% |

**Features:** `pct_idle`, `pct_production`, `pct_downtime`, `pct_perf_loss`, `pct_sched_downtime`, `count_sum`, `num_changes`

Experimento rastreado em MLflow como `piade-degradation-classification`.

---

## Setup

### Pré-requisitos

- Docker Engine 24+ e Docker Compose v2
- `make`
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
# Bronze: envia sequences_1h_data.csv para o MinIO
make upload-bronze

# Silver: normaliza e persiste em piade_telemetry (PostgreSQL)
make parse-silver

# Gold: gera embeddings e indexa no Milvus
make embed-gold

# ML: pré-processa e treina o classificador XGBoost
make preprocess
make train

# RAG: testa uma query diretamente
make rag-query QUERY="máquina s_1 com alto downtime"
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
├── data/
│   └── bronze/piade/sequences_1h_data.csv   # dataset PIADE
├── ingestion/
│   ├── upload_bronze.py      # Bronze: CSV → MinIO
│   ├── parse_silver.py       # Silver: MinIO → PostgreSQL (piade_telemetry)
│   ├── create_tables.sql     # DDL da tabela piade_telemetry
│   ├── embed_gold.py         # Gold: PostgreSQL → Milvus (embeddings)
│   └── test_retrieval.py     # validação da busca vetorial
├── rag/
│   ├── retriever.py          # busca vetorial no Milvus
│   ├── generator.py          # geração de resposta (Ollama / Gemini)
│   └── pipeline.py           # orquestração retrieve → generate
├── mlflow/
│   ├── preprocess.py         # feature engineering + labels de degradação
│   └── train.py              # treinamento XGBoost + tracking MLflow
├── api/
│   └── main.py               # FastAPI: POST /query
├── interface/
│   └── app.py                # Gradio UI
└── infra/
    └── docker/docker-compose.yml
```

---

## Status do Projeto

**Sprint 6 — Pipeline RAG operacional (PIADE sequences_1h_data)**
