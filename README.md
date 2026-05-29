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

**Smart Factory Logs — Zenodo 10.5281/zenodo.14441997** (Seiger, 2024)

Sensores reais de uma fábrica de pequena escala (Indústria 4.0), coletados a 10 Hz de 7 estações, com os processos (armazenagem e produção) orquestrados por BPMN 2.0 / Camunda. Originalmente proposto para detecção de atividades de processo (García-Bañuelos et al., 2025, *Procedia Computer Science* 257, 856–863) e aqui reaproveitado para diagnóstico de falhas via RAG.

Estações: VGR (robô de transporte), HBW (armazém vertical), OV (forno), MM (fresa), SM (separadora por cor), WT (esteira), EC (ambiente + câmera).

Campos de cada evento NDJSON:

| Campo | Descrição |
|---|---|
| `id` | UUID do evento |
| `station` | Estação geradora: MM_1, EC_1, SM_1, HBW_1, OV_1, VGR_1, WT_1 |
| `timestamp` | Timestamp UTC do evento (precisão 100ms) |
| `current_state` | `"ready"` (disponível/ociosa) \| `"not ready"` (executando/ocupada) |
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
- **PostgreSQL** — Silver layer: tabela `smartfactory_logs` (eventos) + Gold `smartfactory_episodes` (episódios)
- **MinIO** — Data lake S3-compatible: buckets `bronze`, `silver`, `gold`
- **Milvus** — Vector store: banco vetorial com embeddings dos episódios (768d, COSINE)
- **fastembed** — embeddings locais via `nomic-ai/nomic-embed-text-v1.5`
- **Ollama** — inferência local de LLMs (Llama 3.2, Qwen 2.5)
- **Gemini API** — backend LLM alternativo via `GEMINI_API_KEY`
- **FastAPI** — API REST
- **Gradio** — interface de demonstração

---

## Arquitetura Medallion

```
MinIO bronze/             PostgreSQL Silver          Gold (Postgres + Milvus)
─────────────────         ──────────────────         ────────────────────────
smartfactory/             smartfactory_logs          smartfactory_episodes
  logs/training.txt ─►      id (PK)          ─agrega─►  episode_id (PK)
  logs/test.txt             station                     station
  bpmn/camunda-             event_timestamp             start_ts / end_ts
        activity.json       current_state               duration_s
                  ─►        current_task                is_anomaly
MinIO silver/               current_task_duration       sensors_changed
  smartfactory/             current_sub_task            text (narrativa PT)
    smartfactory_logs       sensors (JSONB)                  │
      .parquet (408k)       split (train/test)               ▼ embed (nomic 768d)
                                                       Milvus smartfactory_episodes
                                                         (episódios not ready)
```

A camada Gold agrega os eventos brutos de 10 Hz em **episódios** com início,
fim, duração e detecção de anomalia por sub-tarefa (`ingestion/parse_episodes.py`),
e só então os embeda no Milvus (`ingestion/embed_gold.py`). Ver
[Camada de Episódios](#camada-de-episódios-silver--gold).

### Fluxo de dados completo

```
upload_bronze.py   parse_silver.py     parse_episodes.py    embed_gold.py      export_silver.py
      │                  │                    │                   │                    │
      ▼                  ▼                    ▼                   ▼                    ▼
NDJSON → MinIO    MinIO → Postgres    Postgres → Postgres   Postgres → Milvus   Postgres → MinIO
(bronze/)         (smartfactory_logs) (smartfactory_         (episódios →        (silver/)
                  408k eventos         episodes, 485 epi.)   nomic 768d)
```

### Pipeline RAG

```
Técnico
  │
  ▼  pergunta em linguagem natural
FastAPI (POST /query)
  │
  ▼  embed query (nomic-embed-text-v1.5, prefixo search_query:)
Milvus  ──► top-k episódios similares (COSINE) + deduplicação por janela 10s
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

## Camada de Episódios (Silver → Gold)

Os eventos são amostrados a 10 Hz: uma ação de poucos segundos vira dezenas de
linhas quase idênticas. `ingestion/parse_episodes.py` agrega essas linhas em
**episódios** — a unidade que carrega significado para diagnóstico.

Regra de quebra: um novo episódio começa quando muda `station`, `current_task`,
`current_sub_task` ou `current_state` (linhas lidas em ordem de station +
timestamp). Para cada episódio calcula-se início, fim, **duração** (delta de
timestamps), os sensores que mudaram, e uma narrativa em português embedada no
Milvus.

**Detecção de anomalia de duração** (por sub-tarefa, sobre episódios `not ready`):

```
duração > mediana + 3·MAD   E   (duração − mediana) ≥ 2s
```

O piso absoluto de 2s evita falsos alarmes em sub-tarefas muito consistentes
(onde MAD ≈ 0). Episódios anômalos recebem `is_anomaly = True`.

No log de treino: **139.901 eventos → 485 episódios** (415 `not ready`, 7
anomalias). Só os episódios `not ready` são indexados por padrão
(`READY_SAMPLE=0`), porque episódios ociosos sequestram perguntas como
"por que a estação parou?".

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
make create-tables    # DDL: cria tabela smartfactory_logs no PostgreSQL
make ingest-bronze    # Bronze: NDJSON → s3://bronze/smartfactory/
make ingest-silver    # Silver: MinIO bronze → PostgreSQL (408k eventos)
make ingest-episodes  # Gold: agrega eventos → tabela smartfactory_episodes (485 epi.)
make export-silver    # Exporta Silver: PostgreSQL → s3://silver/smartfactory/
make embed            # Gold: episódios → Milvus (embeddings 768d)

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
```

A camada **Gold** (episódios) vive no PostgreSQL (`smartfactory_episodes`) e no
Milvus (coleção `smartfactory_episodes`, embeddings 768d) — não há artefato
exportado no MinIO `gold/`.

### URLs dos serviços

| Serviço | URL | Descrição |
|---|---|---|
| MinIO Console | http://localhost:9001 | Interface web do data lake (Bronze/Silver/Gold) |
| MinIO API | http://localhost:9000 | Endpoint S3-compatible |
| Adminer | http://localhost:8080 | UI web para o PostgreSQL (Silver/Gold) |
| Milvus gRPC | localhost:19530 | Banco vetorial (episódios) |
| Milvus métricas | http://localhost:9091 | Health/metrics do Milvus |
| Ollama | http://localhost:11434 | Inferência local de LLMs (geração) |
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
│   ├── parse_episodes.py     # Gold: agrega eventos → smartfactory_episodes + anomalia
│   ├── bpm_context.py        # enriquecimento BPM (atividade/processo/próxima estação)
│   ├── embed_gold.py         # Gold: episódios → Milvus (nomic embeddings)
│   ├── export_silver.py      # Exporta Silver: PostgreSQL → s3://silver/
│   └── test_retrieval.py     # validação da busca vetorial
├── rag/
│   ├── retriever.py          # busca vetorial no Milvus + deduplicação 10Hz
│   ├── generator.py          # geração de resposta (Ollama / Gemini) + guardrails
│   ├── pipeline.py           # orquestração retrieve → generate
│   └── query_parser.py       # extração de estação/timestamp da query
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
