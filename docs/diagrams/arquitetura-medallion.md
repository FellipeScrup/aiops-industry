# Arquitetura — AIOps Industry RAG Industrial

## Visão Geral

A plataforma segue o padrão **Lakehouse + Medallion** (Armbrust et al., 2021) com
três camadas de qualidade progressiva de dados, alimentadas pelo dataset
**Smart Factory Logs (14441997)** — eventos NDJSON a 10 Hz de 7 estações.

## Camadas Medallion

### Bronze — MinIO (Data Lake)
- Dados brutos sem transformação
- Arquivos: `smartfactory/logs/{training,test}.txt` (NDJSON 10 Hz) e
  `smartfactory/bpmn/camunda-activity.json`
- Upload por `ingestion/upload_bronze.py`

### Silver — PostgreSQL (Banco Relacional)
- Eventos normalizados e persistidos por `ingestion/parse_silver.py`
- Tabela: `smartfactory_logs` (~408k linhas)
  - `id` (PK), `station`, `event_timestamp`, `current_state`
  - `current_task`, `current_task_duration`, `current_sub_task`
  - `sensors` (JSONB), `split` (train/test)

### Gold — Episódios (PostgreSQL + Milvus)
- `ingestion/parse_episodes.py` agrega os eventos 10 Hz em **episódios** —
  sequências contínuas de mesma `station` + `task` + `sub_task` + `state` — na
  tabela `smartfactory_episodes` (485 episódios no treino).
  - Cada episódio: `start_ts`, `end_ts`, `duration_s`, `sensors_changed`,
    `is_anomaly`, `text` (narrativa em PT).
  - Anomalia de duração por sub-tarefa: `duração > mediana + 3·MAD` e excesso ≥ 2s.
- `ingestion/embed_gold.py` embeda a narrativa dos episódios `not ready` na
  collection Milvus `smartfactory_episodes` (768d, `nomic-embed-text-v1.5`,
  IVF_FLAT, COSINE, nlist=128).

## Fluxo RAG

1. Técnico insere a descrição do comportamento da estação na interface Gradio ou via `POST /query`
2. FastAPI repassa para o pipeline RAG
3. Retriever embeda a query com prefixo `search_query:` e busca os top-k episódios mais similares no Milvus (+ deduplicação 10 Hz)
4. Generator monta o prompt com o contexto dos episódios (duração, sensores, anomalia, contexto BPM) e envia ao LLM (Ollama local ou Gemini API)
5. O LLM atua como engenheiro de manutenção: diagnostica o comportamento, aponta causa provável e recomenda ação corretiva

## Infraestrutura

- Tudo containerizado via `docker compose up` (MinIO, PostgreSQL, Adminer, etcd, Milvus, Ollama)
- LLMs locais via Ollama (Llama 3.2:3b, Qwen 2.5)
- LLM remoto opcional via Gemini API (`model=gemini`)
