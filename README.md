# AIOps Industry — RAG para diagnóstico em logs industriais

> **TCC — Engenharia de Computação — Facens (2026)**

**Autores:** Fellipe Scruph e João Victor de Oliveira  
**Orientador:** Prof. Me. Adson Nogueira Alves

Plataforma de **Retrieval-Augmented Generation (RAG)** para análise de logs de uma fábrica inteligente. O sistema transforma eventos NDJSON (10 Hz) em **episódios**, indexa-os em banco vetorial e responde perguntas em linguagem natural, comparando três condições:

| Modo | O que faz |
|---|---|
| `filtrada` (proposta) | Busca vetorial + filtro por estação e janela temporal (±5 s) |
| `vetorial` | Busca vetorial pura (baseline de recuperação) |
| `sem_rag` | LLM sem episódios (controle) |

---

## Dataset

**Smart Factory Logs** — [Zenodo 10.5281/zenodo.14441997](https://doi.org/10.5281/zenodo.14441997) (Seiger, 2024)

Eventos a 10 Hz de 7 estações (VGR, HBW, OV, MM, SM, WT, EC), com processos BPMN/Camunda. Campos principais: `station`, `timestamp`, `current_state` (`ready` / `not ready`), `current_task`, `current_sub_task`, sensores (JSONB).

~408k eventos → **485 episódios** no log de treino (415 `not ready`, 7 anomalias de duração). Só episódios `not ready` entram no índice por padrão.

---

## Stack

- **Python 3.12** · **Docker Compose** · **Makefile**
- **PostgreSQL** — Silver (`smartfactory_logs`) e Gold (`smartfactory_episodes`)
- **MinIO** — data lake (`bronze` / `silver` / `gold`)
- **Milvus** — índice vetorial (COSINE, 768d)
- **fastembed** — `nomic-ai/nomic-embed-text-v1.5`
- **Ollama (local)** — geração: `qwen2.5:7b`, `qwen3.5:9b`, `gemma4:e4b` (+ `llama3.2:3b` opcional)
- **FastAPI** (`:8001`) · **Gradio** (`:7860`)

---

## Arquitetura Medallion

```
NDJSON bruto → MinIO bronze → Postgres Silver → episódios Gold → Milvus
     upload_bronze   parse_silver    parse_episodes     embed_gold
```

**Anomalia de duração** (por sub-tarefa, episódios `not ready`):

```
duração > mediana + 3·MAD   E   (duração − mediana) ≥ 2 s
```

### Pipeline RAG

```
Pergunta → parser (estação / horário)
        → embedding (nomic)
        → Milvus (± filtro escalar)
        → deduplicação (10 s) → top-5
        → prompt com guardrails → Ollama → Diagnóstico / Causa / Ação
```

Diagramas gerados: `evaluation/results/figuras_tcc/fig5_pipeline_rag.png` e `fig6_arquitetura_medallion.png`.

---

## Demo rápida (API + UI)

Pré-requisitos: Docker, Python 3.12, `make`, Ollama no **host** (recomendado no Windows/AMD), dataset em `data/bronze/14441997/`.

```bash
cp .env.example .env

# 1) Infra (MinIO, Postgres, Milvus, Adminer…)
make up

# 2) Ollama no host (não use o profile docker ollama no Windows/AMD)
ollama serve
ollama pull qwen2.5:7b
# opcional para a demo: ollama pull gemma4:e4b

# 3) Pipeline de dados (só na 1ª vez ou após make clean)
make medallion

# 4) API + interface (dois terminais)
python -m venv .venv-eval
.venv-eval/Scripts/pip install \
  -r ingestion/requirements.txt \
  -r rag/requirements.txt \
  -r api/requirements.txt \
  -r interface/requirements.txt \
  -r evaluation/requirements.txt
source .venv-eval/Scripts/activate   # Windows Git Bash
make api                             # http://localhost:8001
make ui                              # http://localhost:7860
```

Na UI: escolha modelo, modo (`RAG com filtragem` / `vetorial` / `sem RAG`) e use os exemplos do golden set.

### URLs

| Serviço | URL |
|---|---|
| Gradio UI | http://localhost:7860 |
| FastAPI | http://localhost:8001 |
| MinIO Console | http://localhost:9001 (`minioadmin` / `minioadmin123`) |
| Adminer | http://localhost:8080 |
| Ollama | http://localhost:11434 |
| Milvus | `localhost:19530` |

```bash
make status
make logs
make down     # para containers (mantém volumes)
make clean    # apaga volumes — exige reindexar
```

---

## Consulta via CLI / API

```bash
make rag-query QUERY="A calibração do motor 4 da HBW_1 às 10:29 foi atípica?"
```

```bash
curl -s http://localhost:8001/query -H 'Content-Type: application/json' \
  -d '{"question":"Por que a VGR_1 ficou not ready?","top_k":5,"model":"qwen2.5:7b","mode":"filtrada"}'
```

---

## Avaliação

Golden set objetivo de **24 perguntas** (8 factual, 8 multiestação, 8 causal). Três sistemas × geradores locais; juiz RAGAS de **outra família** (`phi4`).

| Tipo | Métricas |
|---|---|
| Recuperação | Hit@5, MRR, Recall@5 |
| Resposta | key facts, recusas, veredito de anomalia |
| RAGAS | Faithfulness, Factual Correctness, Answer Relevancy, Context Recall |

Figuras do TCC em `evaluation/results/figuras_tcc/`:

- `fig1_ragas_por_gerador.png` — qualidade da resposta por modelo/sistema  
- `fig2_recuperacao.png` — Hit/MRR/Recall (vetorial × filtrada × híbrida)  
- `fig3_recusa.png` — respostas vs recusas indevidas  
- `fig4_validacao_juiz.png` — sensibilidade do juiz  

```bash
make eval-generate MODEL=qwen2.5:7b
make eval-ragas    MODEL=qwen2.5:7b JUDGE=phi4
make eval-report   MODEL=qwen2.5:7b JUDGE=phi4
make eval-compare
```

Scripts auxiliares: `evaluation/judge_validation.py`, `evaluation/make_figures.py`.

---

## Estrutura

```
aiops-industry/
├── api/main.py                 # FastAPI: /query, /health (modes filtrada|vetorial|sem_rag)
├── interface/app.py            # Gradio: modelos, modos, episódios, parser
├── rag/
│   ├── pipeline.py             # orquestra retrieve → generate
│   ├── retriever.py            # Milvus + filtro escalar + dedup
│   ├── query_parser.py         # estação / timestamp / HH:MM
│   ├── generator.py            # Ollama + guardrails (estrito/brando)
│   └── lexical.py              # BM25 / RRF (ensaio híbrido)
├── ingestion/                  # Medallion: bronze → silver → episodes → embed
├── evaluation/                 # golden set, RAGAS, figuras
├── infra/docker/docker-compose.yml
└── Makefile
```

---

## Observações

- No **Windows com GPU AMD**, use Ollama nativo (Vulkan). O serviço `ollama` no Compose fica no profile `ollama-docker` e **não** sobe com `make up`.
- `OLLAMA_MODELS` deve apontar para `%USERPROFILE%\.ollama\models` (não para a pasta do usuário).
- Material de monografia/apresentação **não** é versionado neste repositório.
