# AIOps Industry — RAG para Diagnóstico de Falhas Industriais

> **TCC — Engenharia de Computação — Facens**

**Autores:** Fellipe Scrup e João Victor de Oliveira
**Orientador:** Prof. Me. Adson Nogueira Alves

---

## Sobre

Plataforma de Retrieval-Augmented Generation (RAG) voltada ao diagnóstico de falhas em equipamentos industriais. O sistema ingere manuais técnicos e históricos de manutenção, indexa o conhecimento em um banco vetorial e permite consultas em linguagem natural para auxiliar técnicos de campo no diagnóstico e resolução de falhas.

---

## Stack

- **Python** — linguagem principal
- **Docker Compose** — orquestração local dos serviços
- **PostgreSQL** — metadados e histórico de consultas
- **MinIO** — armazenamento de objetos (manuais, arquivos brutos)
- **Milvus** — banco de dados vetorial
- **MLflow** — rastreamento de experimentos e versionamento de modelos
- **Ollama** — inferência local de LLMs
- **FastAPI** — API REST
- **Gradio** — interface de demonstração

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
# Edite .env se quiser alterar senhas

# 3. Suba todos os serviços
make up
```

### URLs dos serviços

| Serviço | URL | Descrição |
|---------|-----|-----------|
| MinIO Console | http://localhost:9001 | Interface web do data lake (Bronze) |
| MinIO API | http://localhost:9000 | Endpoint S3-compatible |
| Adminer | http://localhost:8080 | UI web para o PostgreSQL (Silver) |
| MLflow | http://localhost:5000 | Tracking de experimentos |
| Milvus gRPC | localhost:19530 | Banco vetorial (Gold) |
| Milvus métricas | http://localhost:9091 | Health/metrics do Milvus |

**Credenciais padrão MinIO:** `minioadmin / minioadmin123`  
**Credenciais padrão PostgreSQL:** host `localhost:5432`, usuário `aiops`, banco `aiops_industry`  
**Adminer:** selecione "PostgreSQL", servidor `postgres`, usuário/senha do `.env`

### Verificando a saúde dos serviços

```bash
# Status resumido
make status

# Logs em tempo real
make logs

# Healthcheck individual (exemplo)
docker inspect --format='{{.State.Health.Status}}' aiops-postgres
```

### Parando e limpando

```bash
# Para containers (mantém volumes)
make down

# Para containers E remove todos os dados
make clean
```

---

## Status do Projeto

**Sprint 2 — Infraestrutura Base**
