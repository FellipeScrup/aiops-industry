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

> Em construção

---

## Status do Projeto

**Sprint 2 — Infraestrutura Base**
