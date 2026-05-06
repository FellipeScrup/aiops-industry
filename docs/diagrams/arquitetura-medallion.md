# Arquitetura — AIOps Industry RAG Industrial

## Visão Geral

A plataforma segue o padrão **Lakehouse + Medallion** (Armbrust et al., 2021) com três camadas de qualidade progressiva de dados:

## Camadas Medallion

### Bronze — MinIO (Data Lake)
- Dados brutos sem transformação
- Buckets: `bronze/`, `silver/`, `gold/`, `mlflow/`
- Datasets: ALPI (20 máquinas, 154 alarmes) e PIADE (5 máquinas, 133 alertas)
- Manuais de fabricantes: FANUC, Tetra Pak, Mazak

### Silver — PostgreSQL (Banco Relacional)
- Dados validados, limpos, normalizados
- Tabelas: `logs`, `alarm_types`, `machines`, `alarm_dictionary`
- Dicionário de alarmes: código → descrição → causa → ação corretiva

### Gold — Milvus (Banco Vetorial)
- Embeddings 768 dimensões via `nomic-embed-text`
- Collection: `alarm_chunks`
- Busca semântica por similaridade de cosseno

## Fluxo RAG
1. Usuário insere log/código de alarme na interface Gradio
2. FastAPI recebe e envia para o pipeline RAG
3. Retriever busca top-k chunks similares no Milvus
4. Prompt template monta contexto + pergunta
5. Llama 3.2:3b (Ollama, 58 tok/s GPU) gera resposta
6. Resposta retorna: falha + causa + ação corretiva

## Infraestrutura
- Tudo containerizado: `docker compose up`
- Tracking de experimentos: MLflow (backend PostgreSQL, artefatos MinIO)
- Hardware: NVIDIA GTX 1650 Mobile, Fedora 43
