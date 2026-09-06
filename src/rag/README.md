# RAG Module

Módulo de Retrieval-Augmented Generation (RAG) para consultas em documentos técnicos sobre pragas e danos em folhas de soja.

## Visão geral

O módulo cobre duas etapas principais:

1. **Ingestão e preparação de documentos**
	- extração de texto
	- chunking semântico
	- geração de embeddings
	- indexação em base vetorial

2. **Recuperação e geração de respostas**
	- busca híbrida com sparse + dense
	- fusão de rankings via RRF
	- reranking final com ColBERT
	- resposta estruturada via LLM

## Estrutura

- `ingestion/`: pipeline de ingestão e preparação de documentos
- `ingestion/utils/chunker.py`: chunking semântico com Sentence Transformers + HDBSCAN
- `shared/`: constantes e schemas compartilhados
- `engine.py`: motor principal de recuperação e geração
- `predictor.py`: interface de predição/consulta
- `main.py`: ponto de entrada do módulo

## Execução e Testes

Para rodar o script de consulta de teste rápido no terminal:

```bash
UV_SKIP_WHEEL_FILENAME_CHECK=1 uv run --package rag python -m rag.ingestion.test_query
```

## Ingestão de documentos

Para rodar a ingestão de PDFs da pasta `src/rag/data` para a base vetorial:

```bash
# Ingestão padrão (sem recriar a coleção)
UV_SKIP_WHEEL_FILENAME_CHECK=1 uv run --package rag python -m rag.ingestion.main --extractor docling --chunker semantic

# Forçar a recriação da coleção (apaga os vetores antigos)
UV_SKIP_WHEEL_FILENAME_CHECK=1 uv run --package rag python -m rag.ingestion.main --extractor docling --chunker semantic --recreate
```

## Tecnologias utilizadas

- Sentence Transformers
- HDBSCAN
- Qdrant
- Sparse embeddings
- Dense embeddings
- ColBERT
- RRF (Reciprocal Rank Fusion)

## Observações

- A API HTTP de busca e geração está em `../api/`.
- Configure variáveis de ambiente e chaves de modelo antes da execução.
