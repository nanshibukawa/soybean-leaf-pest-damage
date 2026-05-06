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

## Execução

```bash
cd src
python -m rag.main
```

## Ingestão de documentos

```bash
cd src
python -m rag.ingestion.main
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
