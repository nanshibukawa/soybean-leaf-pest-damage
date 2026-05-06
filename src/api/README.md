# API - Soybean Leaf Pest Damage

API para busca semântica e respostas com RAG.

## Funcionalidades
- Busca de trechos relevantes
- Geração de resposta estruturada com LLM
- Retorno de contexto e metadados de fonte

## Estrutura
- `routers/`: rotas HTTP
- `services/search.py`: busca vetorial/semântica
- `services/rag.py`: orquestração RAG (contexto + geração)
- `models/`: schemas de entrada/saída

## Execução local
```bash
cd src
uvicorn api.main:app --reload
```

## Configuração
Defina as variáveis esperadas em `api/config/settings.py` e em um arquivo `.env` na raiz do projeto.

### Variáveis obrigatórias

```env
qdrant_url=https://<sua-instancia-qdrant>
qdrant_api_key=<sua-chave-qdrant>
groq_api_key=<sua-chave-groq>
```

### Variáveis opcionais

```env
collection_name=agronomia-soja
dense_model=intfloat/multilingual-e5-large
sparse_model=Qdrant/bm25
colbert_model=colbert-ir/colbertv2.0
groq_base_url=https://api.groq.com/openai/v1
groq_model=llama-3.3-70b-versatile
```

## Observações
O serviço RAG combina:
1. recuperação de contexto (`SearchService`)
2. montagem de prompt com fontes
3. geração de resposta estruturada