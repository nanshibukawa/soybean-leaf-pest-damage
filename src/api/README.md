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
Defina as variáveis esperadas em `api/config/settings.py` (ex.: chave/modelo do provedor LLM).

## Observações
O serviço RAG combina:
1. recuperação de contexto (`SearchService`)
2. montagem de prompt com fontes
3. geração de resposta estruturada