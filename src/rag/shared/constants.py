import os
from pathlib import Path

PATH_RAG_ROOT = Path(__file__).parent.parent.resolve()


# Definições de constantes compartilhadas entre Ingestion e RAG

# Nomes de Índices do Qdrant / Vetor DB
COLLECTION_NAME = "agronomia-soja"

# Configurações de Embedding (Mantendo consistência com seu ingestion.py)
# DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_MODEL = "intfloat/multilingual-e5-large"

SPARSE_MODEL = "Qdrant/bm25"
COLBERT_MODEL = "colbert-ir/colbertv2.0"

# Configurações de Chunking
DENSE_DIMENSION = 1024  # Importante: o E5-large tem 1024
MAX_TOKENS = 500  # O E5 aceita 512 max, 510 é mais seguro


# Caminhos
DATA_PATH = f"{PATH_RAG_ROOT}/data"
