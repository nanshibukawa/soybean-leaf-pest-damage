import os
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from rag.shared.constants import (
    COLLECTION_NAME,
    DENSE_MODEL,
    DENSE_DIMENSION,
    SPARSE_MODEL,
    COLBERT_MODEL,
)


class VectorStoreProvider:
    def __init__(self):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )

    def create_collection(self, force_recreate: bool = False):
        """
        Cria a coleção se não existir.
        Se force_recreate=True, deleta a existente antes de criar.
        """
        if self.client.collection_exists(COLLECTION_NAME):
            if force_recreate:
                print(
                    f"⚠️ Forçando a recriação: Deletando coleção {COLLECTION_NAME}..."
                )
                self.client.delete_collection(COLLECTION_NAME)
            else:
                print(f"ℹ️ Coleção {COLLECTION_NAME} já existe. Ignorando criação.")
                return

        print(f"🏗️ Criando coleção {COLLECTION_NAME}...")
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": models.VectorParams(
                    size=DENSE_DIMENSION,
                    distance=models.Distance.COSINE,
                ),
                "colbert": models.VectorParams(
                    size=128,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                ),
            },
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        print("Coleção criada com sucesso!")


class EmbeddingProvider:
    def __init__(self):
        self.dense_model = TextEmbedding(DENSE_MODEL)
        self.sparse_model = SparseTextEmbedding(SPARSE_MODEL)
        self.colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL)

    def generate_passage_embeddings(self, chunk: str):
        """Gera o dicionário de vetores (Dense, Sparse, Colbert) para um parágrafo."""
        # Dense (E5) - Retorna o primeiro vetor da lista
        dense_embedding = list(self.dense_model.passage_embed([chunk]))[0].tolist()

        # Sparse (BM25)
        sparse_embedding = list(self.sparse_model.passage_embed([chunk]))[0].as_object()

        # Colbert (Late Interaction)
        colbert_embedding = list(self.colbert_model.passage_embed([chunk]))[0].tolist()

        return {
            "dense": dense_embedding,
            "sparse": sparse_embedding,
            "colbert": colbert_embedding,
        }
