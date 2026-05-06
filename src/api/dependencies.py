from api.config.settings import settings
from api.services.search import SearchService
from api.services.rag import RAGService

# Inicialização centralizada dos serviços (Singleton-like pattern)
# Isso evita acoplamento direto entre as rotas.

search_service = SearchService(
    qdrant_url=settings.qdrant_url,
    qdrant_api_key=settings.qdrant_api_key,
    collection_name=settings.collection_name,
)

rag_service = RAGService(search_service=search_service)


def get_search_service():
    return search_service


def get_rag_service():
    return rag_service
