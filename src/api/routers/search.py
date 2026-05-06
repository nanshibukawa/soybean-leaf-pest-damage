from fastapi import APIRouter
from api.config.settings import settings
from api.models.search import SearchRequest, SearchResponse
from api.services.search import SearchService

router = APIRouter()

search_service = SearchService(
    qdrant_url=settings.qdrant_url,
    qdrant_api_key=settings.qdrant_api_key,
    collection_name=settings.collection_name,
)


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    return search_service.search(query=request.query, limit=request.limit)
