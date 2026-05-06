from fastapi import APIRouter, Depends
from api.models.search import SearchRequest, SearchResponse
from api.services.search import SearchService
from api.dependencies import get_search_service

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest, search_service: SearchService = Depends(get_search_service)
):
    return search_service.search(query=request.query, limit=request.limit)
