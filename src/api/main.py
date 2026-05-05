from fastapi import FastAPI
from api.models.search import SearchRequest, SearchResponse
from api.services.search import SearchService
from api.config.settings import settings

app = FastAPI(title="API sobre pragas de soja")

search_service = SearchService(
    qdrant_url=settings.qdrant_url,
    qdrant_api_key=settings.qdrant_api_key,
    collection_name=settings.collection_name,
)


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    return search_service.search(query=request.query, limit=request.limit)


@app.get("/")
def root():
    return {"status": "online"}
