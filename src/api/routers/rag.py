from fastapi import APIRouter, Depends
from api.models.rag import RAGRequest, RAGResponse
from api.services.rag import RAGService
from api.dependencies import get_rag_service

router = APIRouter()


@router.post("/rag", response_model=RAGResponse)
def rag(request: RAGRequest, rag_service: RAGService = Depends(get_rag_service)):
    return rag_service.generate_answer(request.query, request.limit)
