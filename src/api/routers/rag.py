from fastapi import APIRouter, Depends
from api.models.rag import RAGRequest, RAGResponse
from api.services.rag import RAGService
from api.dependencies import get_rag_service

router = APIRouter()


@router.post("/rag", response_model=RAGResponse)
async def rag(request: RAGRequest, rag_service: RAGService = Depends(get_rag_service)):
    return await rag_service.generate_answer(request.query, request.limit)
