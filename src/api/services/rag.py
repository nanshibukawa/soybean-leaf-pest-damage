import logging

from api.config.settings import settings
from api.config.prompts import RAG_PROMPT
from api.models.rag import RAGResponse, RAGOutput
from api.services.search import SearchService

from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field


class RAGService:
    STAGE_NAME = "RAG Service"

    def __init__(self, search_service: SearchService):
        self.search_service = search_service
        self.model = GroqModel(
            model_name=settings.groq_model,
            provider=GroqProvider(api_key=settings.groq_api_key),
        )
        self.agent = Agent(
            self.model,
            output_type=RAGOutput,
            system_prompt=RAG_PROMPT,
            retries=3,
        )

    async def generate_answer(self, query: str, limit: int = 3) -> RAGResponse:
        search_results = self.search_service.search(query=query, limit=limit)

        context_parts = []
        for result in search_results.results:
            source = result.metadata.get("source", "Fonte desconhecida")
            page = result.metadata.get("page", "N/A")
            context_parts.append(f"[Fonte: {source}, Pág: {page}]\n{result.text}")

        context = "\n\n---\n\n".join(context_parts)

        result = await self.agent.run(f"Contexto: {context}\n\nPergunta: {query}")

        structured_data = result.output

        metadata = [
            {
                **result.metadata,
                "score": result.score,
            }
            for result in search_results.results
        ]

        return RAGResponse(
            query=query,
            summary=structured_data.summary,
            context=context_parts,
            structured_answer=structured_data,
            metadata=metadata,
        )
