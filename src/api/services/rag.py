import logging
import traceback
from groq import Groq
from api.config.settings import settings
from api.config.prompts import RAG_PROMPT
from api.models.rag import RAGResponse
from api.services.search import SearchService

logger = logging.getLogger(__name__)


class RAGService:
    STAGE_NAME = "RAG Service"

    def __init__(self, search_service: SearchService):
        self.search_service = search_service
        self.client = Groq(api_key=settings.groq_api_key)

    def generate_answer(self, query: str, limit: int = 3) -> RAGResponse:
        search_results = self.search_service.search(query=query, limit=limit)

        context = "\n\n".join(result.text for result in search_results.results)
        prompt = RAG_PROMPT.format(context=context, query=query)

        try:
            completion = self.client.chat.completions.create(
                model=settings.groq_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.0,
                stream=False,
            )
        except Exception as e:
            logger.error(
                f"❌ Erro no {self.STAGE_NAME}",
                extra={
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
            )

        metadata = [
            {
                **result.metadata,
                "score": result.score,
            }
            for result in search_results.results
        ]

        return RAGResponse(
            query=query,
            answer=completion.choices[0].message.content,
            metadata=metadata,
        )
