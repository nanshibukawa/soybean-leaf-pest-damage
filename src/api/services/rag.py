from groq import Groq
from api.config.settings import settings
from api.models.rag import RAGResponse
from api.services.search import SearchService


class RAGService:
    def __init__(self, search_service: SearchService):
        self.search_service = search_service
        self.client = Groq(api_key=settings.groq_api_key)

    def generate_answer(self, query: str, limit: int = 3) -> RAGResponse:
        search_results = self.search_service.search(query=query, limit=limit)

        context = "\n\n".join(result.text for result in search_results.results)

        prompt = """
            Você é um especialista em agronomia focado em pragas da soja e culturas associadas. 
            Sua tarefa é responder perguntas técnicas de forma precisa, utilizando exclusivamente o contexto fornecido. 
            Se a informação não estiver presente, responda que não possui dados suficientes nos documentos. 
            Sempre cite o nome do arquivo PDF da fonte e a página (ex: arquivo.pdf, Pág: X) ao afirmar algo.
            
            Context: {context}
            Question: {query}
            Amswer:"""

        completion = self.client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt.format(context=context, query=query),
                },
            ],
            temperature=0.0,
            stream=False,
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
