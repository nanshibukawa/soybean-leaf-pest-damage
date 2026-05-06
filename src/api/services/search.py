from qdrant_client import QdrantClient, models
from api.models.search import SearchResponse, SearchResult
from api.services.embeddings import EmbeddingService


class SearchService:
    def __init__(self, qdrant_url: str, qdrant_api_key: str, collection_name: str):
        self.qdrant_client = QdrantClient(
            api_key=qdrant_api_key,
            url=qdrant_url,
        )
        self.collection_name = collection_name
        self.embedding_service = EmbeddingService()

    def search(self, query: str, limit: int = 3) -> SearchResponse:
        query_dense, query_sparse, query_colbert = self.embedding_service.embed_query(
            query
        )

        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    prefetch=[
                        models.Prefetch(query=query_dense, using="dense", limit=10),
                        models.Prefetch(query=query_sparse, using="sparse", limit=10),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=15,
                )
            ],
            query=query_colbert,
            using="colbert",
            limit=limit,
        )

        max_score = max((result.score for result in results.points), default=1.0)

        search_results = [
            SearchResult(
                score=result.score / max_score,
                text=result.payload["text"],
                metadata=result.payload["metadata"],
            )
            for result in results.points
        ]

        return SearchResponse(results=search_results)


if __name__ == "__main__":
    pass
