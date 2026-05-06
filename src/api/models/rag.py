from pydantic import BaseModel


class RAGRequest(BaseModel):
    query: str
    limit: int = 3


class RAGSource(BaseModel):
    file_name: str
    page_number: int | str
    snippet: str


class RAGOutput(BaseModel):
    pest_name: str
    scientific_name: str | None = None
    summary: str
    key_damages: list[str]
    management_recommendations: list[str]
    sources: list[RAGSource]


class RAGResponse(BaseModel):
    query: str
    summary: str
    context: list[str]
    structured_answer: RAGOutput | None = None
    metadata: list[dict]
