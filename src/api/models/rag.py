from pydantic import BaseModel, Field


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
    key_damages: list[str] = Field(description="Lista exaustiva e detalhada com todos os danos causados pela praga descritos no contexto.")
    management_recommendations: list[str] = Field(description="Lista exaustiva e detalhada com todas as recomendações de controle e manejo mencionadas no contexto.")
    sources: list[RAGSource]


class RAGResponse(BaseModel):
    query: str
    summary: str
    context: list[str]
    structured_answer: RAGOutput | None = None
    metadata: list[dict]
