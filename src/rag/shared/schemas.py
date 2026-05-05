from pydantic import BaseModel, Field
from typing import List, Optional, Any


class PageData(BaseModel):
    """Conteúdo bruto por página."""

    text: str
    page_no: int


class ExtractedData(BaseModel):
    """Schema para o output padronizado dos extratores."""

    full_text: str
    pages: List[PageData] = Field(default_factory=list)
    doc_object: Optional[Any] = None
    metadata: dict = Field(default_factory=dict)


class Chunk(BaseModel):
    """Schema para um chunk individual de texto."""

    text: str
    page: int
    metadata: dict = Field(default_factory=dict)
