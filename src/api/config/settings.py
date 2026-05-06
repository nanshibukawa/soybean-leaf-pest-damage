from pydantic_settings import BaseSettings, SettingsConfigDict
from tenacity import stop_after_attempt, wait_exponential


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )
    qdrant_url: str
    qdrant_api_key: str
    collection_name: str = "agronomia-soja"
    dense_model: str = "intfloat/multilingual-e5-large"
    sparse_model: str = "Qdrant/bm25"
    colbert_model: str = "colbert-ir/colbertv2.0"
    groq_api_key: str
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.3-70b-versatile"


settings = Settings()
