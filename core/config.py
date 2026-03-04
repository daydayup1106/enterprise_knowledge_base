"""
core/config.py
==============
Global configuration loader.
All settings are read from environment variables (or .env file).
No magic numbers are allowed in business logic — read from here.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """
    Pydantic-Settings model.
    Reads from '.env' in the project root automatically.
    """

    # ---- DeepSeek API ----
    deepseek_api_key: str = Field(..., alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field("https://api.deepseek.com/v1", alias="DEEPSEEK_BASE_URL")
    deepseek_model: str = Field("deepseek-chat", alias="DEEPSEEK_MODEL")

    # ---- Local Model Paths (absolute, pre-downloaded) ----
    embed_model_name: str = Field(..., alias="EMBED_MODEL_NAME")
    rerank_model_name: str = Field(..., alias="RERANK_MODEL_NAME")
    device: str = Field("cuda", alias="DEVICE")  # "cuda" | "cuda:0" | "cpu"

    # ---- RAG Pipeline ----
    rag_hybrid_top_k: int = Field(10, alias="RAG_HYBRID_TOP_K")
    rag_rerank_top_n: int = Field(3, alias="RAG_RERANK_TOP_N")
    chunk_size: int = Field(512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(128, alias="CHUNK_OVERLAP")

    # ---- LangGraph Memory ----
    memory_window: int = Field(10, alias="MEMORY_WINDOW")

    # ---- Redis ----
    redis_url: str = Field("redis://localhost:6379/0", alias="REDIS_URL")

    # ---- Knowledge Base ----
    knowledge_base_doc: str = Field(
        "data/medical_ai_papers_2024_2025.md",
        alias="KNOWLEDGE_BASE_DOC"
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "populate_by_name": True}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Singleton accessor — settings object is constructed once and reused.
    lru_cache ensures no repeated file-reads during the application lifetime.
    """
    return Settings()
