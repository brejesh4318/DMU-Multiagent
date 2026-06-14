"""
Central configuration — all values from environment variables.
"""
from __future__ import annotations
import os
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    APP_NAME: str = "DMU Analytics Platform"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # Data
    DATA_DIR: str = "./data"
    DB_PATH: str = "./data/dmu.duckdb"
    VECTOR_STORE_DIR: str = "./data/vector_store"

    # LLM — Groq only
    GROQ_API_KEY: str = Field(..., env="GROQ_API_KEY")
    LLM_MODEL: str = "llama-3.1-8b-instant"
    JUDGE_MODEL: str = "llama-3.1-70b-versatile"   
    TEMPERATURE: float = 0.0
    MAX_TOKENS: int = 1024

    # Embeddings
    EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"

    # Retrieval
    PARENT_CHUNK_SIZE: int = 512    
    CHILD_CHUNK_SIZE: int = 128     
    CHUNK_OVERLAP: int = 20
    TOP_K_RETRIEVAL: int = 10       # retrieve before reranking
    TOP_K_RERANK: int = 5           # keep after reranking

    # Tavily web search
    TAVILY_API_KEY: str = Field("", env="TAVILY_API_KEY")
    TAVILY_ENABLED: bool = True


    # LangSmith
    LANGCHAIN_API_KEY: str = Field("", env="LANGCHAIN_API_KEY")
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_PROJECT: str = "dmu-analytics-v2"

    # SQL
    SQL_MAX_RETRIES: int = 3
    JUDGE_SCORE_THRESHOLD: int = 3

    # Cost tracking (Groq pricing per 1M tokens, USD)
    COST_PER_1M_INPUT: float = 0.05
    COST_PER_1M_OUTPUT: float = 0.08

    model_config = {"env_file": ".env", "extra": "ignore"}

    def setup_langsmith(self):
        if self.LANGCHAIN_API_KEY:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.LANGCHAIN_API_KEY
            os.environ["LANGCHAIN_PROJECT"] = self.LANGCHAIN_PROJECT


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.setup_langsmith()
    return s
