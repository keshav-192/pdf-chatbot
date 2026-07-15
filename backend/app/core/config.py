import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from dotenv import load_dotenv

# Load .env file into os.environ for non-Pydantic config usage
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Load from .env in backend directory, or fallback
        env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    PORT: int = 8000
    HOST: str = "127.0.0.1"
    ENVIRONMENT: str = "development"

    # CORS
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Databases
    DATABASE_URL: str = "sqlite:///./local_db.db"
    CHROMA_PERSIST_DIR: str = "./chroma_db"

    # Third Party APIs
    OPENAI_API_KEY: str = ""
    FIREBASE_PROJECT_ID: str = ""

    # Chunker & Embedding Configuration
    DEFAULT_CHUNK_SIZE: int = 450
    DEFAULT_CHUNK_OVERLAP: int = 90
    EMBEDDING_PROVIDER: str = "local"  # 'openai' or 'local'
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_MODEL_TIER: str = "base"  # 'base' or 'large'
    MIN_RERANK_CONFIDENCE: float = -15.0  # may need empirical tuning once real testing starts

    # OLD: defined LLM_PROVIDER as a simple string without ollama support — replaced below to add ollama variables
    # LLM_PROVIDER: str = "openai"  # 'openai' or 'claude'
    LLM_PROVIDER: str = "openai"  # 'openai', 'claude', or 'ollama'
    OLLAMA_API_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "mistral:7b-instruct-q4_0"
    CLAUDE_API_KEY: str = ""
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 1000
    CONVERSATION_MEMORY_LIMIT: int = 10
    MONTHLY_SPEND_LIMIT_USD: float = 50.0
    
    # Centralized RAG Retrieval Settings
    VECTOR_TOP_K: int = 15
    BM25_TOP_K: int = 15
    FINAL_TOP_N: int = 5

    @property
    def cors_origins_list(self) -> List[str]:
        origins = [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]
        if "*" in origins:
            raise ValueError("CORS_ORIGINS cannot contain wildcard '*' per project security guidelines.")
        return origins

    @field_validator("CORS_ORIGINS")
    @classmethod
    def validate_cors(cls, v: str) -> str:
        origins = [origin.strip() for origin in v.split(",") if origin.strip()]
        if "*" in origins:
            raise ValueError("CORS_ORIGINS cannot contain wildcard '*' per project security guidelines.")
        return v

# Instantiate settings globally
settings = Settings()
