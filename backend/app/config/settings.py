
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Applications Settings
    Automatically loads values from .env file
    """
    #Application
    APP_NAME:str="ForgeMind AI"
    APP_VERSION:str="0.1.0"
    DEBUG:bool=True

    #Server
    HOST:str = "127.0.0.1"
    PORT:int=8000

    #Default AI Providers
    DEFAULT_PROVIDER: str = "groq"
    DEFAULT_MODEL: str = "openai/gpt-oss-120b"
    DEFAULT_EMBEDDING:str="bge"


    #API Key
    GROQ_API_KEY:str=""
    GOOGLE_API_KEY:str=""
    OPENAI_API_KEY:str=""
    TAVILY_API_KEY:str=""

    # Langfuse Observability
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://us.cloud.langfuse.com"
    LANGFUSE_ENABLED: bool = True

    #OLLAMA

    OLLAMA_BASE_URL :str = "http://localhost:11434"

    #Chromadb
    CHROMADB_PATH:str = "./chroma_db"

    # Additional Constants
    DATA_DIR: str = "data"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DEVICE: str = "cpu"
    GEMINI_MODEL: str = "gemini-2.5-flash"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 512
    SEARCH_MAX_RESULTS: int = 5
    RELEVANCE_SCORE_THRESHOLD: float = 0.60
    MAX_UPLOAD_SIZE_MB: int = 15
    CORS_ORIGINS: str = "*"

    def get_cors_origins(self) -> list[str]:
        """Returns list of allowed CORS origins from comma-separated string."""
        if not self.CORS_ORIGINS or self.CORS_ORIGINS.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    def get_data_dir(self) -> Path:
        p = Path(self.DATA_DIR)
        if not p.is_absolute():
            base_dir = Path(__file__).resolve().parent.parent.parent
            p = base_dir / p
        p.mkdir(parents=True, exist_ok=True)
        return p.resolve()

    model_config = SettingsConfigDict(
        env_file=(
            str(Path(__file__).resolve().parent.parent.parent / ".env"),
            ".env",
        ),
        env_file_encoding="utf-8",
        extra="ignore"
    )

@lru_cache()
def get_settings()->Settings:
    """
    Returns a cached object
    """
    return Settings()

settings = get_settings()