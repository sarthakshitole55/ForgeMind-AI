
from functools import lru_cache
from pydantic_settings import BaseSettings,SettingsConfigDict

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
    DEFAULT_MODEL: str = "llama-3.3-70b-versatile"
    DEFAULT_EMBEDDING:str="bge"


    #API Key
    GROQ_API_KEY:str=""
    GOOGLE_API_KEY:str=""
    OPENAI_API_KEY:str=""
    TAVILY_API_KEY:str=""

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

    model_config = SettingsConfigDict(
        env_file=".env",
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