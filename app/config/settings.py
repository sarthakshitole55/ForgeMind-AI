
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