from functools import lru_cache
from app.services.llm_services import LLMService
from app.rag.retriverer.retriverer_service import RetrivererService

@lru_cache
def get_llm():
    return LLMService()

@lru_cache
def get_retriverer():
    return RetrivererService()

