from langchain_groq import ChatGroq

from app.config.settings import settings
from app.llms.base import BaseLLMProvider


class GroqProvider(BaseLLMProvider):

    def get_llm(self) -> ChatGroq:
        return ChatGroq(
            model=settings.DEFAULT_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=0.2,
        )

    def get_name(self) -> str:
        return "Groq"