from langchain_google_genai import ChatGoogleGenerativeAI
from app.config.settings import settings
from app.llms.base import BaseLLMProvider

class GeminiProvider(BaseLLMProvider):

    def get_llm(self) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            api_key=settings.GOOGLE_API_KEY,
            temperature=settings.LLM_TEMPERATURE,
        )

    def get_name(self) -> str:
        return "Gemini"