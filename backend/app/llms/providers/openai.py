from app.llms.base import BaseLLMProvider
from langchain_core.language_models.chat_models import BaseChatModel

class OpenAIProvider(BaseLLMProvider):

    def get_llm(self) -> BaseChatModel:
        raise NotImplementedError("OpenAI provider is not yet fully implemented.")

    def get_name(self) -> str:
        return "OpenAI"