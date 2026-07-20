from app.llms.base import BaseLLMProvider
from app.schemas.chat import ChatRequest

class OpenAIProvider(BaseLLMProvider):

    def chat(self, prompt: ChatRequest) -> str:
        last_message = prompt.messages[-1].content
        return f"[OpenAI] {last_message}"

    def get_name(self) -> str:
        return "OpenAI"