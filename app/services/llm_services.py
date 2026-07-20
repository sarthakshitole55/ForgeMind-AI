from langchain_core.messages import BaseMessage
from app.llms.factory import LLMFactory

class LLMService:
    """
    Service responsible for interacting with the configured LLM
    """
    def __init__(self):
        self.provider = LLMFactory.get_provider()
        self.llm = self.provider.get_llm()

    def chat(self,messages:list[BaseMessage])->str:
        
        return self.llm.invoke(messages)
    def get_llm(self):
        """
        Returns the underlying LangChain chat model.
        Used when structured outputs or advanced LangChain features are needed.
        """
        return self.llm
    def get_provider_name(self)->str:
        return self.provider.get_name()
