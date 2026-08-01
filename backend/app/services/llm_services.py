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
        from app.core.exceptions import LLMError
        from app.core.logger import logger
        try:
            logger.info(f"Sending request to LLM Provider: {self.get_provider_name()}")
            response = self.llm.invoke(messages)
            logger.info(f"Received response from LLM Provider: {self.get_provider_name()}")
            return response
        except Exception as e:
            logger.error(f"LLM API request failed: {str(e)}", exc_info=True)
            raise LLMError(f"Failed to communicate with LLM service: {str(e)}")
    def get_llm(self):
        """
        Returns the underlying LangChain chat model.
        Used when structured outputs or advanced LangChain features are needed.
        """
        return self.llm
    def get_provider_name(self)->str:
        return self.provider.get_name()
