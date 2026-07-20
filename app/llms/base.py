from abc import ABC, abstractmethod

from langchain_core.language_models.chat_models import BaseChatModel

class BaseLLMProvider(ABC):
    """
    Abstract base class for every LLM provider
    """

    
    @abstractmethod
    def get_llm(self)->BaseChatModel:
        """
        Generates a response from the LLM
        """
        pass

    @abstractmethod
    def get_name(self)->str:
        """
        returns provider name
        """
        pass