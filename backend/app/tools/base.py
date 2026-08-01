from abc import ABC,abstractmethod

class BaseTool(ABC):
    """
    Base class for every ForgeMind tool
    """

    @abstractmethod
    def invoke(self,query:str):
        pass
    def name(self):
        pass