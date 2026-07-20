from app.rag.rag_service import RAGService
from app.tools.base import BaseTool

class RAGTool(BaseTool):
    def __init__(self):
        self.rag = RAGService()

    def invoke(self,query:str):
        return self.rag.invoke(query)

    def name(self):
        return "rag"