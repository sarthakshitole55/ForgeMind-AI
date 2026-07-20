from langchain_core.tools import tool
from app.tools.rag_tools import RAGTool
rag = RAGTool()
@tool
def search_documents(question:str)->str:
    """
    Search uploaded manuals and documents
    """
    result = rag.inoke(question)
    return result["answer"]