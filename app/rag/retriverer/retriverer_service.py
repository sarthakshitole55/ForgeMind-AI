from langchain_core.documents import Document
from app.rag.vectorstore.chroma_store import ChromaVectorStore

class RetrivererService:
    """
    Handles retrival of relevant documents from the vector database
    """

    def __init__(self):
        self.vector_store = ChromaVectorStore()

    def retrive(self,query:str,k:int=5)->list[Document]:
        return self.vector_store.similarity_search(query=query,k=k)