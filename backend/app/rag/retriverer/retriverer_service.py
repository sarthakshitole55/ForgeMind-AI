from langchain_core.documents import Document
from app.rag.vectorstore.chroma_store import ChromaVectorStore
from app.core.logger import logger
import time


class RetrivererService:
    """
    Retrieves relevant documents from Chroma Vector Store.
    """

    def __init__(self):
        self.vector_store = ChromaVectorStore()

    def retrive(self, query: str, k: int = 5) -> list[Document]:
        start = time.time()

        documents = self.vector_store.similarity_search(
            query=query,
            k=k,
        )

        elapsed = time.time() - start

        logger.info(
            f"""
Retrieval Summary
-----------------
Question       : {query}
Retrieved Chunks : {len(documents)}
Retrieval Time : {elapsed:.3f}s
"""
        )

        return documents