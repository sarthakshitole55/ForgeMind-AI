from langchain_chroma import Chroma

from app.config.settings import settings
from app.rag.embeddings.embedding_service import EmbeddingService


class ChromaVectorStore:
    """
    Handles storage and retrieval of document embeddings.
    """

    def __init__(self):
        embedding_function = EmbeddingService().get_embeddings()

        self.db = Chroma(
            collection_name="forge_manuals",
            persist_directory=settings.CHROMADB_PATH,
            embedding_function=embedding_function,
        )

    def add_documents(self, documents):
        self.db.add_documents(documents)

    def similarity_search(self, query: str, k: int = 5):
        return self.db.similarity_search(query=query, k=k)