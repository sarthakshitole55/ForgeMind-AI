from langchain_chroma import Chroma
from app.config.settings import settings
from app.rag.embeddings.embedding_service import EmbeddingService

class ChromaVectorStore:
    """
    Handles storage and retrieval of document embeddings using ChromaDB.
    """
    def __init__(self):
        embedding_function = EmbeddingService().get_embeddings()

        self.db = Chroma(
            collection_name="forge_manuals",
            persist_directory=settings.CHROMADB_PATH,
            embedding_function=embedding_function,
        )

    def add_documents(self, chunks):
        """Index a list of LangChain Document chunks into ChromaDB."""
        return self.db.add_documents(chunks)

    def similarity_search(self, query: str, k: int = 5):
        """Return the k most similar documents to the query."""
        return self.db.similarity_search(query, k=k)