from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from app.config.settings import settings
from app.rag.embeddings.embedding_service import EmbeddingService


class ChromaVectorStore:
    """
    Handles storage and retrieval of document embeddings.
    """
    _db = None
    def __init__(self):
        if ChromaVectorStore._db is None:
            embedding_function = EmbeddingService().get_embeddings()

            ChromaVectorStore._db = Chroma(
                collection_name="forge_manuals",
                persist_directory=settings.CHROMADB_PATH,
                embedding_function=embedding_function,
            )

        self.db = ChromaVectorStore._db


    def add_documents(self, documents):
        self.db.add_documents(documents)

    def similarity_search(self, query: str, k: int = 5):
        return self.db.similarity_search(query=query, k=k)



class EmbeddingService:
    _embedding = None

    def get_embeddings(self):
        if EmbeddingService._embedding is None:
            EmbeddingService._embedding = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5"
            )
        return EmbeddingService._embedding