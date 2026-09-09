from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from app.config.settings import settings
from app.core.logger import logger
from app.rag.embeddings.embedding_service import EmbeddingService


class ChromaVectorStore:
    """
    Handles storage and retrieval of document embeddings.
    """
    _db = None
    def __init__(self, db: Chroma | None = None):
        if db is not None:
            self.db = db
            return

        if ChromaVectorStore._db is None:
            embedding_function = EmbeddingService().get_embeddings()

            chroma_path = settings.CHROMADB_PATH
            p = Path(chroma_path)
            if not p.is_absolute():
                base_dir = Path(__file__).resolve().parent.parent.parent.parent
                chroma_path = str((base_dir / p).resolve())

            ChromaVectorStore._db = Chroma(
                collection_name="forge_manuals",
                persist_directory=chroma_path,
                embedding_function=embedding_function,
            )

        self.db = ChromaVectorStore._db


    def add_documents(self, documents):
        self.db.add_documents(documents)

    def similarity_search(self, query: str, k: int = 5):
        return self.db.similarity_search(query=query, k=k)

    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 5, score_threshold: float | None = None
    ) -> list[tuple[Document, float]]:
        """
        Performs similarity search and returns documents paired with relevance scores [0, 1].
        Optionally filters out chunks below score_threshold.
        """
        if hasattr(self.db, "similarity_search_with_relevance_scores"):
            return self.db.similarity_search_with_relevance_scores(
                query=query, k=k, score_threshold=score_threshold
            )
        docs = self.db.similarity_search(query=query, k=k)
        return [(doc, 1.0) for doc in docs]

    def delete_document(self, filename: str, file_path: str | None = None) -> int:
        """
        Deletes all vector embeddings associated with the specified document.

        Matches by explicit 'filename' metadata as well as exact known 'source' paths
        to maintain compatibility across different indexing callers and legacy records.
        Returns the total number of deleted chunks.
        """
        deleted_count = 0

        # 1. Delete by explicit 'filename' metadata
        res = self.db._collection.delete(where={"filename": filename})
        if res and isinstance(res, dict):
            deleted_count += res.get("deleted", 0)

        # 2. Delete by exact known candidate 'source' paths for backward compatibility
        candidate_sources = {
            filename,
            f"data/{filename}",
            f"{settings.DATA_DIR}/{filename}",
        }
        if file_path:
            p = Path(file_path)
            candidate_sources.add(str(p))
            candidate_sources.add(str(p.resolve()))
            candidate_sources.add(p.as_posix())
            try:
                candidate_sources.add(str(p.relative_to(Path.cwd())))
            except ValueError:
                pass

        res = self.db._collection.delete(
            where={"source": {"$in": list(candidate_sources)}}
        )
        if res and isinstance(res, dict):
            deleted_count += res.get("deleted", 0)

        logger.info(f"Deleted {deleted_count} vectors from ChromaDB for document '{filename}'")
        return deleted_count

    def get_document_chunks(self, filename: str, file_path: str | None = None) -> list[str]:
        """
        Returns list of chunk IDs associated with the specified document.
        """
        chunk_ids = set()

        try:
            res = self.db._collection.get(where={"filename": filename})
            if res and "ids" in res:
                chunk_ids.update(res["ids"])
        except Exception:
            pass

        candidate_sources = {
            filename,
            f"data/{filename}",
            f"{settings.DATA_DIR}/{filename}",
        }
        if file_path:
            p = Path(file_path)
            candidate_sources.add(str(p))
            candidate_sources.add(str(p.resolve()))
            candidate_sources.add(p.as_posix())
            try:
                candidate_sources.add(str(p.relative_to(Path.cwd())))
            except ValueError:
                pass

        try:
            res = self.db._collection.get(
                where={"source": {"$in": list(candidate_sources)}}
            )
            if res and "ids" in res:
                chunk_ids.update(res["ids"])
        except Exception:
            pass

        return list(chunk_ids)



class EmbeddingService:
    _embedding = None

    def get_embeddings(self):
        if EmbeddingService._embedding is None:
            EmbeddingService._embedding = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5"
            )
        return EmbeddingService._embedding