from langchain_core.documents import Document
from app.rag.vectorstore.chroma_store import ChromaVectorStore
from app.config.settings import settings
from app.core.exceptions import VectorStoreError
from app.core.logger import logger
from app.core.observability import ObservabilityService
import time


class RetrivererService:
    """
    Retrieves relevant documents from Chroma Vector Store with relevance filtering.
    """

    def __init__(self, vector_store: ChromaVectorStore | None = None):
        self.vector_store = vector_store or ChromaVectorStore()

    def retrive(
        self,
        query: str,
        k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[Document]:
        start = time.time()
        limit = k if k is not None else settings.SEARCH_MAX_RESULTS
        threshold = (
            score_threshold
            if score_threshold is not None
            else settings.RELEVANCE_SCORE_THRESHOLD
        )

        try:
            docs_and_scores = self.vector_store.similarity_search_with_relevance_scores(
                query=query,
                k=limit,
                score_threshold=threshold,
            )
        except Exception as e:
            elapsed = time.time() - start
            logger.error(
                f"Vector store retrieval failed for query '{query}': {e}",
                exc_info=True,
            )
            ObservabilityService.record_retrieval_error(
                query=query,
                error=str(e),
                latency_seconds=elapsed,
            )
            raise VectorStoreError("Vector store is temporarily unavailable for document retrieval.") from e

        for doc, score in docs_and_scores:
            doc.metadata["relevance_score"] = round(score, 4)

        documents = [doc for doc, _score in docs_and_scores]

        elapsed = time.time() - start

        ObservabilityService.record_retrieval(
            query=query,
            retrieved_documents=documents,
            relevance_threshold=threshold,
            latency_seconds=elapsed,
            raw_candidates_count=limit,
        )

        logger.info(
            f"""
Retrieval Summary
-----------------
Question       : {query}
Retrieved Chunks : {len(documents)} (k={limit}, threshold={threshold})
Retrieval Time : {elapsed:.3f}s
"""
        )

        return documents