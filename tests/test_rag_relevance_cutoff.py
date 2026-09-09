import unittest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app.rag.retriverer.retriverer_service import RetrivererService
from app.rag.rag_service import RAGService
from app.rag.vectorstore.chroma_store import ChromaVectorStore
from app.config.settings import settings


class TestRAGRelevanceCutoff(unittest.TestCase):
    """
    Deterministic unit tests for RAG relevance score filtering and standard refusal cutoff.
    Does not require live LLM calls or network access.
    """

    def setUp(self):
        self.doc_high1 = Document(
            page_content="High relevance chunk 1 about algorithms.",
            metadata={"source": "data/algo.pdf", "filename": "algo.pdf", "page": 10},
        )
        self.doc_high2 = Document(
            page_content="High relevance chunk 2 about data structures.",
            metadata={"source": "data/algo.pdf", "filename": "algo.pdf", "page": 12},
        )
        self.doc_marginal = Document(
            page_content="Marginal relevance chunk about Python basics.",
            metadata={"source": "data/algo.pdf", "filename": "algo.pdf", "page": 3},
        )
        self.doc_irrelevant1 = Document(
            page_content="Irrelevant chunk about pediatric dosage.",
            metadata={"source": "data/medical.pdf", "filename": "medical.pdf", "page": 99},
        )
        self.doc_irrelevant2 = Document(
            page_content="Irrelevant chunk about Boeing 737 turbine.",
            metadata={"source": "data/aviation.pdf", "filename": "aviation.pdf", "page": 4},
        )

    def _create_mock_store(self, docs_and_scores):
        """Helper to create mock ChromaVectorStore implementing similarity_search_with_relevance_scores."""
        mock_store = MagicMock(spec=ChromaVectorStore)

        def mock_search(query, k=5, score_threshold=None):
            items = docs_and_scores[:k]
            if score_threshold is not None:
                items = [(doc, score) for doc, score in items if score >= score_threshold]
            return items

        mock_store.similarity_search_with_relevance_scores.side_effect = mock_search
        return mock_store

    def test_all_chunks_above_threshold_retained(self):
        """When all retrieved chunks exceed threshold, all must be retained."""
        docs_and_scores = [
            (self.doc_high1, 0.85),
            (self.doc_high2, 0.78),
            (self.doc_marginal, 0.65),
        ]
        mock_store = self._create_mock_store(docs_and_scores)
        retriever = RetrivererService(vector_store=mock_store)

        results = retriever.retrive("query", score_threshold=0.60)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].page_content, self.doc_high1.page_content)
        self.assertEqual(results[1].page_content, self.doc_high2.page_content)
        self.assertEqual(results[2].page_content, self.doc_marginal.page_content)

    def test_some_chunks_below_threshold_removed(self):
        """Chunks below threshold must be dropped while retaining chunks meeting threshold."""
        docs_and_scores = [
            (self.doc_high1, 0.82),
            (self.doc_marginal, 0.62),
            (self.doc_irrelevant1, 0.55),
            (self.doc_irrelevant2, 0.40),
        ]
        mock_store = self._create_mock_store(docs_and_scores)
        retriever = RetrivererService(vector_store=mock_store)

        results = retriever.retrive("query", score_threshold=0.60)
        self.assertEqual(len(results), 2)
        self.assertIn(self.doc_high1, results)
        self.assertIn(self.doc_marginal, results)
        self.assertNotIn(self.doc_irrelevant1, results)
        self.assertNotIn(self.doc_irrelevant2, results)

    def test_all_chunks_below_threshold_returns_empty_retrieval(self):
        """When all retrieved chunks are below threshold, retriever must return empty list."""
        docs_and_scores = [
            (self.doc_irrelevant1, 0.58),
            (self.doc_irrelevant2, 0.45),
        ]
        mock_store = self._create_mock_store(docs_and_scores)
        retriever = RetrivererService(vector_store=mock_store)

        results = retriever.retrive("unanswerable query", score_threshold=0.60)
        self.assertEqual(len(results), 0)

    def test_all_chunks_below_threshold_refusal_without_llm_call(self):
        """When no chunks pass threshold, RAGService returns standard refusal without calling LLM."""
        docs_and_scores = [
            (self.doc_irrelevant1, 0.58),
            (self.doc_irrelevant2, 0.45),
        ]
        mock_store = self._create_mock_store(docs_and_scores)
        retriever = RetrivererService(vector_store=mock_store)

        mock_llm = MagicMock()
        rag_service = RAGService(retriverer=retriever, llm=mock_llm)

        result = rag_service.invoke("What is the pediatric dosage of ibuprofen?")

        # Assert standard refusal string
        self.assertEqual(
            result["answer"],
            "I couldn't find this information in the uploaded documents.",
        )
        self.assertEqual(result["documents"], [])
        # Critical: LLM must NEVER be called when context is empty due to cutoff
        mock_llm.chat.assert_not_called()

    def test_boundary_score_behavior(self):
        """Exact threshold matches (>=) must be retained; scores strictly below (<) must be discarded."""
        doc_exact = Document(page_content="Exact boundary", metadata={"source": "d.pdf"})
        doc_above = Document(page_content="Above boundary", metadata={"source": "d.pdf"})
        doc_below = Document(page_content="Below boundary", metadata={"source": "d.pdf"})

        docs_and_scores = [
            (doc_above, 0.6001),
            (doc_exact, 0.6000),
            (doc_below, 0.5999),
        ]
        mock_store = self._create_mock_store(docs_and_scores)
        retriever = RetrivererService(vector_store=mock_store)

        results = retriever.retrive("boundary test", score_threshold=0.6000)
        self.assertEqual(len(results), 2)
        self.assertIn(doc_above, results)
        self.assertIn(doc_exact, results)
        self.assertNotIn(doc_below, results)

    def test_source_and_page_metadata_preserved(self):
        """Metadata such as filename, source path, page must be preserved on passing documents."""
        docs_and_scores = [
            (self.doc_high1, 0.85),
        ]
        mock_store = self._create_mock_store(docs_and_scores)
        retriever = RetrivererService(vector_store=mock_store)

        mock_llm = MagicMock()
        mock_llm.chat.return_value = AIMessage(content="Generated answer based on algo.pdf")
        rag_service = RAGService(retriverer=retriever, llm=mock_llm)

        result = rag_service.invoke("Tell me about algorithms")

        self.assertEqual(len(result["documents"]), 1)
        doc = result["documents"][0]
        self.assertEqual(doc.metadata["filename"], "algo.pdf")
        self.assertEqual(doc.metadata["page"], 10)
        self.assertEqual(doc.metadata["source"], "data/algo.pdf")

        # Verify context sent to LLM contains document name and page number
        call_args = mock_llm.chat.call_args[0][0]
        # Messages: [SystemMessage, HumanMessage]
        system_content = call_args[0].content
        self.assertIn("algo.pdf", system_content)
        self.assertIn("Page: 10", system_content)
        self.assertIn("High relevance chunk 1 about algorithms.", system_content)

    def test_default_threshold_from_settings(self):
        """Retriever must use settings.RELEVANCE_SCORE_THRESHOLD when none is passed."""
        self.assertEqual(settings.RELEVANCE_SCORE_THRESHOLD, 0.60)
        docs_and_scores = [
            (self.doc_high1, 0.65),
            (self.doc_irrelevant1, 0.55),
        ]
        mock_store = self._create_mock_store(docs_and_scores)
        retriever = RetrivererService(vector_store=mock_store)

        # Do not pass score_threshold, let it default to settings
        results = retriever.retrive("test query")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], self.doc_high1)

    def test_chroma_vector_store_delegation(self):
        """ChromaVectorStore delegation to underlying db.similarity_search_with_relevance_scores."""
        mock_db = MagicMock()
        mock_db.similarity_search_with_relevance_scores.return_value = [(self.doc_high1, 0.88)]

        store = ChromaVectorStore(db=mock_db)
        out = store.similarity_search_with_relevance_scores("query", k=3, score_threshold=0.7)

        mock_db.similarity_search_with_relevance_scores.assert_called_once_with(
            query="query", k=3, score_threshold=0.7
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][1], 0.88)


if __name__ == "__main__":
    unittest.main()
