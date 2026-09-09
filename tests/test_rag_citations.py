import re
import unittest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app.rag.rag_service import RAGService
from app.rag.retriverer.retriverer_service import RetrivererService
from app.prompts.rag import RAG_SYSTEM_PROMPT


def extract_cited_sources(answer: str) -> list[str]:
    """Extracts filenames cited in [Source: <filename>, p. X] or [Source: <filename>] format."""
    pattern = r"\[Source:\s*([^,|\]]+?)(?:,\s*p\.?\s*\d+)?\s*\]"
    return [m.strip() for m in re.findall(pattern, answer)]


def validate_citations_in_context(answer: str, context_documents: list[Document]) -> bool:
    """Verifies that every cited document in the answer exists in the retrieved context documents."""
    cited = extract_cited_sources(answer)
    if not cited:
        return True
    valid_filenames = set()
    for doc in context_documents:
        fn = doc.metadata.get("filename")
        if fn:
            valid_filenames.add(fn)
        src = doc.metadata.get("source")
        if src:
            from pathlib import Path
            valid_filenames.add(Path(src).name)

    return all(c in valid_filenames for c in cited)


class TestRAGSourceCitations(unittest.TestCase):
    """
    Deterministic unit tests for RAG source citation formatting, prompt instructions,
    and source attribution integrity.
    """

    def test_chunk_receives_source_marker(self):
        """Every chunk must have a clearly formatted source marker."""
        doc = Document(
            page_content="Sample technical chunk content.",
            metadata={"filename": "manual.pdf", "page": 12},
        )
        formatted = RAGService.format_chunk(doc)
        self.assertTrue(formatted.startswith("[Source: manual.pdf | Page: 12]"))
        self.assertIn("Sample technical chunk content.", formatted)

    def test_page_metadata_included_when_available(self):
        """Page numbers, including 0-indexed page 0, must be included accurately."""
        doc_pg_10 = Document(page_content="Content A", metadata={"filename": "doc.pdf", "page": 10})
        self.assertEqual(RAGService.format_chunk(doc_pg_10), "[Source: doc.pdf | Page: 10]\nContent A")

        doc_pg_0 = Document(page_content="Cover Page", metadata={"filename": "doc.pdf", "page": 0})
        self.assertEqual(RAGService.format_chunk(doc_pg_0), "[Source: doc.pdf | Page: 0]\nCover Page")

    def test_missing_page_metadata_omits_page_without_fake_numbers(self):
        """Missing or unknown page metadata must omit the Page field rather than inventing one."""
        doc_no_page = Document(page_content="No page info", metadata={"filename": "doc.pdf"})
        self.assertEqual(RAGService.format_chunk(doc_no_page), "[Source: doc.pdf]\nNo page info")

        doc_none_page = Document(page_content="None page info", metadata={"filename": "doc.pdf", "page": None})
        self.assertEqual(RAGService.format_chunk(doc_none_page), "[Source: doc.pdf]\nNone page info")

        doc_unknown_page = Document(page_content="Unknown page info", metadata={"filename": "doc.pdf", "page": "Unknown"})
        self.assertEqual(RAGService.format_chunk(doc_unknown_page), "[Source: doc.pdf]\nUnknown page info")

    def test_source_filenames_preserved_exactly(self):
        """Exact document filenames must be preserved in the marker without distortion."""
        long_filename = "competitive-programming-in-python-128-algorithms-to-develop-your-coding-skills.pdf"
        doc = Document(
            page_content="Algorithm text",
            metadata={"filename": long_filename, "page": 44},
        )
        formatted = RAGService.format_chunk(doc)
        self.assertIn(f"[Source: {long_filename} | Page: 44]", formatted)

        # Derivation from source path when filename key is absent
        doc_from_path = Document(
            page_content="Path text",
            metadata={"source": "/var/data/books/deep_learning.pdf", "page": 100},
        )
        self.assertEqual(
            RAGService.format_chunk(doc_from_path),
            "[Source: deep_learning.pdf | Page: 100]\nPath text",
        )

    def test_prompt_contains_citation_instructions(self):
        """RAG prompt must explicitly instruct the LLM on citation format, exactness, and refusal."""
        self.assertIn("[Source: filename, p. X]", RAG_SYSTEM_PROMPT)
        self.assertIn("[Source: filename]", RAG_SYSTEM_PROMPT)
        self.assertIn("Cite the source of factual claims", RAG_SYSTEM_PROMPT)
        self.assertIn("Do NOT invent citations", RAG_SYSTEM_PROMPT)
        self.assertIn("I couldn't find this information in the uploaded documents.", RAG_SYSTEM_PROMPT)

    def test_existing_refusal_behavior_unchanged(self):
        """Refusal response and empty documents list must be preserved when no documents exist."""
        mock_retriever = MagicMock()
        mock_retriever.retrive.return_value = []
        mock_llm = MagicMock()

        rag = RAGService(retriverer=mock_retriever, llm=mock_llm)
        result = rag.invoke("Unknown question")

        self.assertEqual(
            result["answer"],
            "I couldn't find this information in the uploaded documents.",
        )
        self.assertEqual(result["documents"], [])
        mock_llm.chat.assert_not_called()

    def test_existing_relevance_cutoff_behavior_unchanged(self):
        """Relevance cutoff filters low-scoring chunks before prompt creation."""
        mock_store = MagicMock()
        doc_valid = Document(page_content="Valid", metadata={"filename": "a.pdf", "page": 1})
        mock_store.similarity_search_with_relevance_scores.return_value = [
            (doc_valid, 0.75),
        ]
        retriever = RetrivererService(vector_store=mock_store)

        mock_llm = MagicMock()
        mock_llm.chat.return_value = AIMessage(content="Answer [Source: a.pdf, p. 1]")
        rag = RAGService(retriverer=retriever, llm=mock_llm)

        result = rag.invoke("Valid question")
        self.assertEqual(len(result["documents"]), 1)
        self.assertIn("[Source: a.pdf | Page: 1]", mock_llm.chat.call_args[0][0][0].content)

    def test_citation_attribution_validation(self):
        """Verify that citations can only reference sources actually present in retrieved context."""
        retrieved_docs = [
            Document(page_content="Text A", metadata={"filename": "asset_managers.pdf", "page": 10}),
            Document(page_content="Text B", metadata={"filename": "python_algorithms.pdf", "page": 20}),
        ]

        # Valid citation matching context
        valid_answer = "The ONC algorithm clusters matrices [Source: asset_managers.pdf, p. 10]."
        self.assertTrue(validate_citations_in_context(valid_answer, retrieved_docs))

        # Invalid citation referencing non-retrieved document
        hallucinated_answer = "According to medical science [Source: pediatric_journal.pdf, p. 4], dosage is 10mg."
        self.assertFalse(validate_citations_in_context(hallucinated_answer, retrieved_docs))

        # No citations (e.g. general phrasing)
        uncited_answer = "The answer is 42."
        self.assertTrue(validate_citations_in_context(uncited_answer, retrieved_docs))


if __name__ == "__main__":
    unittest.main()
