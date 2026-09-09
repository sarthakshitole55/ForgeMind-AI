import io
import os
import sqlite3
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("GROQ_API_KEY", "gsk_test_dummy_key_12345678901234567890")

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from app.main import app
from app.config.settings import settings
from app.core.exceptions import (
    DocumentError,
    ForgeMindError,
    LLMError,
    ValidationError,
    VectorStoreError,
)
from app.rag.indexer import DocumentIndexer
from app.rag.rag_service import RAGService
from app.rag.retriverer.retriverer_service import RetrivererService
from app.services.document_service import DocumentService
from app.services.llm_services import LLMService, classify_llm_exception
from app.core.observability import ObservabilityService


def _create_valid_pdf_bytes() -> bytes:
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class TestTicket7ProductionReliability(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.data_dir = settings.get_data_dir()
        self.valid_pdf = _create_valid_pdf_bytes()
        self.cleanup_files = []

    def tearDown(self):
        for p in self.cleanup_files:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    # =========================================================================
    # A. Chat API Request Reliability & Sanitization
    # =========================================================================

    def test_chat_empty_messages_rejected_with_400(self):
        """Chat request with empty messages list must be rejected with 400 ValidationError."""
        response = self.client.post("/chat", json={"messages": []})
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertFalse(data.get("success", True))
        self.assertEqual(data.get("error", {}).get("type"), "ValidationError")
        self.assertIn("at least one message", data.get("detail", ""))

    def test_chat_empty_message_content_rejected_with_400(self):
        """Chat request with empty or whitespace message content must be rejected with 400."""
        response = self.client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "   \n\t  "}]},
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertFalse(data.get("success", True))
        self.assertEqual(data.get("error", {}).get("type"), "ValidationError")
        self.assertIn("cannot be empty", data.get("detail", ""))

    @patch("app.api.v1.chat.graph.invoke")
    def test_chat_llm_error_returns_sanitized_502(self, mock_invoke):
        """Chat workflow experiencing an LLM failure returns HTTP 502 with sanitized message."""
        mock_invoke.side_effect = LLMError("AI model service authentication failed.", status_code=502)
        response = self.client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "What is ResNet?"}]},
        )
        self.assertEqual(response.status_code, 502)
        data = response.json()
        self.assertFalse(data.get("success", True))
        self.assertEqual(data.get("error", {}).get("type"), "LLMError")
        self.assertEqual(data.get("detail"), "AI model service authentication failed.")
        # Ensure no tracebacks or internal file paths leak
        self.assertNotIn("Traceback", response.text)

    @patch("app.api.v1.chat.graph.invoke")
    def test_chat_vector_store_error_returns_sanitized_503(self, mock_invoke):
        """Chat workflow experiencing a vector store failure returns HTTP 503 with sanitized message."""
        mock_invoke.side_effect = VectorStoreError(
            "Vector store is temporarily unavailable for document retrieval.",
            status_code=503,
        )
        response = self.client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "What is ResNet?"}]},
        )
        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertFalse(data.get("success", True))
        self.assertEqual(data.get("error", {}).get("type"), "VectorStoreError")
        self.assertEqual(data.get("detail"), "Vector store is temporarily unavailable for document retrieval.")

    @patch("app.api.v1.chat.graph.invoke")
    def test_chat_unexpected_exception_returns_clean_500_without_traceback(self, mock_invoke):
        """Unexpected internal exception in chat returns generic 500 without leaking details."""
        mock_invoke.side_effect = RuntimeError("Fatal internal memory corruption /var/run/secret.sock")
        response = self.client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertFalse(data.get("success", True))
        self.assertNotIn("secret.sock", response.text)
        self.assertNotIn("Traceback", response.text)
        self.assertIn("unexpected error occurred", data.get("detail", "").lower())

    # =========================================================================
    # B. LLM Provider Failure Handling & Bounded Retries
    # =========================================================================

    def test_classify_llm_exception_types(self):
        """Verify exception classifier correctly partitions transient vs non-transient errors."""
        # 1. Auth failure: non-transient
        is_transient, msg, code = classify_llm_exception(Exception("401 Unauthorized: Invalid API key"))
        self.assertFalse(is_transient)
        self.assertIn("authentication failed", msg)
        self.assertEqual(code, 502)

        # 2. Rate limit: transient, retryable
        class RateLimitError(Exception):
            status_code = 429
        is_transient, msg, code = classify_llm_exception(RateLimitError("Rate limit exceeded"))
        self.assertTrue(is_transient)
        self.assertIn("rate limit exceeded", msg)
        self.assertEqual(code, 502)

        # 3. Timeout: transient, retryable
        is_transient, msg, code = classify_llm_exception(TimeoutError("Request timed out"))
        self.assertTrue(is_transient)
        self.assertIn("timed out", msg)
        self.assertEqual(code, 504)

        # 4. Connection error: transient, retryable
        is_transient, msg, code = classify_llm_exception(ConnectionError("Network connection reset"))
        self.assertTrue(is_transient)
        self.assertIn("connect", msg)
        self.assertEqual(code, 503)

        # 5. Bad request / 400: non-transient, fail fast
        class BadRequestError(Exception):
            status_code = 400
        is_transient, msg, code = classify_llm_exception(BadRequestError("Invalid argument"))
        self.assertFalse(is_transient)
        self.assertIn("Invalid request", msg)
        self.assertEqual(code, 502)

    def test_llm_transient_rate_limit_retries_and_succeeds(self):
        """Transient error (429) triggers bounded retry and succeeds on subsequent attempt."""
        service = LLMService(max_retries=2, initial_delay=0.0)
        mock_response = MagicMock()
        mock_response.content = "Successful answer"

        service.llm = MagicMock()
        service.llm.invoke.side_effect = [
            Exception("Rate limit 429: Too Many Requests"),
            mock_response,
        ]

        result = service.chat([HumanMessage(content="Test query")])
        self.assertEqual(result.content, "Successful answer")
        self.assertEqual(service.llm.invoke.call_count, 2)

    def test_llm_transient_rate_limit_exhausts_retries_and_raises_sanitized_error(self):
        """Transient error exceeding max_retries raises sanitized LLMError without leaking raw text."""
        service = LLMService(max_retries=2, initial_delay=0.0)
        service.llm = MagicMock()
        service.llm.invoke.side_effect = Exception("Rate limit 429: API key gsk_secret_123 rate exceeded")

        with self.assertRaises(LLMError) as ctx:
            service.chat([HumanMessage(content="Test query")])

        self.assertEqual(service.llm.invoke.call_count, 3)  # 1 initial + 2 retries
        self.assertIn("rate limit exceeded", ctx.exception.message.lower())
        self.assertNotIn("gsk_secret_123", ctx.exception.message)

    def test_llm_auth_failure_fails_fast_without_retrying(self):
        """Authentication error (401) must fail fast on the first attempt without retrying."""
        service = LLMService(max_retries=2, initial_delay=0.0)
        service.llm = MagicMock()
        service.llm.invoke.side_effect = Exception("401 Unauthorized: Invalid API Key")

        with self.assertRaises(LLMError) as ctx:
            service.chat([HumanMessage(content="Test query")])

        self.assertEqual(service.llm.invoke.call_count, 1)  # Only 1 attempt, zero retries
        self.assertIn("authentication failed", ctx.exception.message)

    def test_llm_timeout_transient_retries_and_raises_504(self):
        """Timeout errors retry up to max_retries and raise 504 on exhaustion."""
        service = LLMService(max_retries=1, initial_delay=0.0)
        service.llm = MagicMock()
        service.llm.invoke.side_effect = TimeoutError("Groq request timed out after 30s")

        with self.assertRaises(LLMError) as ctx:
            service.chat([HumanMessage(content="Test query")])

        self.assertEqual(service.llm.invoke.call_count, 2)  # 1 initial + 1 retry
        self.assertEqual(ctx.exception.status_code, 504)
        self.assertIn("timed out", ctx.exception.message)

    # =========================================================================
    # C. RAG Retrieval Failure vs Threshold Refusal
    # =========================================================================

    def test_retrieval_threshold_refusal_returns_standard_refusal(self):
        """Zero documents returned due to relevance cutoff returns 200 standard refusal."""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = []

        retriverer = RetrivererService(vector_store=mock_vs)
        rag_service = RAGService(retriverer=retriverer)

        result = rag_service.invoke("Unanswerable medical query")
        self.assertEqual(
            result["answer"],
            "I couldn't find this information in the uploaded documents.",
        )
        self.assertEqual(result["documents"], [])

    def test_retrieval_infrastructure_failure_raises_vector_store_error(self):
        """Vector store database crash raises VectorStoreError instead of returning refusal."""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.side_effect = sqlite3.OperationalError(
            "database is locked: /var/chroma/chroma.sqlite3"
        )

        retriverer = RetrivererService(vector_store=mock_vs)
        rag_service = RAGService(retriverer=retriverer)

        with self.assertRaises(VectorStoreError) as ctx:
            rag_service.invoke("Any question")

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(
            ctx.exception.message,
            "Vector store is temporarily unavailable for document retrieval.",
        )
        self.assertNotIn("chroma.sqlite3", ctx.exception.message)

    # =========================================================================
    # D. Document Indexing Consistency & Partial Vector Cleanup
    # =========================================================================

    def test_indexer_batch_failure_cleans_up_partial_vectors(self):
        """DocumentIndexer purges previously inserted batches if a subsequent batch fails."""
        mock_store = MagicMock()
        # Batch 1 succeeds, Batch 2 fails
        mock_store.add_documents.side_effect = [None, RuntimeError("Disk full writing batch 2")]

        indexer = DocumentIndexer(vector_store=mock_store)
        indexer.loader = MagicMock()
        # Return 2 pages
        indexer.loader.load.return_value = [
            Document(page_content="Page 1 content"),
            Document(page_content="Page 2 content"),
        ]
        indexer.chunker = MagicMock()
        # Return 40 chunks (forces 2 batches of size 32)
        indexer.chunker.split.return_value = [
            Document(page_content=f"Chunk {i}") for i in range(40)
        ]

        test_pdf = self.data_dir / "partial_test.pdf"
        test_pdf.write_bytes(self.valid_pdf)
        self.cleanup_files.append(test_pdf)

        with self.assertRaises(RuntimeError):
            indexer.index_pdf(str(test_pdf))

        # Verify that delete_document was invoked to purge partial vectors
        mock_store.delete_document.assert_called_once()
        args, kwargs = mock_store.delete_document.call_args
        self.assertEqual(args[0], "partial_test.pdf")

    def test_upload_failure_cleans_up_physical_file_and_vectors(self):
        """Upload endpoint removes disk file and calls vector cleanup on indexing exception."""
        test_filename = "failing_index_test.pdf"
        target_path = self.data_dir / test_filename
        self.cleanup_files.append(target_path)

        with patch("app.api.v1.upload.indexer.index_pdf") as mock_index:
            with patch("app.api.v1.upload.indexer.store.delete_document") as mock_del:
                mock_index.side_effect = RuntimeError("Chroma embedding index corrupted")

                response = self.client.post(
                    "/upload",
                    files={"file": (test_filename, io.BytesIO(self.valid_pdf), "application/pdf")},
                )

                self.assertEqual(response.status_code, 500)
                data = response.json()
                self.assertFalse(data.get("success", True))
                self.assertEqual(data.get("error", {}).get("type"), "DocumentError")
                self.assertEqual(data.get("detail"), "Failed to process and index the PDF document.")
                # Verify physical file was unlinked
                self.assertFalse(target_path.exists())
                # Verify vector cleanup was triggered
                mock_del.assert_called_once()

    # =========================================================================
    # E. Document Deletion Consistency (Ticket #2 Guarantee Preservation)
    # =========================================================================

    def test_deletion_vector_store_failure_preserves_physical_file(self):
        """Vector deletion failure must NOT delete physical file, keeping state consistent."""
        test_filename = "test_del_failure.pdf"
        target_path = self.data_dir / test_filename
        target_path.write_bytes(self.valid_pdf)
        self.cleanup_files.append(target_path)

        mock_vs = MagicMock()
        mock_vs.delete_document.side_effect = RuntimeError("ChromaDB connection closed")
        doc_service = DocumentService(vector_store=mock_vs)

        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            doc_service.delete_document(test_filename)

        self.assertEqual(ctx.exception.status_code, 500)
        self.assertEqual(ctx.exception.detail, "Failed to delete document vectors from vector store")
        # Physical file MUST still exist
        self.assertTrue(target_path.exists())

    def test_deletion_file_unlink_failure_produces_500_after_vector_deletion(self):
        """Physical unlink failure after vector deletion produces controlled HTTP 500."""
        test_filename = "test_unlink_failure.pdf"
        target_path = self.data_dir / test_filename
        target_path.write_bytes(self.valid_pdf)
        self.cleanup_files.append(target_path)

        mock_vs = MagicMock()
        mock_vs.delete_document.return_value = 5
        doc_service = DocumentService(vector_store=mock_vs)

        with patch.object(Path, "unlink", side_effect=PermissionError("Permission denied")):
            from fastapi import HTTPException
            with self.assertRaises(HTTPException) as ctx:
                doc_service.delete_document(test_filename)

            self.assertEqual(ctx.exception.status_code, 500)
            self.assertEqual(ctx.exception.detail, "Failed to delete physical document file")

    # =========================================================================
    # F. Observability & Credential Privacy
    # =========================================================================

    def test_observability_failure_does_not_break_retrieval(self):
        """Langfuse crash during retrieval recording never crashes the user request."""
        mock_vs = MagicMock()
        doc = Document(page_content="Content", metadata={"filename": "doc.pdf"})
        mock_vs.similarity_search_with_relevance_scores.return_value = [(doc, 0.85)]

        retriverer = RetrivererService(vector_store=mock_vs)

        with patch("app.core.observability.ObservabilityService.get_client") as mock_client:
            mock_client.return_value.create_event.side_effect = Exception("Langfuse network unreachable")
            # Should complete without error
            docs = retriverer.retrive("Sample query")
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].metadata["filename"], "doc.pdf")

    def test_error_logs_do_not_leak_api_credentials(self):
        """Error handling and logging must not print raw API credentials."""
        secret_key = "gsk_super_secret_production_key_xyz987"
        service = LLMService(max_retries=0, initial_delay=0.0)
        service.llm = MagicMock()
        service.llm.invoke.side_effect = Exception(f"401 Unauthorized using key {secret_key}")

        with self.assertRaises(LLMError) as ctx:
            service.chat([HumanMessage(content="Hello")])

        # Client error message must NOT leak the key
        self.assertNotIn(secret_key, ctx.exception.message)
