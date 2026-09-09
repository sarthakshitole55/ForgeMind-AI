import io
import logging
import unittest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app.config.settings import settings
from app.core.observability import ObservabilityService
from app.rag.rag_service import RAGService
from app.rag.retriverer.retriverer_service import RetrivererService


class TestLangfuseObservability(unittest.TestCase):
    """
    Deterministic offline unit tests for Langfuse observability integration.
    Verifies error isolation, secret privacy, metadata fidelity, and safety without live credentials.
    """

    def setUp(self):
        ObservabilityService.reset()

    def tearDown(self):
        ObservabilityService.reset()

    def test_langfuse_disabled_by_default_without_keys(self):
        """When keys are absent, observability must be cleanly disabled."""
        with patch.object(settings, "LANGFUSE_PUBLIC_KEY", ""), \
             patch.object(settings, "LANGFUSE_SECRET_KEY", ""):
            self.assertFalse(ObservabilityService.is_enabled())
            self.assertIsNone(ObservabilityService.get_client())
            self.assertIsNone(ObservabilityService.get_callback_handler())

            with ObservabilityService.start_request_trace("test", "hello") as (trace_id, handler):
                self.assertIsNone(trace_id)
                self.assertIsNone(handler)

    def test_langfuse_configuration_loaded_correctly(self):
        """Settings must expose Langfuse configuration properties."""
        self.assertTrue(hasattr(settings, "LANGFUSE_PUBLIC_KEY"))
        self.assertTrue(hasattr(settings, "LANGFUSE_SECRET_KEY"))
        self.assertTrue(hasattr(settings, "LANGFUSE_HOST"))
        self.assertTrue(hasattr(settings, "LANGFUSE_ENABLED"))

    def test_explicitly_disabled_flag(self):
        """When LANGFUSE_ENABLED is False, observability is disabled even if keys are set."""
        with patch.object(settings, "LANGFUSE_ENABLED", False), \
             patch.object(settings, "LANGFUSE_PUBLIC_KEY", "pk-test"), \
             patch.object(settings, "LANGFUSE_SECRET_KEY", "sk-test"):
            self.assertFalse(ObservabilityService.is_enabled())
            self.assertIsNone(ObservabilityService.get_client())

    def test_observability_failures_do_not_break_request_handling(self):
        """If Langfuse client raises an exception during trace start, request handling must continue."""
        with patch.object(ObservabilityService, "is_enabled", return_value=True), \
             patch.object(ObservabilityService, "get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.start_observation.side_effect = RuntimeError("Langfuse API connection timeout")
            mock_get_client.return_value = mock_client

            # Must not raise RuntimeError
            with ObservabilityService.start_request_trace("chat", "query") as (trace_id, handler):
                self.assertIsNone(trace_id)
                self.assertIsNone(handler)

    def test_no_secrets_appear_in_logs(self):
        """Verify API keys and secrets are never exposed in log outputs."""
        secret_value = "sk-lf-super-secret-unique-998877"
        public_value = "pk-lf-public-unique-112233"

        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        test_logger = logging.getLogger("app.core.observability")
        test_logger.addHandler(handler)

        try:
            with patch.object(settings, "LANGFUSE_PUBLIC_KEY", public_value), \
                 patch.object(settings, "LANGFUSE_SECRET_KEY", secret_value), \
                 patch.object(settings, "LANGFUSE_ENABLED", True):

                # Trigger client get and trace start
                with ObservabilityService.start_request_trace("secret_test", "query") as _:
                    pass
                ObservabilityService.record_retrieval("q", [], 0.60, 0.05)

            log_output = log_capture.getvalue()
            self.assertNotIn(secret_value, log_output)
        finally:
            test_logger.removeHandler(handler)

    def test_retrieval_metadata_captured_correctly(self):
        """Retrieval event captures count, threshold, bounded metadata, and scores."""
        mock_client = MagicMock()
        with patch.object(ObservabilityService, "is_enabled", return_value=True), \
             patch.object(ObservabilityService, "get_client", return_value=mock_client):

            docs = [
                Document(page_content="Content 1", metadata={"filename": "doc1.pdf", "page": 10, "relevance_score": 0.82}),
                Document(page_content="Content 2", metadata={"filename": "doc2.pdf", "relevance_score": 0.74}),
            ]

            ObservabilityService.record_retrieval(
                query="test retrieval",
                retrieved_documents=docs,
                relevance_threshold=0.60,
                latency_seconds=0.042,
                raw_candidates_count=5,
            )

            mock_client.create_event.assert_called_once()
            call_kwargs = mock_client.create_event.call_args.kwargs
            self.assertEqual(call_kwargs["name"], "rag_retrieval")
            self.assertEqual(call_kwargs["input"]["query"], "test retrieval")

            output = call_kwargs["output"]
            self.assertEqual(output["chunks_retrieved"], 2)
            self.assertFalse(output["threshold_refused"])
            self.assertEqual(len(output["sources"]), 2)
            self.assertEqual(output["sources"][0], {"filename": "doc1.pdf", "page": 10, "relevance_score": 0.82})
            self.assertEqual(output["sources"][1], {"filename": "doc2.pdf", "relevance_score": 0.74})

            metadata = call_kwargs["metadata"]
            self.assertEqual(metadata["relevance_threshold"], 0.60)
            self.assertEqual(metadata["raw_candidates_count"], 5)
            self.assertEqual(metadata["retrieval_latency_seconds"], 0.042)

    def test_retrieval_refusal_event_captured_when_empty(self):
        """When 0 chunks pass threshold, threshold_refused is flagged as True."""
        mock_client = MagicMock()
        with patch.object(ObservabilityService, "is_enabled", return_value=True), \
             patch.object(ObservabilityService, "get_client", return_value=mock_client):

            ObservabilityService.record_retrieval(
                query="out of domain",
                retrieved_documents=[],
                relevance_threshold=0.60,
                latency_seconds=0.035,
                raw_candidates_count=5,
            )

            mock_client.create_event.assert_called_once()
            output = mock_client.create_event.call_args.kwargs["output"]
            self.assertEqual(output["chunks_retrieved"], 0)
            self.assertTrue(output["threshold_refused"])
            self.assertEqual(output["sources"], [])

    def test_citation_and_cutoff_behavior_intact_with_observability(self):
        """Full pipeline invocation preserves relevance cutoff, refusal, and citation formatting."""
        mock_store = MagicMock()
        doc = Document(page_content="Deterministic chunk.", metadata={"filename": "manual.pdf", "page": 7})
        mock_store.similarity_search_with_relevance_scores.return_value = [(doc, 0.78)]

        retriever = RetrivererService(vector_store=mock_store)
        mock_llm = MagicMock()
        mock_llm.chat.return_value = AIMessage(content="Verified answer [Source: manual.pdf, p. 7].")
        rag = RAGService(retriverer=retriever, llm=mock_llm)

        # Execute with observability trace wrapper
        with ObservabilityService.start_request_trace("test_run", "query") as (_tid, handler):
            callbacks = [handler] if handler else None
            result = rag.invoke("What is in manual?", callbacks=callbacks)

        self.assertEqual(len(result["documents"]), 1)
        self.assertEqual(result["documents"][0].metadata["relevance_score"], 0.78)
        self.assertIn("[Source: manual.pdf, p. 7]", result["answer"])


if __name__ == "__main__":
    unittest.main()
