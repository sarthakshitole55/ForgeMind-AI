import io
import os
import re
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("GROQ_API_KEY", "gsk_test_dummy_key_12345678901234567890")

from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app.main import app
from app.config.settings import settings
from app.core.exceptions import LLMError, VectorStoreError
from app.core.middleware import SAFE_REQUEST_ID_REGEX
from app.schemas.chat import ChatResponse
from app.schemas.document import DeleteDocumentResponse, DocumentInfo, UploadResponse


def _create_valid_pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class TestAPIHardeningBoundary(unittest.TestCase):

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
    # 1. Request Models & Validation
    # =========================================================================

    @patch("app.api.v1.chat.graph.invoke")
    def test_valid_chat_request_returns_expected_response_schema(self, mock_invoke):
        """Valid chat request returns HTTP 200 with schema matching ChatResponse."""
        mock_invoke.return_value = {
            "answer": "ResNet utilizes skip connections to address gradient vanishing.",
            "route": "rag",
            "question": "What is ResNet?",
            "context": "[Source: resnet.pdf | Page: 1]\nDeep Residual Learning",
        }

        response = self.client.post(
            "/api/v1/chat",
            json={"messages": [{"role": "user", "content": "What is ResNet?"}]},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Validate with Pydantic response model
        parsed = ChatResponse(**data)
        self.assertEqual(parsed.answer, "ResNet utilizes skip connections to address gradient vanishing.")
        self.assertEqual(parsed.route, "rag")
        self.assertEqual(parsed.question, "What is ResNet?")
        self.assertIsNotNone(parsed.context)

    def test_empty_messages_returns_http_400(self):
        """Empty messages list must return HTTP 400 with ValidationError."""
        for path in ("/api/v1/chat", "/chat"):
            with self.subTest(path=path):
                response = self.client.post(path, json={"messages": []})
                self.assertEqual(response.status_code, 400)
                data = response.json()
                self.assertFalse(data.get("success", True))
                self.assertEqual(data.get("error", {}).get("type"), "ValidationError")
                self.assertIn("at least one message", data.get("detail", ""))

    def test_blank_message_content_returns_http_400(self):
        """Blank or whitespace-only message content must return HTTP 400."""
        for path in ("/api/v1/chat", "/chat"):
            with self.subTest(path=path):
                response = self.client.post(
                    path,
                    json={"messages": [{"role": "user", "content": "   \n\t  "}]},
                )
                self.assertEqual(response.status_code, 400)
                data = response.json()
                self.assertFalse(data.get("success", True))
                self.assertEqual(data.get("error", {}).get("type"), "ValidationError")
                self.assertIn("cannot be empty", data.get("detail", ""))

    def test_invalid_request_structure_returns_http_422(self):
        """Invalid request structure (e.g. unsupported role) returns HTTP 422 with clean schema."""
        response = self.client.post(
            "/api/v1/chat",
            json={"messages": [{"role": "invalid_role", "content": "hello"}]},
        )
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertFalse(data.get("success", True))
        self.assertEqual(data.get("error", {}).get("type"), "ValidationError")
        self.assertIn("role", data.get("detail", ""))
        self.assertNotIn("Traceback", response.text)

    # =========================================================================
    # 2. Sanitized Error Contracts Across Status Codes
    # =========================================================================

    @patch("app.api.v1.chat.graph.invoke")
    def test_provider_failure_preserves_sanitized_error_contract(self, mock_invoke):
        """AI model provider failure returns HTTP 502 with sanitized error contract."""
        mock_invoke.side_effect = LLMError("AI model service rate limit exceeded. Please try again later.", status_code=502)
        response = self.client.post(
            "/api/v1/chat",
            json={"messages": [{"role": "user", "content": "What is ResNet?"}]},
        )
        self.assertEqual(response.status_code, 502)
        data = response.json()
        self.assertFalse(data.get("success", True))
        self.assertEqual(data.get("error", {}).get("type"), "LLMError")
        self.assertIn("rate limit exceeded", data.get("detail", ""))
        self.assertNotIn("Traceback", response.text)

    @patch("app.api.v1.chat.graph.invoke")
    def test_retrieval_failure_preserves_sanitized_error_contract(self, mock_invoke):
        """Vector store retrieval failure returns HTTP 503 with sanitized error contract."""
        mock_invoke.side_effect = VectorStoreError(
            "Vector store is temporarily unavailable for document retrieval.",
            status_code=503,
        )
        response = self.client.post(
            "/api/v1/chat",
            json={"messages": [{"role": "user", "content": "What is ResNet?"}]},
        )
        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertFalse(data.get("success", True))
        self.assertEqual(data.get("error", {}).get("type"), "VectorStoreError")
        self.assertEqual(data.get("detail"), "Vector store is temporarily unavailable for document retrieval.")
        self.assertNotIn("Traceback", response.text)

    @patch("app.api.v1.chat.graph.invoke")
    def test_unexpected_exception_returns_sanitized_500(self, mock_invoke):
        """Unexpected internal exception returns clean HTTP 500 without leaking internals."""
        mock_invoke.side_effect = RuntimeError("Fatal disk corruption at /var/lib/chroma/chroma.sqlite3")
        response = self.client.post(
            "/api/v1/chat",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertFalse(data.get("success", True))
        self.assertIn("unexpected error occurred", data.get("detail", "").lower())
        self.assertNotIn("/var/lib/chroma", response.text)
        self.assertNotIn("Traceback", response.text)

    # =========================================================================
    # 3. Request ID Correlation & Propagation
    # =========================================================================

    def test_request_id_generated_when_absent(self):
        """Every response must contain an X-Request-ID header generated if absent in request."""
        response = self.client.get("/health")
        self.assertIn("X-Request-ID", response.headers)
        request_id = response.headers["X-Request-ID"]
        self.assertTrue(SAFE_REQUEST_ID_REGEX.match(request_id), f"Invalid request ID: {request_id}")

    def test_request_id_returned_across_all_endpoints(self):
        """Verify X-Request-ID header is present on GET, POST, and error responses."""
        endpoints = [
            ("GET", "/"),
            ("GET", "/health"),
            ("GET", "/health/live"),
            ("GET", "/health/ready"),
            ("GET", "/api/v1/health"),
            ("GET", "/documents/"),
            ("POST", "/chat"),  # Error response (400)
        ]
        for method, path in endpoints:
            with self.subTest(method=method, path=path):
                if method == "GET":
                    res = self.client.get(path)
                else:
                    res = self.client.post(path, json={"messages": []})
                self.assertIn("X-Request-ID", res.headers, f"Missing X-Request-ID on {method} {path}")

    def test_valid_incoming_request_id_preserved(self):
        """Valid incoming X-Request-ID header is preserved and echoed back."""
        custom_id = "client-trace-id-abcdef-123456"
        response = self.client.get("/health", headers={"X-Request-ID": custom_id})
        self.assertEqual(response.headers.get("X-Request-ID"), custom_id)

    def test_malformed_incoming_request_id_sanitized(self):
        """Malformed or oversized X-Request-ID is safely replaced with a valid UUID."""
        malformed_ids = [
            "id with spaces inside",
            "malicious-id'; DROP TABLE users;--",
            "oversized_" + "x" * 100,
            "<script>alert(1)</script>",
        ]
        for bad_id in malformed_ids:
            with self.subTest(bad_id=bad_id):
                response = self.client.get("/health", headers={"X-Request-ID": bad_id})
                resulting_id = response.headers.get("X-Request-ID")
                self.assertNotEqual(resulting_id, bad_id)
                self.assertTrue(SAFE_REQUEST_ID_REGEX.match(resulting_id))

    # =========================================================================
    # 4. Health & Readiness Endpoints
    # =========================================================================

    def test_liveness_endpoint_succeeds_without_external_services(self):
        """Liveness endpoints return 200 without calling external LLMs or networks."""
        for path in ("/health", "/health/live", "/api/v1/health", "/api/v1/health/live"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data["status"], "ok")
                self.assertEqual(data["project"], "ForgeMind AI")
                self.assertEqual(data["version"], "0.1.0")

    def test_readiness_endpoint_reports_ready_when_healthy(self):
        """Readiness endpoint returns 200 ready when local storage and ChromaDB are available."""
        for path in ("/health/ready", "/api/v1/health/ready"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data["status"], "ready")
                self.assertEqual(data["details"]["storage"], "ready")
                self.assertEqual(data["details"]["vector_store"], "ready")

    @patch("app.rag.vectorstore.chroma_store.ChromaVectorStore")
    def test_readiness_endpoint_reports_503_when_vectorstore_unhealthy(self, mock_chroma_cls):
        """Readiness endpoint returns HTTP 503 not_ready if vector store is unavailable."""
        mock_instance = MagicMock()
        mock_instance.db._collection.count.side_effect = RuntimeError("SQLite database locked")
        mock_chroma_cls.return_value = mock_instance

        response = self.client.get("/health/ready")
        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertEqual(data["status"], "not_ready")
        self.assertEqual(data["details"]["vector_store"], "unavailable")

    # =========================================================================
    # 5. Observability Isolation & Trace Correlation
    # =========================================================================

    @patch("app.api.v1.chat.graph.invoke")
    def test_langfuse_disabled_does_not_affect_api_behavior(self, mock_invoke):
        """When Langfuse is disabled, chat endpoint operates normally with request IDs."""
        mock_invoke.return_value = {
            "answer": "Test answer",
            "route": "rag",
            "question": "Test query",
        }

        with patch("app.core.observability.ObservabilityService.is_enabled", return_value=False):
            response = self.client.post(
                "/api/v1/chat",
                json={"messages": [{"role": "user", "content": "Test query"}]},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["answer"], "Test answer")
            self.assertIn("X-Request-ID", response.headers)

    @patch("app.api.v1.chat.graph.invoke")
    def test_request_id_passed_to_langfuse_metadata(self, mock_invoke):
        """Request ID is included in trace metadata when Langfuse is active."""
        mock_invoke.return_value = {"answer": "Answer", "route": "rag", "question": "Q"}

        custom_id = "test-langfuse-trace-id-123"
        with patch("app.core.observability.ObservabilityService.start_request_trace") as mock_trace:
            mock_trace.return_value.__enter__.return_value = ("trace_id", None)
            mock_trace.return_value.__exit__.return_value = False

            response = self.client.post(
                "/api/v1/chat",
                headers={"X-Request-ID": custom_id},
                json={"messages": [{"role": "user", "content": "Q"}]},
            )
            self.assertEqual(response.status_code, 200)
            # Verify start_request_trace received metadata with request_id
            mock_trace.assert_called_once()
            _, kwargs = mock_trace.call_args
            self.assertEqual(kwargs.get("metadata", {}).get("request_id"), custom_id)

    # =========================================================================
    # 6. CORS Configuration & Headers
    # =========================================================================

    def test_cors_preflight_request_handling(self):
        """CORS OPTIONS preflight returns configured allowed origin and methods, and actual request exposes X-Request-ID."""
        # 1. Preflight OPTIONS check
        options_res = self.client.options(
            "/api/v1/chat",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type,X-Request-ID",
            },
        )
        self.assertEqual(options_res.status_code, 200)
        self.assertIn(options_res.headers.get("access-control-allow-origin"), ("*", "http://localhost:3000"))
        self.assertIn("POST", options_res.headers.get("access-control-allow-methods", ""))

        # 2. Actual GET/POST request check with Origin header
        actual_res = self.client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"},
        )
        self.assertEqual(actual_res.status_code, 200)
        self.assertIn(actual_res.headers.get("access-control-allow-origin"), ("*", "http://localhost:3000"))
        exposed_headers = actual_res.headers.get("access-control-expose-headers", "")
        self.assertIn("X-Request-ID", exposed_headers)

    # =========================================================================
    # 7. Response Models Across Endpoints
    # =========================================================================

    def test_response_models_match_actual_responses(self):
        """Verify response models match actual payloads for documents and upload."""
        # 1. Documents list
        test_file = self.data_dir / "sample_doc.pdf"
        test_file.write_bytes(self.valid_pdf)
        self.cleanup_files.append(test_file)

        res = self.client.get("/api/v1/documents/")
        self.assertEqual(res.status_code, 200)
        docs = [DocumentInfo(**item) for item in res.json()]
        self.assertTrue(any(d.name == "sample_doc.pdf" for d in docs))

        # 2. Upload
        with patch("app.rag.indexer.DocumentIndexer.index_pdf", return_value={"pages": 1, "chunks": 2}):
            res_upload = self.client.post(
                "/api/v1/upload",
                files={"file": ("upload_model_test.pdf", io.BytesIO(self.valid_pdf), "application/pdf")},
            )
            self.assertEqual(res_upload.status_code, 200)
            upload_parsed = UploadResponse(**res_upload.json())
            self.assertEqual(upload_parsed.filename, "upload_model_test.pdf")
            self.assertEqual(upload_parsed.pages, 1)
            self.assertEqual(upload_parsed.chunks, 2)
            self.cleanup_files.append(self.data_dir / "upload_model_test.pdf")

        # 3. Delete
        with patch("app.services.document_service.DocumentService.delete_document", return_value=True):
            res_del = self.client.delete("/api/v1/documents/sample_doc.pdf")
            self.assertEqual(res_del.status_code, 200)
            del_parsed = DeleteDocumentResponse(**res_del.json())
            self.assertIn("deleted successfully", del_parsed.message)

    # =========================================================================
    # 8. Canonical /api/v1 and Compatibility Paths
    # =========================================================================

    def test_api_v1_and_unversioned_paths_are_identical(self):
        """Verify both /api/v1/... and legacy unversioned routes are functional."""
        pairs = [
            ("/api/v1/health", "/health"),
            ("/api/v1/documents/", "/documents/"),
        ]
        for v1_path, legacy_path in pairs:
            with self.subTest(v1=v1_path, legacy=legacy_path):
                res_v1 = self.client.get(v1_path)
                res_legacy = self.client.get(legacy_path)
                self.assertEqual(res_v1.status_code, res_legacy.status_code)
                self.assertEqual(res_v1.json(), res_legacy.json())
