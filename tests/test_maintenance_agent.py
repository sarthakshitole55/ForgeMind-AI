import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.agents.maintenance_agent import MaintenanceAgent, maintenance_node
from app.agents.supervisor import supervisor
from app.main import app
from app.schemas.maintenance import (
    ActionExecutionResult,
    IssueSeverity,
    MaintenanceActionType,
    MaintenanceReport,
    MaintenanceStatus,
)
from app.config.settings import Settings
from app.schemas.router import RouteDecision
from app.services.maintenance_service import MaintenanceService


class TestMaintenanceRouting(unittest.TestCase):
    """Verifies that supervisor routes maintenance queries to Maintenance Agent and RAG queries to RAG."""

    @patch("app.agents.supervisor.RouterAgent.route")
    def test_maintenance_request_routes_to_maintenance(self, mock_route):
        mock_route.return_value = RouteDecision(
            route="maintenance",
            reason="User requested document consistency check",
        )
        state = {"question": "Check document indexing consistency and health"}
        decision = supervisor(state)
        self.assertEqual(decision["route"], "maintenance")

    @patch("app.agents.supervisor.RouterAgent.route")
    def test_rag_request_routes_to_rag(self, mock_route):
        mock_route.return_value = RouteDecision(
            route="rag",
            reason="User asked about document contents",
        )
        state = {"question": "What is the refund policy in the employee manual?"}
        decision = supervisor(state)
        self.assertEqual(decision["route"], "rag")


class TestMaintenanceInspectionAndConsistency(unittest.TestCase):
    """Verifies deterministic inspection, health checks, and consistency detection."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.mock_store = MagicMock()
        self.mock_indexer = MagicMock()
        self.service = MaintenanceService(
            vector_store=self.mock_store,
            indexer=self.mock_indexer,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_healthy_system_returns_healthy_status(self):
        # Setup: 1 physical file, matching 1 indexed doc, 5 chunks
        (self.data_dir / "valid_doc.pdf").write_bytes(b"%PDF-1.4 test document content")
        self.mock_store.db._collection.count.return_value = 5
        self.mock_store.db._collection.get.return_value = {
            "metadatas": [{"filename": "valid_doc.pdf", "source": "data/valid_doc.pdf"}]
        }
        self.mock_store.get_document_chunks.return_value = ["c1", "c2", "c3", "c4", "c5"]

        with patch.object(Settings, "get_data_dir", return_value=self.data_dir):
            report = self.service.inspect_system_and_consistency()

        self.assertEqual(report.status, MaintenanceStatus.HEALTHY)
        self.assertEqual(report.health.storage, "healthy")
        self.assertEqual(report.health.vector_store, "healthy")
        self.assertEqual(report.health.physical_document_count, 1)
        self.assertEqual(report.health.vector_chunk_count, 5)
        self.assertEqual(report.health.indexed_document_count, 1)
        self.assertEqual(len(report.issues), 0)

    def test_missing_storage_dependency_returns_error_status(self):
        non_existent_dir = self.data_dir / "missing_subdir"

        with patch.object(Settings, "get_data_dir", return_value=non_existent_dir):
            report = self.service.inspect_system_and_consistency()

        self.assertEqual(report.status, MaintenanceStatus.ERROR)
        self.assertEqual(report.health.storage, "unhealthy")
        storage_issues = [i for i in report.issues if i.issue_type == "STORAGE_UNAVAILABLE"]
        self.assertTrue(len(storage_issues) >= 1)

    def test_vector_store_failure_returns_error_status(self):
        self.mock_store.db._collection.count.side_effect = RuntimeError("ChromaDB connection refused")

        with patch.object(Settings, "get_data_dir", return_value=self.data_dir):
            report = self.service.inspect_system_and_consistency()

        self.assertEqual(report.status, MaintenanceStatus.ERROR)
        self.assertEqual(report.health.vector_store, "unhealthy")
        vs_issues = [i for i in report.issues if i.issue_type == "VECTOR_STORE_UNAVAILABLE"]
        self.assertTrue(len(vs_issues) >= 1)

    def test_case_a_missing_index_detected(self):
        """Case A: Physical file exists but has no vector index in ChromaDB."""
        (self.data_dir / "unindexed.pdf").write_bytes(b"%PDF-1.4 file with no vectors")
        self.mock_store.db._collection.count.return_value = 0
        self.mock_store.db._collection.get.return_value = {"metadatas": []}
        self.mock_store.get_document_chunks.return_value = []

        with patch.object(Settings, "get_data_dir", return_value=self.data_dir):
            report = self.service.inspect_system_and_consistency()

        self.assertEqual(report.status, MaintenanceStatus.REPAIR_REQUIRED)
        missing_issues = [i for i in report.issues if i.issue_type == "MISSING_INDEX"]
        self.assertEqual(len(missing_issues), 1)
        self.assertEqual(missing_issues[0].resource, "unindexed.pdf")
        self.assertEqual(missing_issues[0].recommended_action, MaintenanceActionType.REINDEX_DOCUMENT)
        self.assertTrue(missing_issues[0].auto_repair_safe)

    def test_case_b_orphaned_vectors_detected(self):
        """Case B: Vector index references document that no longer exists in storage."""
        # No physical files in data_dir, but Chroma contains metadatas for deleted.pdf
        self.mock_store.db._collection.count.return_value = 3
        self.mock_store.db._collection.get.return_value = {
            "metadatas": [{"filename": "deleted_document.pdf"}]
        }

        with patch.object(Settings, "get_data_dir", return_value=self.data_dir):
            report = self.service.inspect_system_and_consistency()

        self.assertEqual(report.status, MaintenanceStatus.REPAIR_REQUIRED)
        orphan_issues = [i for i in report.issues if i.issue_type == "ORPHANED_VECTORS"]
        self.assertEqual(len(orphan_issues), 1)
        self.assertEqual(orphan_issues[0].resource, "deleted_document.pdf")
        self.assertEqual(orphan_issues[0].recommended_action, MaintenanceActionType.REMOVE_ORPHAN_VECTOR)
        self.assertTrue(orphan_issues[0].auto_repair_safe)

    def test_case_c_corrupt_empty_file_detected(self):
        """Case C: Physical file has 0 bytes (corrupt/empty)."""
        (self.data_dir / "empty.pdf").write_bytes(b"")
        self.mock_store.db._collection.count.return_value = 0
        self.mock_store.db._collection.get.return_value = {"metadatas": []}

        with patch.object(Settings, "get_data_dir", return_value=self.data_dir):
            report = self.service.inspect_system_and_consistency()

        self.assertEqual(report.status, MaintenanceStatus.REPAIR_REQUIRED)
        corrupt_issues = [i for i in report.issues if i.issue_type == "CORRUPT_FILE"]
        self.assertEqual(len(corrupt_issues), 1)
        self.assertEqual(corrupt_issues[0].resource, "empty.pdf")
        self.assertFalse(corrupt_issues[0].auto_repair_safe)


class TestMaintenanceActionsAndSafety(unittest.TestCase):
    """Verifies allowlist enforcement, destructive protections, and verified execution."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.mock_store = MagicMock()
        self.mock_indexer = MagicMock()
        self.service = MaintenanceService(
            vector_store=self.mock_store,
            indexer=self.mock_indexer,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_reindex_action_executes_and_verifies(self):
        pdf_file = self.data_dir / "sample.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 Content for indexing")

        self.mock_indexer.index_pdf.return_value = {"pages": 2, "chunks": 6}
        self.mock_store.get_document_chunks.return_value = ["c1", "c2", "c3", "c4", "c5", "c6"]

        with patch.object(Settings, "get_data_dir", return_value=self.data_dir):
            res = self.service.execute_action(MaintenanceActionType.REINDEX_DOCUMENT, "sample.pdf")

        self.assertTrue(res.success)
        self.assertEqual(res.action, MaintenanceActionType.REINDEX_DOCUMENT)
        self.assertEqual(res.target, "sample.pdf")
        self.mock_store.delete_document.assert_called_once()
        self.mock_indexer.index_pdf.assert_called_once_with(str(pdf_file.resolve()))

    def test_reindex_rejected_if_file_missing(self):
        with patch.object(Settings, "get_data_dir", return_value=self.data_dir):
            res = self.service.execute_action(MaintenanceActionType.REINDEX_DOCUMENT, "nonexistent.pdf")

        self.assertFalse(res.success)
        self.assertIn("does not exist", res.message)
        self.mock_indexer.index_pdf.assert_not_called()

    def test_reindex_verification_failure_detected(self):
        """When indexer completes but 0 chunks exist in Chroma, report failure."""
        pdf_file = self.data_dir / "bad.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 corrupted text layer")

        self.mock_indexer.index_pdf.return_value = {"pages": 1, "chunks": 0}
        self.mock_store.get_document_chunks.return_value = []

        with patch.object(Settings, "get_data_dir", return_value=self.data_dir):
            res = self.service.execute_action(MaintenanceActionType.REINDEX_DOCUMENT, "bad.pdf")

        self.assertFalse(res.success)
        self.assertIn("post-verification", res.message)

    def test_valid_orphan_removal_executes_and_verifies(self):
        target = "ghost_doc.pdf"
        # Pre-condition: physical file does NOT exist
        self.assertFalse((self.data_dir / target).exists())
        # Vectors exist before deletion
        self.mock_store.get_document_chunks.side_effect = [["c1", "c2"], []]  # before, after
        self.mock_store.delete_document.return_value = 2

        with patch.object(Settings, "get_data_dir", return_value=self.data_dir):
            res = self.service.execute_action(MaintenanceActionType.REMOVE_ORPHAN_VECTOR, target)

        self.assertTrue(res.success)
        self.assertEqual(res.action, MaintenanceActionType.REMOVE_ORPHAN_VECTOR)
        self.assertIn("purged 2 orphaned vectors", res.message)

    def test_orphan_removal_safety_violation_blocked_when_file_exists(self):
        """Safety protection: cannot delete active physical file's vectors via orphan cleanup."""
        target = "active_document.pdf"
        (self.data_dir / target).write_bytes(b"%PDF-1.4 Important physical document")

        with patch.object(Settings, "get_data_dir", return_value=self.data_dir):
            res = self.service.execute_action(MaintenanceActionType.REMOVE_ORPHAN_VECTOR, target)

        self.assertFalse(res.success)
        self.assertIn("Safety violation", res.message)
        self.mock_store.delete_document.assert_not_called()

    def test_unsupported_action_rejected_by_allowlist(self):
        with patch.object(Settings, "get_data_dir", return_value=self.data_dir):
            res = self.service.execute_action("DELETE_DATABASE", "manual.pdf")

        self.assertFalse(res.success)
        self.assertEqual(res.action, MaintenanceActionType.UNSUPPORTED)
        self.assertIn("not supported", res.message)

    def test_path_traversal_in_target_rejected(self):
        with patch.object(Settings, "get_data_dir", return_value=self.data_dir):
            for malicious in ["../../etc/passwd.pdf", "/root/bad.pdf", "subdir/file.pdf", r"foo\..\..\escape.pdf"]:
                res = self.service.execute_action(MaintenanceActionType.REINDEX_DOCUMENT, malicious)
                self.assertFalse(res.success)
                self.assertIn("Path traversal", res.message)


class TestMaintenanceAgentNodeAndObservability(unittest.TestCase):
    """Verifies MaintenanceAgent handling, intent parsing, resilient synthesis, and Langfuse tracing."""

    def test_parse_action_intent(self):
        agent = MaintenanceAgent(maintenance_service=MagicMock())

        action, target = agent._parse_action_intent("Please reindex manual.pdf")
        self.assertEqual(action, MaintenanceActionType.REINDEX_DOCUMENT)
        self.assertEqual(target, "manual.pdf")

        action, target = agent._parse_action_intent("Please re-index guide.pdf")
        self.assertEqual(action, MaintenanceActionType.REINDEX_DOCUMENT)
        self.assertEqual(target, "guide.pdf")

        action, target = agent._parse_action_intent("Clean up orphaned vectors for deleted.pdf")
        self.assertEqual(action, MaintenanceActionType.REMOVE_ORPHAN_VECTOR)
        self.assertEqual(target, "deleted.pdf")

        action, target = agent._parse_action_intent("Check whether all documents are healthy")
        self.assertEqual(action, MaintenanceActionType.NONE)
        self.assertIsNone(target)

    def test_maintenance_node_deterministic_fallback_on_llm_failure(self):
        """When LLM provider fails, maintenance_node still returns structured report without crashing."""
        mock_service = MagicMock()
        mock_service.inspect_system_and_consistency.return_value = MaintenanceReport(
            status=MaintenanceStatus.HEALTHY,
            checks_performed=["storage_availability", "vector_store_connectivity"],
            health={
                "storage": "healthy",
                "vector_store": "healthy",
                "physical_document_count": 2,
                "vector_chunk_count": 10,
                "indexed_document_count": 2,
            },
            issues=[],
        )

        mock_llm = MagicMock()
        mock_llm.chat.side_effect = RuntimeError("Groq model not found 404")

        agent = MaintenanceAgent(maintenance_service=mock_service, llm_service=mock_llm)

        with patch("app.agents.maintenance_agent._maintenance_agent", agent):
            state = {"question": "Run system health check"}
            output_state = maintenance_node(state)

        self.assertIn("ForgeMind System Maintenance Report", output_state["answer"])
        self.assertIn("HEALTHY", output_state["answer"])
        self.assertIn("Status: HEALTHY", output_state["context"])

    @patch("app.core.observability.ObservabilityService.get_client")
    def test_observability_event_emitted_without_secrets(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_service = MagicMock()
        mock_service.inspect_system_and_consistency.return_value = MaintenanceReport(
            status=MaintenanceStatus.HEALTHY,
            checks_performed=["storage_availability"],
            health={"storage": "healthy", "vector_store": "healthy"},
            issues=[],
        )

        agent = MaintenanceAgent(maintenance_service=mock_service)
        agent.handle("Check system status")

        mock_client.create_event.assert_called_once()
        call_kwargs = mock_client.create_event.call_args[1]
        self.assertEqual(call_kwargs["name"], "maintenance_operation")
        self.assertEqual(call_kwargs["output"]["status"], "HEALTHY")
        # Ensure no secrets in output or metadata
        payload_str = str(call_kwargs)
        self.assertNotIn("KEY", payload_str)
        self.assertNotIn("SECRET", payload_str)


class TestMaintenanceAPIEndpoints(unittest.TestCase):
    """Verifies FastAPI maintenance endpoints, schemas, and correlation IDs."""

    def setUp(self):
        self.client = TestClient(app)

    @patch("app.api.v1.maintenance._service.inspect_system_and_consistency")
    def test_get_maintenance_report_endpoint(self, mock_inspect):
        mock_inspect.return_value = MaintenanceReport(
            status=MaintenanceStatus.HEALTHY,
            checks_performed=["storage_availability", "vector_store_connectivity"],
            health={
                "storage": "healthy",
                "vector_store": "healthy",
                "physical_document_count": 3,
                "vector_chunk_count": 15,
                "indexed_document_count": 3,
            },
            issues=[],
            diagnosis="System is healthy",
        )

        for route in ["/api/v1/maintenance/report", "/maintenance/report"]:
            res = self.client.get(route, headers={"X-Request-ID": "maint-req-001"})
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.headers.get("x-request-id"), "maint-req-001")
            data = res.json()
            self.assertEqual(data["status"], "HEALTHY")
            self.assertEqual(data["health"]["physical_document_count"], 3)

    @patch("app.api.v1.maintenance._service.execute_action")
    def test_post_maintenance_execute_endpoint(self, mock_execute):
        mock_execute.return_value = ActionExecutionResult(
            action=MaintenanceActionType.REINDEX_DOCUMENT,
            target="manual.pdf",
            success=True,
            message="Successfully re-indexed 'manual.pdf' (1 pages, 4 chunks verified).",
        )

        for route in ["/api/v1/maintenance/execute", "/maintenance/execute"]:
            res = self.client.post(
                route,
                json={"action": "REINDEX_DOCUMENT", "target": "manual.pdf"},
                headers={"X-Request-ID": "exec-req-123"},
            )
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.headers.get("x-request-id"), "exec-req-123")
            data = res.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["action"], "REINDEX_DOCUMENT")
            self.assertEqual(data["target"], "manual.pdf")


if __name__ == "__main__":
    unittest.main()
