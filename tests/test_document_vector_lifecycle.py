import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("GROQ_API_KEY", "gsk_test_dummy_key_12345678901234567890")

import fitz
from fastapi.testclient import TestClient
from langchain_chroma import Chroma
from langchain_core.embeddings import FakeEmbeddings

from app.config.settings import Settings
from app.main import app
from app.rag.indexer import DocumentIndexer
from app.rag.vectorstore.chroma_store import ChromaVectorStore
from app.services.document_service import DocumentService


def _create_pdf_with_text(path: Path, text: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    doc.save(str(path))
    doc.close()


class TestDocumentVectorLifecycle(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir = os.path.join(self.temp_dir, "chroma")

        self.fake_embeddings = FakeEmbeddings(size=384)
        self.lc_chroma = Chroma(
            collection_name="test_forge_manuals",
            persist_directory=self.chroma_dir,
            embedding_function=self.fake_embeddings,
        )
        self.store = ChromaVectorStore(db=self.lc_chroma)
        self.indexer = DocumentIndexer(vector_store=self.store)
        self.service = DocumentService(vector_store=self.store)
        self.client = TestClient(app)

    def tearDown(self):
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass

    def test_document_indexed_and_retrievable_before_deletion(self):
        """1 & 2: A document is indexed and its vectors can be retrieved before deletion."""
        doc_path = self.data_dir / "quantum_physics.pdf"
        _create_pdf_with_text(doc_path, "Quantum flux capacitor resonance frequency alpha")

        result = self.indexer.index_pdf(str(doc_path))
        self.assertGreater(result["chunks"], 0)

        # Retrieve via chunk IDs
        chunks = self.store.get_document_chunks("quantum_physics.pdf", str(doc_path))
        self.assertEqual(len(chunks), result["chunks"])

        # Retrieve via similarity search
        search_results = self.store.similarity_search("quantum flux capacitor", k=5)
        self.assertTrue(len(search_results) > 0)
        self.assertTrue(any("Quantum" in doc.page_content for doc in search_results))

    def test_document_deletion_removes_vectors(self):
        """3 & 4: The document is deleted and its vectors are no longer retrievable afterward."""
        doc_path = self.data_dir / "quantum_physics.pdf"
        _create_pdf_with_text(doc_path, "Quantum flux capacitor resonance frequency alpha")

        self.indexer.index_pdf(str(doc_path))
        self.assertGreater(len(self.store.get_document_chunks("quantum_physics.pdf", str(doc_path))), 0)

        with patch.object(Settings, "get_data_dir", return_value=self.data_dir):
            deleted = self.service.delete_document("quantum_physics.pdf")

        self.assertTrue(deleted)
        self.assertFalse(doc_path.exists(), "File should be removed from disk")

        # Vectors must no longer be retrievable
        chunks_after = self.store.get_document_chunks("quantum_physics.pdf", str(doc_path))
        self.assertEqual(len(chunks_after), 0, "No vectors should remain in ChromaDB for deleted document")

        # Similarity search should not retrieve the deleted content
        search_after = self.store.similarity_search("quantum flux capacitor", k=5)
        self.assertFalse(
            any("Quantum" in doc.page_content for doc in search_after),
            "Deleted document content must not be retrievable via similarity search",
        )

    def test_other_document_vectors_not_accidentally_deleted(self):
        """5: Another document's vectors are not accidentally deleted."""
        doc_a_path = self.data_dir / "quantum_physics.pdf"
        doc_b_path = self.data_dir / "plant_biology.pdf"

        _create_pdf_with_text(doc_a_path, "Quantum flux capacitor resonance frequency alpha")
        _create_pdf_with_text(doc_b_path, "Photosynthesis chlorophyll magnesium light reaction beta")

        self.indexer.index_pdf(str(doc_a_path))
        self.indexer.index_pdf(str(doc_b_path))

        chunks_b_before = self.store.get_document_chunks("plant_biology.pdf", str(doc_b_path))
        self.assertGreater(len(chunks_b_before), 0)

        # Delete document A only
        with patch.object(Settings, "get_data_dir", return_value=self.data_dir):
            deleted_a = self.service.delete_document("quantum_physics.pdf")

        self.assertTrue(deleted_a)
        self.assertFalse(doc_a_path.exists())
        self.assertTrue(doc_b_path.exists(), "Document B file must remain intact on disk")

        # Document A vectors are gone
        chunks_a_after = self.store.get_document_chunks("quantum_physics.pdf", str(doc_a_path))
        self.assertEqual(len(chunks_a_after), 0)

        # Document B vectors must remain completely intact
        chunks_b_after = self.store.get_document_chunks("plant_biology.pdf", str(doc_b_path))
        self.assertEqual(len(chunks_b_after), len(chunks_b_before))

        search_b = self.store.similarity_search("photosynthesis chlorophyll", k=5)
        self.assertTrue(
            any("Photosynthesis" in doc.page_content for doc in search_b),
            "Document B vectors must remain retrievable after Document A is deleted",
        )

    def test_delete_nonexistent_document_behaves_as_before(self):
        """6: Deleting a nonexistent document returns False / preserves behavior without error."""
        # Index document A first so we verify it's unaffected
        doc_a_path = self.data_dir / "existing_doc.pdf"
        _create_pdf_with_text(doc_a_path, "Important knowledge base article content")
        self.indexer.index_pdf(str(doc_a_path))
        chunks_before = self.store.get_document_chunks("existing_doc.pdf", str(doc_a_path))

        with patch.object(Settings, "get_data_dir", return_value=self.data_dir):
            deleted = self.service.delete_document("nonexistent_file_9999.pdf")

        self.assertFalse(deleted, "Deleting nonexistent document must return False")

        # Verify existing document was not affected
        chunks_after = self.store.get_document_chunks("existing_doc.pdf", str(doc_a_path))
        self.assertEqual(len(chunks_after), len(chunks_before))

    def test_http_api_delete_document_removes_vectors_and_returns_200(self):
        """API endpoint /documents/{filename} removes vectors and returns 200 on success, 404 on missing."""
        doc_path = self.data_dir / "api_doc.pdf"
        _create_pdf_with_text(doc_path, "Secure corporate guidelines and protocols")
        self.indexer.index_pdf(str(doc_path))

        self.assertGreater(len(self.store.get_document_chunks("api_doc.pdf", str(doc_path))), 0)

        with patch.object(Settings, "get_data_dir", return_value=self.data_dir):
            with patch("app.api.v1.documents.service", self.service):
                # 1. Successful deletion via API
                response = self.client.delete("/documents/api_doc.pdf")
                self.assertEqual(response.status_code, 200)
                self.assertIn("deleted successfully", response.json().get("message", ""))
                self.assertFalse(doc_path.exists())
                self.assertEqual(len(self.store.get_document_chunks("api_doc.pdf", str(doc_path))), 0)

                # 2. Deleting nonexistent via API returns 404
                response_404 = self.client.delete("/documents/api_doc.pdf")
                self.assertEqual(response_404.status_code, 404)

    def test_legacy_source_metadata_deleted_reliably(self):
        """Vectors with legacy source metadata (no filename attribute) are cleanly deleted."""
        doc_path = self.data_dir / "legacy_doc.pdf"
        _create_pdf_with_text(doc_path, "Legacy indexed document text without filename metadata")

        # Manually add chunk with legacy metadata (only 'source', no 'filename')
        from langchain_core.documents import Document
        legacy_doc = Document(
            page_content="Legacy indexed document text without filename metadata",
            metadata={"source": f"data/legacy_doc.pdf", "page": 0},
        )
        self.store.add_documents([legacy_doc])

        # Verify legacy chunk is present
        legacy_chunks = self.store.get_document_chunks("legacy_doc.pdf", str(doc_path))
        self.assertEqual(len(legacy_chunks), 1)

        with patch.object(Settings, "get_data_dir", return_value=self.data_dir):
            deleted = self.service.delete_document("legacy_doc.pdf")

        self.assertTrue(deleted)
        self.assertFalse(doc_path.exists())

        # Verify legacy chunk was removed
        remaining_chunks = self.store.get_document_chunks("legacy_doc.pdf", str(doc_path))
        self.assertEqual(len(remaining_chunks), 0)

    # -------------------------------------------------------------------------
    # Failure Consistency & Retry Tests
    # -------------------------------------------------------------------------
    def test_vector_deletion_failure_keeps_file_and_returns_500(self):
        """When vector deletion raises an exception: file remains, HTTP 500 returned, raw error masked."""
        doc_path = self.data_dir / "vector_fail.pdf"
        _create_pdf_with_text(doc_path, "Secret internal document content")
        self.indexer.index_pdf(str(doc_path))

        with patch.object(Settings, "get_data_dir", return_value=self.data_dir):
            with patch.object(
                self.store,
                "delete_document",
                side_effect=RuntimeError("Internal ChromaDB sqlite3 lock failure: /var/db/chroma.sqlite3"),
            ):
                with patch("app.api.v1.documents.service", self.service):
                    response = self.client.delete("/documents/vector_fail.pdf")

                    # 1. API returns HTTP 500
                    self.assertEqual(response.status_code, 500)
                    detail = response.json().get("detail", "")
                    self.assertEqual(detail, "Failed to delete document vectors from vector store")

                    # 2. Raw exception text and paths are NOT exposed to the client
                    self.assertNotIn("sqlite3 lock failure", detail)
                    self.assertNotIn("/var/db/chroma.sqlite3", detail)
                    self.assertNotIn("RuntimeError", detail)

                    # 3. Physical file must remain intact
                    self.assertTrue(doc_path.exists(), "Physical file must NOT be deleted when vector deletion fails")

    def test_file_deletion_failure_after_vectors_deleted_allows_safe_retry(self):
        """When file unlink fails after vectors deleted: HTTP 500 returned, masked error, retry succeeds."""
        doc_path = self.data_dir / "unlink_fail.pdf"
        _create_pdf_with_text(doc_path, "Important document with transient OS file lock")
        self.indexer.index_pdf(str(doc_path))

        self.assertGreater(len(self.store.get_document_chunks("unlink_fail.pdf", str(doc_path))), 0)

        with patch.object(Settings, "get_data_dir", return_value=self.data_dir):
            with patch("app.api.v1.documents.service", self.service):
                # 1. Simulate file unlinking failure
                with patch.object(Path, "unlink", side_effect=PermissionError("OS error 13: Permission denied /data/unlink_fail.pdf")):
                    response = self.client.delete("/documents/unlink_fail.pdf")

                    self.assertEqual(response.status_code, 500)
                    detail = response.json().get("detail", "")
                    self.assertEqual(detail, "Failed to delete physical document file")
                    # Raw exception text and paths are NOT exposed
                    self.assertNotIn("Permission denied", detail)
                    self.assertNotIn("/data/unlink_fail.pdf", detail)
                    self.assertNotIn("PermissionError", detail)

                    # Physical file remains on disk
                    self.assertTrue(doc_path.exists(), "File should still exist on disk when unlink fails")

                    # Vectors were already deleted
                    chunks_after = self.store.get_document_chunks("unlink_fail.pdf", str(doc_path))
                    self.assertEqual(len(chunks_after), 0, "Vectors should be deleted prior to unlink")

                # 2. Subsequent retry can safely remove the file and succeed
                retry_response = self.client.delete("/documents/unlink_fail.pdf")
                self.assertEqual(retry_response.status_code, 200)
                self.assertIn("deleted successfully", retry_response.json().get("message", ""))
                self.assertFalse(doc_path.exists(), "File should be deleted after successful retry")

    def test_retry_behavior_when_vectors_already_deleted(self):
        """Simulate vectors already deleted; physical file still exists; DELETE safely removes file."""
        doc_path = self.data_dir / "already_deleted_vectors.pdf"
        _create_pdf_with_text(doc_path, "Content whose vectors were already purged")

        # Vectors are empty
        self.assertEqual(len(self.store.get_document_chunks("already_deleted_vectors.pdf", str(doc_path))), 0)
        self.assertTrue(doc_path.exists())

        with patch.object(Settings, "get_data_dir", return_value=self.data_dir):
            with patch("app.api.v1.documents.service", self.service):
                response = self.client.delete("/documents/already_deleted_vectors.pdf")
                self.assertEqual(response.status_code, 200)
                self.assertIn("deleted successfully", response.json().get("message", ""))
                self.assertFalse(doc_path.exists(), "Physical file must be unlinked successfully")


if __name__ == "__main__":
    unittest.main()

