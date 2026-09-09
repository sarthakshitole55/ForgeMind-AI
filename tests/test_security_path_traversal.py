import io
import os
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("GROQ_API_KEY", "gsk_test_dummy_key_12345678901234567890")

from fastapi.testclient import TestClient
from app.main import app
from app.config.settings import settings


def _create_valid_pdf_bytes() -> bytes:
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class TestSecurityPathTraversal(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.data_dir = settings.get_data_dir()
        self.valid_pdf = _create_valid_pdf_bytes()
        self.test_files_to_cleanup = []

    def tearDown(self):
        for path in self.test_files_to_cleanup:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # 1. ../../ Path Traversal Tests (Upload)
    # -------------------------------------------------------------------------
    def test_upload_path_traversal_dot_dot(self):
        """Verify upload with ../../ in filename is rejected with 400 and not written outside DATA_DIR."""
        traversal_filename = "../../escape_test.pdf"
        escaped_file = (self.data_dir.parent / "escape_test.pdf").resolve()
        self.test_files_to_cleanup.append(escaped_file)

        response = self.client.post(
            "/upload",
            files={"file": (traversal_filename, io.BytesIO(self.valid_pdf), "application/pdf")},
        )

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("Path traversal", data.get("detail", "") or str(data.get("error", {})))
        self.assertFalse(escaped_file.exists(), "Security failure: File was written outside DATA_DIR!")

    def test_upload_path_traversal_deep_dot_dot(self):
        """Verify upload with deeply nested ../../../../ is rejected."""
        traversal_filename = "../../../../tmp/deep_escape.pdf"
        response = self.client.post(
            "/upload",
            files={"file": (traversal_filename, io.BytesIO(self.valid_pdf), "application/pdf")},
        )
        self.assertEqual(response.status_code, 400)

    # -------------------------------------------------------------------------
    # 2. Nested Path Traversal Tests (Upload)
    # -------------------------------------------------------------------------
    def test_upload_path_traversal_nested(self):
        """Verify upload with subdirectory or nested path traversal is rejected."""
        nested_filenames = [
            "subfolder/test.pdf",
            "foo/../../escape.pdf",
            "subdir/nested/doc.pdf",
            r"subfolder\test.pdf",
            r"foo\..\..\escape.pdf",
        ]
        for filename in nested_filenames:
            with self.subTest(filename=filename):
                response = self.client.post(
                    "/upload",
                    files={"file": (filename, io.BytesIO(self.valid_pdf), "application/pdf")},
                )
                self.assertEqual(response.status_code, 400)

    # -------------------------------------------------------------------------
    # 3. Absolute Path Tests (Upload)
    # -------------------------------------------------------------------------
    def test_upload_path_traversal_absolute(self):
        """Verify upload with absolute path in filename is rejected."""
        absolute_filenames = [
            "/etc/passwd.pdf",
            "/tmp/absolute_test.pdf",
            "//network/share/test.pdf",
            "C:malicious.pdf",
        ]
        for filename in absolute_filenames:
            with self.subTest(filename=filename):
                response = self.client.post(
                    "/upload",
                    files={"file": (filename, io.BytesIO(self.valid_pdf), "application/pdf")},
                )
                self.assertEqual(response.status_code, 400)

    # -------------------------------------------------------------------------
    # 4. Valid PDF Upload
    # -------------------------------------------------------------------------
    def test_upload_valid_pdf(self):
        """Verify valid PDF upload succeeds and is saved strictly inside DATA_DIR."""
        valid_filename = "security_test_valid.pdf"
        expected_path = self.data_dir / valid_filename
        self.test_files_to_cleanup.append(expected_path)

        with patch("app.rag.indexer.DocumentIndexer.index_pdf", return_value={"pages": 1, "chunks": 1}):
            response = self.client.post(
                "/upload",
                files={"file": (valid_filename, io.BytesIO(self.valid_pdf), "application/pdf")},
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["filename"], valid_filename)
        self.assertEqual(data["pages"], 1)
        self.assertEqual(data["chunks"], 1)
        self.assertTrue(expected_path.exists(), "Valid file should exist in DATA_DIR")
        self.assertTrue(expected_path.is_relative_to(self.data_dir.resolve()))

    # -------------------------------------------------------------------------
    # 5. Non-PDF Upload Tests
    # -------------------------------------------------------------------------
    def test_upload_non_pdf_extension(self):
        """Verify non-PDF extensions are rejected with 400."""
        invalid_extensions = [
            "script.sh",
            "malware.exe",
            "data.txt",
            "index.html",
            "payload.py",
            "no_extension",
        ]
        for filename in invalid_extensions:
            with self.subTest(filename=filename):
                response = self.client.post(
                    "/upload",
                    files={"file": (filename, io.BytesIO(b"echo 'evil'"), "text/plain")},
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn("Only PDF documents", response.json().get("detail", ""))

    def test_upload_invalid_pdf_content_signature(self):
        """Verify file named .pdf but containing non-PDF bytes is rejected with 400."""
        fake_pdf_filename = "fake_document.pdf"
        fake_content = b"This is plain text and does not have the PDF magic bytes signature."

        response = self.client.post(
            "/upload",
            files={"file": (fake_pdf_filename, io.BytesIO(fake_content), "application/pdf")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Missing %PDF header", response.json().get("detail", ""))
        self.assertFalse((self.data_dir / fake_pdf_filename).exists())

    def test_upload_empty_file(self):
        """Verify 0-byte file is rejected with 400."""
        response = self.client.post(
            "/upload",
            files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
        )
        self.assertEqual(response.status_code, 400)

    # -------------------------------------------------------------------------
    # 6. Oversized Upload Test
    # -------------------------------------------------------------------------
    def test_upload_oversized_file(self):
        """Verify upload exceeding MAX_UPLOAD_SIZE_MB is rejected with 413 and cleaned up."""
        oversized_filename = "oversized_test.pdf"
        target_path = self.data_dir / oversized_filename
        self.test_files_to_cleanup.append(target_path)

        # Temporarily mock MAX_UPLOAD_SIZE_MB to 1 MB for fast testing
        with patch.object(settings, "MAX_UPLOAD_SIZE_MB", 1):
            # Create a 1.2 MB stream starting with %PDF-
            oversized_stream = io.BytesIO(b"%PDF-1.4\n" + b"X" * (1024 * 1024 + 200 * 1024))
            response = self.client.post(
                "/upload",
                files={"file": (oversized_filename, oversized_stream, "application/pdf")},
            )

            self.assertEqual(response.status_code, 413)
            self.assertIn("exceeds maximum allowed limit", response.json().get("detail", ""))
            self.assertFalse(target_path.exists(), "Oversized file should have been unlinked/cleaned up!")

    # -------------------------------------------------------------------------
    # 7. Deletion Traversal Tests
    # -------------------------------------------------------------------------
    def test_deletion_path_traversal_dot_dot(self):
        """Verify deletion with ../../ is blocked with 400."""
        from app.services.document_service import DocumentService
        from fastapi import HTTPException

        # 1. Direct service call with path traversal must raise 400
        service = DocumentService()
        with self.assertRaises(HTTPException) as ctx:
            service.delete_document("../../do_not_delete.pdf")
        self.assertEqual(ctx.exception.status_code, 400)

        # 2. HTTP call with encoded traversal must return 400
        outside_file = self.data_dir.parent / "do_not_delete.pdf"
        outside_file.write_text("critical data")
        self.test_files_to_cleanup.append(outside_file)

        response = self.client.delete("/documents/%2e%2e%2f%2e%2e%2fdo_not_delete.pdf")
        self.assertEqual(response.status_code, 400)
        self.assertTrue(outside_file.exists(), "Security failure: File outside DATA_DIR was deleted!")

    def test_deletion_path_traversal_nested(self):
        """Verify deletion with nested path traversal is blocked with 400."""
        from app.services.document_service import DocumentService
        from fastapi import HTTPException

        service = DocumentService()
        nested_attempts = [
            "subfolder/test.pdf",
            "foo/../../escape.pdf",
            "../escape.pdf",
            r"foo\..\..\escape.pdf",
        ]
        for filename in nested_attempts:
            with self.subTest(filename=filename):
                # Service check
                with self.assertRaises(HTTPException) as ctx:
                    service.delete_document(filename)
                self.assertEqual(ctx.exception.status_code, 400)

                # HTTP endpoint check (URL encoded)
                encoded = filename.replace("/", "%2F").replace("\\", "%5C")
                response = self.client.delete(f"/documents/{encoded}")
                self.assertEqual(response.status_code, 400)

    def test_deletion_non_pdf_rejected(self):
        """Verify deletion of non-PDF files is rejected with 400."""
        response = self.client.delete("/documents/main.py")
        self.assertEqual(response.status_code, 400)
        self.assertIn("Only PDF files", response.json().get("detail", ""))

    def test_deletion_valid_file(self):
        """Verify valid PDF deletion deletes the file and returns 200."""
        test_pdf = self.data_dir / "temp_to_delete.pdf"
        test_pdf.write_bytes(self.valid_pdf)

        response = self.client.delete("/documents/temp_to_delete.pdf")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(test_pdf.exists())

    def test_deletion_missing_file_returns_404(self):
        """Verify deleting non-existent PDF returns 404."""
        response = self.client.delete("/documents/non_existent_file_12345.pdf")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
