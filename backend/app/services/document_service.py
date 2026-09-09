from pathlib import Path
from fastapi import HTTPException
from app.config.settings import settings
from app.core.logger import logger
from app.rag.vectorstore.chroma_store import ChromaVectorStore


class DocumentService:

    def __init__(self, vector_store: ChromaVectorStore | None = None):
        self.vector_store = vector_store or ChromaVectorStore()

    def list_documents(self):
        data_dir = settings.get_data_dir()
        return [
            {
                "name": pdf.name,
                "size_kb": round(pdf.stat().st_size / 1024, 2),
            }
            for pdf in data_dir.glob("*.pdf")
        ]

    def delete_document(self, filename: str) -> bool:
        data_dir = settings.get_data_dir()

        # Reject path traversal patterns, null bytes, drive letters, and path separators
        if (
            not filename
            or "\x00" in filename
            or ".." in filename
            or "/" in filename
            or "\\" in filename
            or ":" in filename
        ):
            logger.warning(f"Path traversal attempt blocked in document deletion: '{filename}'")
            raise HTTPException(
                status_code=400,
                detail="Invalid filename: Path traversal detected",
            )

        safe_filename = Path(filename).name
        if safe_filename != filename or not safe_filename or safe_filename in (".", ".."):
            logger.warning(f"Invalid filename blocked in document deletion: '{filename}'")
            raise HTTPException(
                status_code=400,
                detail="Invalid filename: Path traversal detected",
            )

        if not safe_filename.lower().endswith(".pdf"):
            logger.warning(f"Non-PDF deletion attempt blocked: '{filename}'")
            raise HTTPException(
                status_code=400,
                detail="Invalid filename: Only PDF files can be deleted",
            )

        file_path = (data_dir / safe_filename).resolve()

        if (
            not file_path.is_relative_to(data_dir.resolve())
            or file_path.parent != data_dir.resolve()
            or file_path == data_dir.resolve()
        ):
            logger.warning(f"Path escape attempt blocked in document deletion: '{filename}' -> '{file_path}'")
            raise HTTPException(
                status_code=400,
                detail="Invalid filename: Path traversal detected",
            )

        if not file_path.exists() or not file_path.is_file():
            return False

        # 1. Delete corresponding vectors/chunks from ChromaDB FIRST
        try:
            deleted_vectors = self.vector_store.delete_document(
                filename=safe_filename,
                file_path=str(file_path),
            )
            logger.info(
                f"Successfully deleted {deleted_vectors} ChromaDB vectors for {safe_filename}"
            )
        except Exception as e:
            logger.error(
                f"Failed to delete ChromaDB vectors for {safe_filename}: {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to delete document vectors from vector store",
            )

        # 2. Delete the physical PDF file SECOND
        try:
            file_path.unlink()
            logger.info(f"Successfully deleted document file: {safe_filename}")
        except Exception as e:
            logger.error(
                f"Failed to delete physical file for {safe_filename}: {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to delete physical document file",
            )

        return True