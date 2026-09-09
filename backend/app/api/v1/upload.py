from pathlib import Path

from fastapi import APIRouter, UploadFile, File
from app.core.exceptions import DocumentError
from app.core.logger import logger
from app.rag.indexer import DocumentIndexer
from app.config.settings import settings
from app.schemas.document import UploadResponse

router = APIRouter(tags=["Documents"])

indexer = DocumentIndexer()

CHUNK_SIZE = 64 * 1024  # 64 KB streaming buffer


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload and index PDF document",
    description="Validates, stores, and chunks a PDF document into ChromaDB embeddings for RAG retrieval.",
    responses={
        200: {"description": "PDF uploaded and indexed successfully", "model": UploadResponse},
        400: {"description": "Invalid filename, path traversal, or non-PDF file"},
        413: {"description": "File size exceeds maximum upload limit"},
        422: {"description": "No text could be extracted from this PDF"},
        500: {"description": "Failed to process and index the PDF document"},
    },
)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.strip():
        raise DocumentError("Filename is required", status_code=400)

    # 1. Path traversal prevention
    raw_filename = file.filename.strip()
    if (
        "\x00" in raw_filename
        or ".." in raw_filename
        or "/" in raw_filename
        or "\\" in raw_filename
        or ":" in raw_filename
    ):
        logger.warning(f"Path traversal attempt blocked in upload: '{raw_filename}'")
        raise DocumentError("Invalid filename: Path traversal detected", status_code=400)

    safe_filename = Path(raw_filename).name
    if safe_filename != raw_filename or not safe_filename or safe_filename in (".", ".."):
        logger.warning(f"Invalid filename blocked in upload: '{raw_filename}'")
        raise DocumentError("Invalid filename: Path traversal detected", status_code=400)

    # 2. File extension restriction
    if not safe_filename.lower().endswith(".pdf"):
        logger.warning(f"Non-PDF upload rejected: '{raw_filename}'")
        raise DocumentError("Invalid file type: Only PDF documents (.pdf) are supported", status_code=400)

    # 3. Path containment verification
    upload_dir = settings.get_data_dir()
    file_path = (upload_dir / safe_filename).resolve()

    if (
        not file_path.is_relative_to(upload_dir.resolve())
        or file_path.parent != upload_dir.resolve()
        or file_path == upload_dir.resolve()
    ):
        logger.warning(f"Path escape attempt blocked in upload: '{raw_filename}' -> '{file_path}'")
        raise DocumentError("Invalid destination: Path escapes storage directory", status_code=400)

    logger.info(f"Receiving PDF upload: {safe_filename}")

    # 4. Stream upload with magic bytes verification and size limit enforcement
    max_size_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    total_size = 0
    first_chunk = True

    try:
        with open(file_path, "wb") as f:
            while chunk := await file.read(CHUNK_SIZE):
                if first_chunk:
                    if not chunk.startswith(b"%PDF"):
                        raise DocumentError(
                            "Invalid PDF file: Missing %PDF header signature",
                            status_code=400,
                        )
                    first_chunk = False

                total_size += len(chunk)
                if total_size > max_size_bytes:
                    raise DocumentError(
                        f"File size exceeds maximum allowed limit of {settings.MAX_UPLOAD_SIZE_MB}MB",
                        status_code=413,
                    )
                f.write(chunk)

        if first_chunk:
            raise DocumentError("Uploaded file is empty", status_code=400)

    except Exception:
        if file_path.exists():
            file_path.unlink()
        raise

    # 5. Index PDF and clean up on failure
    try:
        result = indexer.index_pdf(str(file_path))
    except ValueError as e:
        logger.warning(f"Failed to extract text from PDF {safe_filename}: {e}")
        if file_path.exists():
            file_path.unlink()
        try:
            indexer.store.delete_document(safe_filename, str(file_path))
        except Exception:
            pass
        raise DocumentError(
            message="No text could be extracted from this PDF. It may be a scanned image-only document with no text layer. Please use a text-based PDF.",
            status_code=422,
        )
    except DocumentError:
        if file_path.exists():
            file_path.unlink()
        try:
            indexer.store.delete_document(safe_filename, str(file_path))
        except Exception:
            pass
        raise
    except Exception as e:
        logger.error(f"Failed to index PDF {safe_filename}: {str(e)}", exc_info=True)
        if file_path.exists():
            file_path.unlink()
        try:
            indexer.store.delete_document(safe_filename, str(file_path))
        except Exception:
            pass
        raise DocumentError(
            message="Failed to process and index the PDF document.",
            status_code=500,
        )

    logger.info(f"Successfully indexed PDF: {safe_filename} with {result['chunks']} chunks")

    return {
        "filename": safe_filename,
        "pages": result["pages"],
        "chunks": result["chunks"],
        "message": "PDF uploaded and indexed successfully",
    }