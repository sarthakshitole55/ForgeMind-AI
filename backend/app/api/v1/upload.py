from pathlib import Path

from fastapi import APIRouter, UploadFile, File
from app.core.exceptions import DocumentError
from app.core.logger import logger
from app.rag.indexer import DocumentIndexer
from app.config.settings import settings

router = APIRouter()

UPLOAD_DIR = Path(settings.DATA_DIR)
UPLOAD_DIR.mkdir(exist_ok=True)

indexer = DocumentIndexer()

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    logger.info(f"Receiving PDF upload: {file.filename}")
    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        result = indexer.index_pdf(str(file_path))
    except ValueError as e:
        logger.warning(f"Failed to extract text from PDF: {file.filename}")
        raise DocumentError(
            message="No text could be extracted from this PDF. It may be a scanned image-only document with no text layer. Please use a text-based PDF."
        )

    logger.info(f"Successfully indexed PDF: {file.filename} with {result['chunks']} chunks")

    return {
        "filename": file.filename,
        "pages": result["pages"],
        "chunks": result["chunks"],
        "message": "PDF uploaded and indexed successfully",
    }