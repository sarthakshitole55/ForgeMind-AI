from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.rag.indexer import DocumentIndexer

router = APIRouter()

UPLOAD_DIR = Path("data")
UPLOAD_DIR.mkdir(exist_ok=True)

indexer = DocumentIndexer()


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        result = indexer.index_pdf(str(file_path))
    except ValueError as e:
        # PDF has no extractable text (e.g. scanned image-only PDF)
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "filename": file.filename,
        "pages": result["pages"],
        "chunks": result["chunks"],
        "message": "PDF uploaded and indexed successfully",
    }