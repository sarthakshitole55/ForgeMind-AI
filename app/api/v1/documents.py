from fastapi import APIRouter, HTTPException

from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["Documents"])

service = DocumentService()


@router.get("/")
def get_documents():

    return service.list_documents()


@router.delete("/{filename}")
def delete_document(filename: str):

    deleted = service.delete_document(filename)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    return {
        "message": f"{filename} deleted successfully."
    }