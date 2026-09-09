from fastapi import APIRouter, HTTPException

from app.services.document_service import DocumentService
from app.schemas.document import DocumentInfo, DeleteDocumentResponse

router = APIRouter(prefix="/documents", tags=["Documents"])

service = DocumentService()


@router.get(
    "/",
    response_model=list[DocumentInfo],
    summary="List indexed documents",
    description="Returns a list of all indexed PDF documents in the knowledge base with sizes.",
    responses={
        200: {"description": "List of indexed documents", "model": list[DocumentInfo]},
    },
)
def get_documents():

    return service.list_documents()


@router.delete(
    "/{filename:path}",
    response_model=DeleteDocumentResponse,
    summary="Delete document",
    description="Removes vector embeddings from ChromaDB first, then deletes physical PDF file from disk.",
    responses={
        200: {"description": "Document deleted successfully", "model": DeleteDocumentResponse},
        400: {"description": "Invalid filename, non-PDF, or path traversal attempt"},
        404: {"description": "Document not found"},
        500: {"description": "Vector store or filesystem deletion failure"},
    },
)
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