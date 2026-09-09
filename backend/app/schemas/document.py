from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    name: str = Field(..., description="Filename of the indexed document")
    size_kb: float = Field(..., description="File size in kilobytes")


class DeleteDocumentResponse(BaseModel):
    message: str = Field(..., description="Deletion confirmation message")


class UploadResponse(BaseModel):
    filename: str = Field(..., description="Filename of the uploaded and indexed PDF")
    pages: int = Field(..., description="Number of parsed pages from the document")
    chunks: int = Field(..., description="Number of vector chunks generated and stored")
    message: str = Field(..., description="Status summary message")
