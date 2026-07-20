from pathlib import Path

from app.rag.chunking.text_splitter import DocumentChunker
from app.rag.loader.pdf_loader import PDFLoader
from app.rag.vectorstore.chroma_store import ChromaVectorStore

class DocumentIndexer:

    def __init__(self):
        self.loader = PDFLoader()
        self.chunker = DocumentChunker()
        self.store = ChromaVectorStore()

    def index_pdf(self, pdf_path: str):

        documents = self.loader.load(pdf_path)

        chunks = self.chunker.split(documents)

        if not chunks:
            raise ValueError(
                "No text could be extracted from this PDF. "
                "It may be a scanned image-only document with no text layer. "
                "Please use a text-based PDF."
            )

        self.store.add_documents(chunks)

        return {
            "pages": len(documents),
            "chunks": len(chunks),
        }