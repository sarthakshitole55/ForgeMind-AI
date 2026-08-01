from pathlib import Path
from app.core.logger import logger
from app.rag.chunking.text_splitter import DocumentChunker
from app.rag.loader.pdf_loader import PDFLoader
from app.rag.vectorstore.chroma_store import ChromaVectorStore


class DocumentIndexer:

    def __init__(self):
        self.loader = PDFLoader()
        self.chunker = DocumentChunker()
        self.store = ChromaVectorStore()
        

    def index_pdf(self, pdf_path: str):
        logger.info(f"Starting indexing process for PDF: {pdf_path}")
        documents = self.loader.load(pdf_path)

        chunks = self.chunker.split(documents)

        if not chunks:
            raise ValueError(
                "No text could be extracted from this PDF. "
                "It may be a scanned image-only document with no text layer. "
                "Please use a text-based PDF."
            )

        logger.info(f"Generated {len(chunks)} chunks from {len(documents)} pages. Storing in ChromaDB...")
        BATCH_SIZE = 32

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            logger.info(
                f"Indexing batch {i // BATCH_SIZE + 1} "
                f"of {(len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE}"
            )
            self.store.add_documents(batch)
        

        return {
            "pages": len(documents),
            "chunks": len(chunks),
        }