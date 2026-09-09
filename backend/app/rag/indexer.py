from pathlib import Path
from app.core.logger import logger
from app.rag.chunking.text_splitter import DocumentChunker
from app.rag.loader.pdf_loader import PDFLoader
from app.rag.vectorstore.chroma_store import ChromaVectorStore


class DocumentIndexer:

    def __init__(self, vector_store: ChromaVectorStore | None = None):
        self.loader = PDFLoader()
        self.chunker = DocumentChunker()
        self.store = vector_store or ChromaVectorStore()

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

        safe_filename = Path(pdf_path).name
        canonical_source = str(Path(pdf_path).resolve())
        for chunk in chunks:
            chunk.metadata["filename"] = safe_filename
            chunk.metadata["source"] = canonical_source
        logger.info(f"Generated {len(chunks)} chunks from {len(documents)} pages. Storing in ChromaDB...")
        BATCH_SIZE = 32

        inserted_batches = False
        try:
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i:i + BATCH_SIZE]
                logger.info(
                    f"Indexing batch {i // BATCH_SIZE + 1} "
                    f"of {(len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE}"
                )
                self.store.add_documents(batch)
                inserted_batches = True
        except Exception as e:
            logger.error(
                f"Vector insertion failed for '{safe_filename}': {e}",
                exc_info=True,
            )
            if inserted_batches:
                logger.info(f"Purging partial vectors for '{safe_filename}' following batch failure...")
                try:
                    self.store.delete_document(safe_filename, canonical_source)
                except Exception as del_err:
                    logger.error(f"Failed to clean up partial vectors for '{safe_filename}': {del_err}", exc_info=True)
            raise
        

        return {
            "pages": len(documents),
            "chunks": len(chunks),
        }