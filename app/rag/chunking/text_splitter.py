from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentChunker:
    """
    Splits Langchain Documents into Smaller chunks
    """
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", ""]
        )

    def split(self, documents: list[Document]) -> list[Document]:
        # Filter out pages with no extractable text (e.g. scanned/image PDFs)
        non_empty = [doc for doc in documents if doc.page_content.strip()]
        return self.splitter.split_documents(non_empty)