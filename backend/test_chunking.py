from app.rag.chunking.text_splitter import DocumentChunker
from app.rag.loader.pdf_loader import PDFLoader

loader = PDFLoader()
documents = loader.load("_OceanofPDF.com_Deep_Learning_for_Coders_with_fastai_and_P_-_Jeremy_Howard.pdf")
chunker = DocumentChunker()
chunks = chunker.split(documents)
print(f"Pages : {len(documents)}")
print(f"Chunk : {len(chunks)}")

print()
print(chunks[155].page_content)
print()
print(chunks[155].metadata)