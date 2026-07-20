from app.rag.loader.pdf_loader import PDFLoader
loader = PDFLoader()
documents = loader.load("_OceanofPDF.com_Deep_Learning_for_Coders_with_fastai_and_P_-_Jeremy_Howard.pdf")
print(f"Total Pages : {len(documents)}")
print(f"Content : {documents[136].page_content[:200]}\n")
print(
    documents[0].metadata
)