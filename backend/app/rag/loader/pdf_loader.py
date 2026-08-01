from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

class PDFLoader:
    """
    Loads PDF Files and converts them into langchain Document objects
    """
    def load(self,file_path:str)->list[Document]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(
                f"PDF Not Found : {file_path}"

            )
        loader = PyPDFLoader(
            str(path)
        )
        documents = loader.load()
        return documents
            
