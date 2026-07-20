from pathlib import Path

DATA_DIR = Path("data")


class DocumentService:

    def list_documents(self):
        return [
            {
                "name": pdf.name,
                "size_kb": round(pdf.stat().st_size / 1024, 2),
            }
            for pdf in DATA_DIR.glob("*.pdf")
        ]

    def delete_document(self, filename: str):

        file_path = DATA_DIR / filename

        if not file_path.exists():
            return False

        file_path.unlink()

        return True