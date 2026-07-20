from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingService:
    """
    Handles embedding generation
    """

    def __init__(self,model_name:str = "BAAI/bge-small-en-v1.5"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name = model_name,
            model_kwargs = {
                "device":"cpu",
            },
            encode_kwargs = {
                "normalize_embeddings":True,
            }
        )

    def get_embeddings(self):
        """Return Embeddings Model"""
        return self.embeddings