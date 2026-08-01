from langchain_huggingface import HuggingFaceEmbeddings
from app.config.settings import settings

class EmbeddingService:
    """
    Handles embedding generation
    """

    def __init__(self,model_name:str = settings.EMBEDDING_MODEL_NAME):
        from app.core.exceptions import EmbeddingError
        from app.core.logger import logger
        try:
            logger.info(f"Initializing Embedding Model: {model_name}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name = model_name,
                model_kwargs = {
                    "device": settings.EMBEDDING_DEVICE,
                },
                encode_kwargs = {
                    "normalize_embeddings":True,
                }
            )
        except Exception as e:
            logger.error(f"Failed to initialize Embedding Model: {str(e)}", exc_info=True)
            raise EmbeddingError(f"Embedding initialization failed: {str(e)}")

    def get_embeddings(self):
        """Return Embeddings Model"""
        return self.embeddings