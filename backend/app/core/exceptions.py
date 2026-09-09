class ForgeMindError(Exception):
    def __init__(self, message: str, error_type: str = "InternalError", status_code: int = 500):
        self.message = message
        self.error_type = error_type
        self.status_code = status_code

class LLMError(ForgeMindError):
    def __init__(self, message: str = "LLM service unavailable", status_code: int = 502):
        super().__init__(message, error_type="LLMError", status_code=status_code)

class VectorStoreError(ForgeMindError):
    def __init__(self, message: str = "Vector store unavailable", status_code: int = 503):
        super().__init__(message, error_type="VectorStoreError", status_code=status_code)

class EmbeddingError(ForgeMindError):
    def __init__(self, message: str = "Embedding failure"):
        super().__init__(message, error_type="EmbeddingError", status_code=500)

class SearchError(ForgeMindError):
    def __init__(self, message: str = "Search service failed"):
        super().__init__(message, error_type="SearchError", status_code=502)

class DocumentError(ForgeMindError):
    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message, error_type="DocumentError", status_code=status_code)

class ValidationError(ForgeMindError):
    def __init__(self, message: str = "Invalid request"):
        super().__init__(message, error_type="ValidationError", status_code=400)

class MaintenanceError(ForgeMindError):
    def __init__(self, message: str = "Maintenance operation failed", status_code: int = 500):
        super().__init__(message, error_type="MaintenanceError", status_code=status_code)

