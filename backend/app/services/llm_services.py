import time
from typing import Any, Callable
from langchain_core.messages import BaseMessage
from app.llms.factory import LLMFactory
from app.core.exceptions import LLMError
from app.core.logger import logger


def classify_llm_exception(exc: Exception) -> tuple[bool, str, int]:
    """
    Classifies an LLM/provider exception into:
    (is_transient, sanitized_client_message, http_status_code)
    """
    status_code = getattr(exc, "status_code", None)
    if status_code is None and hasattr(exc, "response"):
        status_code = getattr(exc.response, "status_code", None)

    exc_str = str(exc).lower()
    exc_type = exc.__class__.__name__.lower()

    # 1. Rate Limit / Quota (Transient, Retryable)
    if (
        status_code == 429
        or "429" in exc_str
        or "ratelimit" in exc_type
        or "rate limit" in exc_str
        or "too many requests" in exc_str
    ):
        return True, "AI model service rate limit exceeded. Please try again later.", 502

    # 2. Timeout (Transient, Retryable)
    if (
        isinstance(exc, TimeoutError)
        or status_code in (408, 504)
        or "timeout" in exc_type
        or "timed out" in exc_str
        or "timeout" in exc_str
    ):
        return True, "AI model service request timed out.", 504

    # 3. Connection / Network Errors (Transient, Retryable)
    if (
        isinstance(exc, ConnectionError)
        or "connection" in exc_type
        or "connecterror" in exc_type
        or "connection error" in exc_str
        or "connection refused" in exc_str
        or "network" in exc_str
    ):
        return True, "Failed to connect to AI model service.", 503

    # 4. Authentication / Permission (Non-Transient, Fail Fast)
    if (
        status_code in (401, 403)
        or "401" in exc_str
        or "403" in exc_str
        or "unauthorized" in exc_str
        or "authentication" in exc_str
        or "permission" in exc_str
        or "forbidden" in exc_str
        or "api key" in exc_str
        or "api_key" in exc_str
        or "authentication" in exc_type
        or "permission" in exc_type
        or "unauthorized" in exc_type
        or "auth" in exc_type
    ):
        return False, "AI model service authentication failed.", 502

    # 5. Invalid Request / Validation / Model Not Found (Non-Transient, Fail Fast)
    if (
        status_code in (400, 404, 422)
        or "400" in exc_str
        or "404" in exc_str
        or "bad request" in exc_str
        or "not found" in exc_str
        or "notfound" in exc_type
        or "badrequest" in exc_type
        or "invalidrequest" in exc_type
    ):
        return False, "Invalid request to AI model service.", 502

    # 4. Timeout (Transient, Retryable)
    if (
        isinstance(exc, TimeoutError)
        or status_code in (408, 504)
        or "timeout" in exc_type
        or "timed out" in exc_str
        or "timeout" in exc_str
    ):
        return True, "AI model service request timed out.", 504

    # 5. Connection / Network Errors (Transient, Retryable)
    if (
        isinstance(exc, ConnectionError)
        or "connection" in exc_type
        or "connecterror" in exc_type
        or "connection error" in exc_str
        or "network" in exc_str
    ):
        return True, "Failed to connect to AI model service.", 503

    # 6. Provider Service Unavailable / Gateway Error (Transient, Retryable)
    if (
        status_code in (500, 502, 503)
        or "internalservererror" in exc_type
        or "serviceunavailable" in exc_type
    ):
        return True, "AI model service is temporarily unavailable.", 502

    # 7. Fallback: Generic provider error (Non-Transient, Fail Fast)
    return False, "Failed to communicate with AI model service.", 502


class LLMService:
    """
    Service responsible for interacting with the configured LLM.
    Implements bounded exponential backoff retries for transient provider failures,
    fail-fast behavior for configuration/authentication errors, and sanitized error reporting.
    """

    def __init__(
        self,
        max_retries: int = 2,
        initial_delay: float = 0.5,
        backoff_factor: float = 2.0,
    ):
        self.provider = LLMFactory.get_provider()
        self.llm = self.provider.get_llm()
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor

    def invoke_with_retry(self, fn: Callable[[], Any], operation_name: str = "LLM request") -> Any:
        """
        Executes an LLM callable with bounded retry and exponential backoff for transient errors.
        Fails fast on non-transient errors (e.g. auth, bad request).
        Sanitizes error messages to protect credentials and internal details.
        """
        provider_name = self.get_provider_name()
        last_exception = None
        last_client_msg = "Failed to communicate with AI model service."
        last_status_code = 502

        for attempt in range(self.max_retries + 1):
            try:
                return fn()
            except Exception as e:
                last_exception = e
                is_transient, client_msg, status_code = classify_llm_exception(e)
                last_client_msg = client_msg
                last_status_code = status_code

                if is_transient and attempt < self.max_retries:
                    delay = self.initial_delay * (self.backoff_factor ** attempt)
                    logger.warning(
                        f"Transient error during {operation_name} from {provider_name} "
                        f"(attempt {attempt + 1}/{self.max_retries + 1}): {e}. Retrying in {delay:.1f}s..."
                    )
                    if delay > 0:
                        time.sleep(delay)
                    continue

                if not is_transient:
                    logger.error(
                        f"Non-retryable failure during {operation_name} from {provider_name}: {e}",
                        exc_info=True,
                    )
                    raise LLMError(client_msg, status_code=status_code) from e

        logger.error(
            f"{operation_name} from {provider_name} failed after {self.max_retries + 1} attempts: {last_exception}",
            exc_info=True,
        )
        raise LLMError(last_client_msg, status_code=last_status_code) from last_exception

    def chat(self, messages: list[BaseMessage], callbacks: list | None = None) -> Any:
        config = {"callbacks": callbacks} if callbacks else None

        def _invoke():
            logger.info(f"Sending request to LLM Provider: {self.get_provider_name()}")
            response = self.llm.invoke(messages, config=config)
            logger.info(f"Received response from LLM Provider: {self.get_provider_name()}")
            return response

        return self.invoke_with_retry(_invoke, operation_name=f"LLM Chat ({self.get_provider_name()})")

    def get_llm(self):
        """
        Returns the underlying LangChain chat model.
        Used when structured outputs or advanced LangChain features are needed.
        """
        return self.llm

    def get_provider_name(self) -> str:
        return self.provider.get_name()
