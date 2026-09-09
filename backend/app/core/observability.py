import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
from langchain_core.callbacks.base import BaseCallbackHandler

from app.config.settings import settings
from app.core.logger import logger


class ObservabilityService:
    """
    Manages Langfuse observability for ForgeMind AI.
    Ensures safe error handling, credential privacy, and zero crash guarantees.
    """

    _client = None
    _client_initialized = False

    @classmethod
    def is_enabled(cls) -> bool:
        """Returns True if Langfuse is configured and enabled."""
        try:
            return (
                getattr(settings, "LANGFUSE_ENABLED", True)
                and bool(getattr(settings, "LANGFUSE_PUBLIC_KEY", None))
                and bool(getattr(settings, "LANGFUSE_SECRET_KEY", None))
            )
        except Exception:
            return False

    @classmethod
    def get_client(cls):
        """Returns the Langfuse client singleton if enabled, or None."""
        if not cls.is_enabled():
            return None

        if not cls._client_initialized:
            try:
                from langfuse import Langfuse
                cls._client = Langfuse(
                    public_key=settings.LANGFUSE_PUBLIC_KEY,
                    secret_key=settings.LANGFUSE_SECRET_KEY,
                    host=settings.LANGFUSE_HOST,
                )
                logger.info("Langfuse observability client initialized.")
            except Exception as e:
                logger.warning(f"Failed to initialize Langfuse client: {e}. Observability disabled.")
                cls._client = None
            cls._client_initialized = True

        return cls._client

    @classmethod
    def get_callback_handler(cls, trace_context: Optional[Dict[str, Any]] = None) -> Optional[BaseCallbackHandler]:
        """
        Returns a Langfuse CallbackHandler for LangChain / LangGraph execution.
        Returns None if disabled or on error.
        """
        if not cls.is_enabled():
            return None

        try:
            from langfuse.langchain import CallbackHandler
            kwargs: Dict[str, Any] = {}
            if settings.LANGFUSE_PUBLIC_KEY:
                kwargs["public_key"] = settings.LANGFUSE_PUBLIC_KEY
            if trace_context:
                kwargs["trace_context"] = trace_context
            return CallbackHandler(**kwargs)
        except Exception as e:
            logger.warning(f"Failed to create Langfuse CallbackHandler: {e}")
            return None

    @classmethod
    @contextmanager
    def start_request_trace(cls, name: str, user_query: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for end-to-end request tracing.
        Yields (trace_id, callback_handler).
        Guaranteed to never raise an exception even if Langfuse fails.
        """
        client = cls.get_client()
        span = None
        handler = None
        trace_id = None

        if client:
            try:
                span = client.start_observation(
                    name=name,
                    as_type="chain",
                    input={"query": user_query},
                    metadata=metadata or {},
                )
                trace_id = getattr(span, "trace_id", None)
                handler = cls.get_callback_handler(trace_context={"trace_id": trace_id} if trace_id else None)
            except Exception as e:
                logger.warning(f"Error starting Langfuse request trace: {e}")
                span = None
                handler = None
                trace_id = None

        try:
            yield trace_id, handler
        finally:
            if span:
                try:
                    span.end()
                    cls.flush()
                except Exception as e:
                    logger.warning(f"Error ending Langfuse trace: {e}")

    @classmethod
    def record_retrieval(
        cls,
        query: str,
        retrieved_documents: List[Any],
        relevance_threshold: float,
        latency_seconds: float,
        raw_candidates_count: Optional[int] = None,
        trace_context: Optional[Dict[str, Any]] = None,
    ):
        """
        Records a retrieval observation span with source metadata and relevance scores.
        Bounded metadata avoids sensitive/large payloads.
        """
        client = cls.get_client()
        if not client:
            return

        try:
            from pathlib import Path
            sources_summary = []
            for doc in retrieved_documents:
                meta = getattr(doc, "metadata", {})
                fn = meta.get("filename") or Path(meta.get("source", "Unknown")).name
                page = meta.get("page")
                entry: Dict[str, Any] = {"filename": fn}
                if page is not None and str(page).lower() not in ("none", "unknown"):
                    entry["page"] = page
                if "relevance_score" in meta:
                    entry["relevance_score"] = meta["relevance_score"]
                sources_summary.append(entry)

            threshold_refused = len(retrieved_documents) == 0

            client.create_event(
                trace_context=trace_context,
                name="rag_retrieval",
                input={"query": query},
                output={
                    "chunks_retrieved": len(retrieved_documents),
                    "threshold_refused": threshold_refused,
                    "sources": sources_summary,
                },
                metadata={
                    "relevance_threshold": relevance_threshold,
                    "raw_candidates_count": raw_candidates_count if raw_candidates_count is not None else len(retrieved_documents),
                    "retrieval_latency_seconds": round(latency_seconds, 4),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to record retrieval in Langfuse: {e}")

    @classmethod
    def record_retrieval_error(
        cls,
        query: str,
        error: str,
        latency_seconds: float = 0.0,
        trace_context: Optional[Dict[str, Any]] = None,
    ):
        """
        Records a retrieval error event in Langfuse safely without leaking credentials or details.
        """
        client = cls.get_client()
        if not client:
            return

        try:
            client.create_event(
                trace_context=trace_context,
                name="rag_retrieval_error",
                input={"query": query},
                output={"error": "Vector store retrieval failed"},
                metadata={
                    "retrieval_latency_seconds": round(latency_seconds, 4),
                    "error_summary": "VectorStoreError",
                },
            )
        except Exception as e:
            logger.warning(f"Failed to record retrieval error in Langfuse: {e}")

    @classmethod
    def flush(cls):
        """Safely flushes any buffered Langfuse events."""
        client = cls.get_client()
        if client:
            try:
                client.flush()
            except Exception as e:
                logger.warning(f"Failed to flush Langfuse events: {e}")

    @classmethod
    def reset(cls):
        """Resets client singleton state (primarily for unit tests)."""
        cls._client = None
        cls._client_initialized = False
