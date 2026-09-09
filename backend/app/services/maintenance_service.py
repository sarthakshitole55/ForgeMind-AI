from pathlib import Path
from typing import Optional, Set

from app.config.settings import settings
from app.core.logger import logger
from app.core.exceptions import ValidationError
from app.rag.indexer import DocumentIndexer
from app.rag.vectorstore.chroma_store import ChromaVectorStore
from app.schemas.maintenance import (
    ActionExecutionResult,
    IssueSeverity,
    MaintenanceActionType,
    MaintenanceIssue,
    MaintenanceReport,
    MaintenanceStatus,
    SystemHealthDetails,
)


class MaintenanceService:
    """
    Deterministic maintenance and diagnostic service for ForgeMind-AI.
    Performs system health checks, document/vector consistency verification,
    and safe, allowlisted maintenance actions with verification.
    """

    ALLOWED_ACTIONS = {
        MaintenanceActionType.REINDEX_DOCUMENT,
        MaintenanceActionType.REMOVE_ORPHAN_VECTOR,
    }

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        indexer: Optional[DocumentIndexer] = None,
    ):
        self.vector_store = vector_store or ChromaVectorStore()
        self.indexer = indexer or DocumentIndexer(vector_store=self.vector_store)

    def validate_target(self, target: str) -> Path:
        """
        Validates target filename for strict path safety and PDF format.
        Reuses security controls to prevent path traversal and arbitrary filesystem operations.
        """
        if not target or not isinstance(target, str):
            raise ValidationError("Target filename cannot be empty")

        if (
            "\x00" in target
            or ".." in target
            or "/" in target
            or "\\" in target
            or ":" in target
        ):
            logger.warning(f"Path traversal attempt blocked in maintenance target: '{target}'")
            raise ValidationError("Invalid target: Path traversal detected")

        safe_filename = Path(target).name
        if safe_filename != target or safe_filename in (".", ".."):
            logger.warning(f"Invalid target name blocked in maintenance: '{target}'")
            raise ValidationError("Invalid target: Path traversal detected")

        if not safe_filename.lower().endswith(".pdf"):
            logger.warning(f"Non-PDF target blocked in maintenance: '{target}'")
            raise ValidationError("Invalid target: Only PDF documents (.pdf) are supported")

        data_dir = settings.get_data_dir()
        resolved_path = (data_dir / safe_filename).resolve()

        if (
            not resolved_path.is_relative_to(data_dir.resolve())
            or resolved_path.parent != data_dir.resolve()
            or resolved_path == data_dir.resolve()
        ):
            logger.warning(f"Path escape attempt blocked in maintenance: '{target}' -> '{resolved_path}'")
            raise ValidationError("Invalid target: Path traversal detected")

        return resolved_path

    def inspect_system_and_consistency(self) -> MaintenanceReport:
        """
        Performs comprehensive, deterministic inspection of storage, vector store,
        and document/index consistency without making external LLM calls.
        """
        checks_performed = [
            "storage_availability",
            "vector_store_connectivity",
            "document_index_consistency",
            "orphaned_vectors_detection",
            "file_integrity",
        ]

        issues: list[MaintenanceIssue] = []
        data_dir = settings.get_data_dir()

        # 1. Storage check
        storage_status = "healthy"
        try:
            if not data_dir.exists() or not data_dir.is_dir():
                storage_status = "unhealthy"
                issues.append(
                    MaintenanceIssue(
                        issue_type="STORAGE_UNAVAILABLE",
                        severity=IssueSeverity.CRITICAL,
                        resource="data_directory",
                        description=f"Document storage directory is missing or inaccessible: '{data_dir}'",
                        recommended_action=MaintenanceActionType.NONE,
                        auto_repair_safe=False,
                    )
                )
        except Exception as e:
            logger.error(f"Maintenance inspection failed checking storage: {e}", exc_info=True)
            storage_status = "unhealthy"
            issues.append(
                MaintenanceIssue(
                    issue_type="STORAGE_ERROR",
                    severity=IssueSeverity.CRITICAL,
                    resource="data_directory",
                    description=f"Error accessing storage directory: {e}",
                    recommended_action=MaintenanceActionType.NONE,
                    auto_repair_safe=False,
                )
            )

        # 2. Vector store check
        vector_store_status = "healthy"
        vector_chunk_count = 0
        indexed_doc_names: Set[str] = set()

        try:
            vector_chunk_count = self.vector_store.db._collection.count()
            # Inspect metadata safely to find all indexed document references
            data = self.vector_store.db._collection.get(include=["metadatas"])
            for meta in (data.get("metadatas") or []):
                if not meta:
                    continue
                fn = meta.get("filename")
                if not fn and "source" in meta:
                    fn = Path(meta["source"]).name
                if fn:
                    indexed_doc_names.add(fn)
        except Exception as e:
            logger.error(f"Maintenance inspection failed querying ChromaDB: {e}", exc_info=True)
            vector_store_status = "unhealthy"
            issues.append(
                MaintenanceIssue(
                    issue_type="VECTOR_STORE_UNAVAILABLE",
                    severity=IssueSeverity.CRITICAL,
                    resource="chroma_db",
                    description="ChromaDB vector store is unreachable or corrupted.",
                    recommended_action=MaintenanceActionType.NONE,
                    auto_repair_safe=False,
                )
            )

        # 3. Physical documents and file integrity
        physical_files = []
        if storage_status == "healthy":
            try:
                physical_files = list(data_dir.glob("*.pdf"))
            except Exception as e:
                logger.error(f"Failed to list physical PDF files: {e}", exc_info=True)

        physical_doc_names = {f.name for f in physical_files}

        # Check for empty/corrupt physical files
        for pdf_file in physical_files:
            try:
                if pdf_file.stat().st_size == 0:
                    issues.append(
                        MaintenanceIssue(
                            issue_type="CORRUPT_FILE",
                            severity=IssueSeverity.HIGH,
                            resource=pdf_file.name,
                            description=f"Document '{pdf_file.name}' is empty (0 bytes).",
                            recommended_action=MaintenanceActionType.NONE,
                            auto_repair_safe=False,
                        )
                    )
            except Exception:
                pass

        # Case A: Physical file exists but has no vector index in ChromaDB
        if storage_status == "healthy" and vector_store_status == "healthy":
            for pdf_file in physical_files:
                if pdf_file.stat().st_size == 0:
                    continue  # already flagged as corrupt

                # Verify if document is in indexed set
                is_indexed = pdf_file.name in indexed_doc_names
                if not is_indexed:
                    # Double check via get_document_chunks
                    chunks = self.vector_store.get_document_chunks(pdf_file.name, str(pdf_file))
                    if not chunks:
                        issues.append(
                            MaintenanceIssue(
                                issue_type="MISSING_INDEX",
                                severity=IssueSeverity.MEDIUM,
                                resource=pdf_file.name,
                                description=(
                                    f"Document '{pdf_file.name}' exists in storage but has no vector index "
                                    "in ChromaDB. It cannot be retrieved by the RAG agent."
                                ),
                                recommended_action=MaintenanceActionType.REINDEX_DOCUMENT,
                                auto_repair_safe=True,
                            )
                        )

            # Case B: Vector index contains chunks for a document that no longer exists in storage
            for indexed_name in indexed_doc_names:
                if indexed_name not in physical_doc_names:
                    issues.append(
                        MaintenanceIssue(
                            issue_type="ORPHANED_VECTORS",
                            severity=IssueSeverity.MEDIUM,
                            resource=indexed_name,
                            description=(
                                f"Vector store contains indexed chunks referencing '{indexed_name}', "
                                "but the physical file no longer exists in storage."
                            ),
                            recommended_action=MaintenanceActionType.REMOVE_ORPHAN_VECTOR,
                            auto_repair_safe=True,
                        )
                    )

        # 4. Compute overall status
        if storage_status == "unhealthy" or vector_store_status == "unhealthy":
            overall_status = MaintenanceStatus.ERROR
        elif any(
            issue.severity in (IssueSeverity.HIGH, IssueSeverity.CRITICAL)
            or issue.recommended_action in (
                MaintenanceActionType.REINDEX_DOCUMENT,
                MaintenanceActionType.REMOVE_ORPHAN_VECTOR,
            )
            for issue in issues
        ):
            overall_status = MaintenanceStatus.REPAIR_REQUIRED
        elif issues:
            overall_status = MaintenanceStatus.WARNING
        else:
            overall_status = MaintenanceStatus.HEALTHY

        health_details = SystemHealthDetails(
            storage=storage_status,
            vector_store=vector_store_status,
            physical_document_count=len(physical_files),
            vector_chunk_count=vector_chunk_count,
            indexed_document_count=len(indexed_doc_names),
        )

        return MaintenanceReport(
            status=overall_status,
            checks_performed=checks_performed,
            health=health_details,
            issues=issues,
        )

    def execute_action(
        self, action: MaintenanceActionType | str, target: str
    ) -> ActionExecutionResult:
        """
        Executes an allowlisted maintenance action with strict safety boundaries,
        pre-condition validation, and post-condition verification.
        """
        # Convert string to enum if necessary
        try:
            action_type = (
                action
                if isinstance(action, MaintenanceActionType)
                else MaintenanceActionType(action)
            )
        except ValueError:
            action_type = MaintenanceActionType.UNSUPPORTED

        # Safety Boundary: Check Allowlist
        if action_type not in self.ALLOWED_ACTIONS:
            logger.warning(f"Maintenance action rejected: '{action}' is not allowlisted.")
            return ActionExecutionResult(
                action=MaintenanceActionType.UNSUPPORTED,
                target=target,
                success=False,
                message=f"Action '{action}' is not supported. Allowed actions: {[a.value for a in self.ALLOWED_ACTIONS]}",
            )

        # Safety Boundary: Target Validation
        try:
            target_path = self.validate_target(target)
            safe_target = target_path.name
        except ValidationError as e:
            return ActionExecutionResult(
                action=action_type,
                target=target,
                success=False,
                message=f"Target validation rejected: {e.message}",
            )

        # Action 1: Re-index an existing document
        if action_type == MaintenanceActionType.REINDEX_DOCUMENT:
            return self._execute_reindex(safe_target, target_path)

        # Action 2: Remove orphaned vectors
        if action_type == MaintenanceActionType.REMOVE_ORPHAN_VECTOR:
            return self._execute_remove_orphan(safe_target, target_path)

        return ActionExecutionResult(
            action=MaintenanceActionType.UNSUPPORTED,
            target=safe_target,
            success=False,
            message="Unrecognized maintenance action execution path.",
        )

    def _execute_reindex(self, safe_target: str, target_path: Path) -> ActionExecutionResult:
        """Re-indexes an existing document safely via DocumentIndexer with verification."""
        # Pre-condition: Physical file must exist on disk
        if not target_path.exists() or not target_path.is_file():
            logger.warning(f"Re-indexing failed: Physical file not found for '{safe_target}'")
            return ActionExecutionResult(
                action=MaintenanceActionType.REINDEX_DOCUMENT,
                target=safe_target,
                success=False,
                message=f"Cannot re-index: physical file '{safe_target}' does not exist in storage.",
            )

        # Pre-condition: Non-empty file
        if target_path.stat().st_size == 0:
            return ActionExecutionResult(
                action=MaintenanceActionType.REINDEX_DOCUMENT,
                target=safe_target,
                success=False,
                message=f"Cannot re-index: file '{safe_target}' is empty (0 bytes).",
            )

        try:
            # 1. Clean up any stale or partial vectors first
            self.vector_store.delete_document(filename=safe_target, file_path=str(target_path))

            # 2. Re-index using existing DocumentIndexer pipeline
            index_result = self.indexer.index_pdf(str(target_path))

            # 3. Post-condition verification: Verify new chunks exist in vector store
            chunks = self.vector_store.get_document_chunks(safe_target, str(target_path))
            if not chunks:
                logger.error(f"Re-indexing verification failed for '{safe_target}': No chunks in vector store")
                return ActionExecutionResult(
                    action=MaintenanceActionType.REINDEX_DOCUMENT,
                    target=safe_target,
                    success=False,
                    message=f"Re-indexing failed post-verification: 0 chunks recorded in ChromaDB.",
                )

            logger.info(
                f"Successfully re-indexed '{safe_target}': {index_result.get('pages', 1)} pages, "
                f"{len(chunks)} chunks verified."
            )
            return ActionExecutionResult(
                action=MaintenanceActionType.REINDEX_DOCUMENT,
                target=safe_target,
                success=True,
                message=f"Successfully re-indexed '{safe_target}' ({index_result.get('pages', 1)} pages, {len(chunks)} chunks verified).",
                details={
                    "pages": index_result.get("pages"),
                    "chunks": len(chunks),
                },
            )
        except Exception as e:
            logger.error(f"Failed to re-index document '{safe_target}': {e}", exc_info=True)
            return ActionExecutionResult(
                action=MaintenanceActionType.REINDEX_DOCUMENT,
                target=safe_target,
                success=False,
                message=f"Re-indexing operation failed: {str(e)}",
            )

    def _execute_remove_orphan(self, safe_target: str, target_path: Path) -> ActionExecutionResult:
        """Removes orphaned vectors with strict safety validation and verification."""
        # Safety Check: Target must NOT exist on disk (cannot delete active files via orphan removal)
        if target_path.exists():
            logger.warning(
                f"Safety violation: Rejecting orphan removal for '{safe_target}' because physical file exists."
            )
            return ActionExecutionResult(
                action=MaintenanceActionType.REMOVE_ORPHAN_VECTOR,
                target=safe_target,
                success=False,
                message=f"Safety violation: Cannot remove vectors as orphan because physical file '{safe_target}' exists on disk.",
            )

        try:
            # Pre-condition: Verify that orphaned vectors exist before attempting removal
            existing_chunks = self.vector_store.get_document_chunks(safe_target)
            if not existing_chunks:
                return ActionExecutionResult(
                    action=MaintenanceActionType.REMOVE_ORPHAN_VECTOR,
                    target=safe_target,
                    success=False,
                    message=f"No vector records found in ChromaDB for '{safe_target}'.",
                )

            # Execution: Delete vectors using existing deletion lifecycle
            deleted_count = self.vector_store.delete_document(
                filename=safe_target, file_path=str(target_path)
            )

            # Post-condition verification: Verify vectors are completely removed
            remaining_chunks = self.vector_store.get_document_chunks(safe_target)
            if remaining_chunks:
                logger.error(
                    f"Orphan vector deletion verification failed for '{safe_target}': "
                    f"{len(remaining_chunks)} chunks remain."
                )
                return ActionExecutionResult(
                    action=MaintenanceActionType.REMOVE_ORPHAN_VECTOR,
                    target=safe_target,
                    success=False,
                    message=f"Orphan removal failed post-verification: {len(remaining_chunks)} vectors still present.",
                )

            logger.info(f"Successfully purged {deleted_count} orphaned vectors for '{safe_target}'")
            return ActionExecutionResult(
                action=MaintenanceActionType.REMOVE_ORPHAN_VECTOR,
                target=safe_target,
                success=True,
                message=f"Successfully purged {deleted_count} orphaned vectors for '{safe_target}'.",
                details={"deleted_chunks": deleted_count},
            )
        except Exception as e:
            logger.error(f"Failed to remove orphaned vectors for '{safe_target}': {e}", exc_info=True)
            return ActionExecutionResult(
                action=MaintenanceActionType.REMOVE_ORPHAN_VECTOR,
                target=safe_target,
                success=False,
                message=f"Orphan removal operation failed: {str(e)}",
            )
