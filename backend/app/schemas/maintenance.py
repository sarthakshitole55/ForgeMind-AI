from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MaintenanceStatus(str, Enum):
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    REPAIR_REQUIRED = "REPAIR_REQUIRED"
    ERROR = "ERROR"


class IssueSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class MaintenanceActionType(str, Enum):
    REINDEX_DOCUMENT = "REINDEX_DOCUMENT"
    REMOVE_ORPHAN_VECTOR = "REMOVE_ORPHAN_VECTOR"
    UNSUPPORTED = "UNSUPPORTED"
    NONE = "NONE"


class MaintenanceIssue(BaseModel):
    issue_type: str = Field(..., description="Issue classification (e.g. MISSING_INDEX, ORPHANED_VECTORS, CORRUPT_FILE)")
    severity: IssueSeverity = Field(..., description="Issue severity level")
    resource: str = Field(..., description="Target resource or filename affected")
    description: str = Field(..., description="Human-readable explanation of the issue")
    recommended_action: MaintenanceActionType = Field(..., description="Allowlisted maintenance action")
    auto_repair_safe: bool = Field(default=False, description="Whether automated repair is safe to execute")


class SystemHealthDetails(BaseModel):
    storage: str = Field(..., description="Storage subsystem status ('healthy', 'unhealthy')")
    vector_store: str = Field(..., description="Vector store subsystem status ('healthy', 'unhealthy')")
    physical_document_count: int = Field(default=0, description="Total physical PDF documents in storage directory")
    vector_chunk_count: int = Field(default=0, description="Total vector chunks stored in ChromaDB")
    indexed_document_count: int = Field(default=0, description="Distinct documents indexed in ChromaDB")


class ActionExecutionResult(BaseModel):
    action: MaintenanceActionType = Field(..., description="Maintenance action executed")
    target: str = Field(..., description="Target filename or resource")
    success: bool = Field(..., description="Whether the action succeeded and was verified")
    message: str = Field(..., description="Summary of action outcome")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Diagnostic action details")


class MaintenanceReport(BaseModel):
    status: MaintenanceStatus = Field(..., description="Overall maintenance status")
    checks_performed: List[str] = Field(..., description="List of checks performed during inspection")
    health: SystemHealthDetails = Field(..., description="Subsystem health metrics")
    issues: List[MaintenanceIssue] = Field(default_factory=list, description="Detected maintenance issues")
    diagnosis: Optional[str] = Field(default=None, description="Diagnostic summary and analysis")
    action_taken: Optional[ActionExecutionResult] = Field(default=None, description="Result of maintenance action, if executed")


from pydantic import BaseModel, Field, field_validator


class MaintenanceActionRequest(BaseModel):
    action: MaintenanceActionType = Field(..., description="Maintenance action to execute")
    target: str = Field(..., min_length=1, max_length=255, description="Target document filename (e.g. 'manual.pdf')")

    @field_validator("action", mode="before")
    @classmethod
    def normalize_action(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_norm = v.strip().upper().replace("-", "_")
            if v_norm in ("REINDEX", "RE_INDEX", "REINDEX_DOCUMENT"):
                return MaintenanceActionType.REINDEX_DOCUMENT
            if v_norm in ("REMOVE_ORPHAN", "REMOVE_ORPHAN_VECTOR", "CLEAN_ORPHAN", "PURGE_ORPHAN"):
                return MaintenanceActionType.REMOVE_ORPHAN_VECTOR
        return v
