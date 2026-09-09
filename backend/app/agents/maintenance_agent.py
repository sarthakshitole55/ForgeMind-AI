import json
import re
from typing import Optional, Tuple
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logger import logger
from app.core.observability import ObservabilityService
from app.schemas.maintenance import (
    ActionExecutionResult,
    MaintenanceActionType,
    MaintenanceReport,
    MaintenanceStatus,
)
from app.services.llm_services import LLMService
from app.services.maintenance_service import MaintenanceService


class MaintenanceAgent:
    """
    First-class Maintenance Agent for ForgeMind-AI.
    Inspects document and vector consistency deterministically,
    diagnoses system health, executes allowlisted maintenance actions,
    and returns strongly-typed, verifiable maintenance outcomes.
    """

    def __init__(
        self,
        maintenance_service: Optional[MaintenanceService] = None,
        llm_service: Optional[LLMService] = None,
    ):
        self.service = maintenance_service or MaintenanceService()
        self.llm = llm_service

    def _parse_action_intent(self, question: str) -> Tuple[MaintenanceActionType, Optional[str]]:
        """
        Deterministically parses intent from user query for explicitly requested
        maintenance operations.
        """
        q = question.strip()

        # Check for reindex pattern: "reindex document.pdf" or "re-index document.pdf"
        reindex_match = re.search(
            r"(?:re-?index)\s+(?:file\s+|document\s+)?([a-zA-Z0-9_\-\.]+\.pdf)",
            q,
            re.IGNORECASE,
        )
        if reindex_match:
            return MaintenanceActionType.REINDEX_DOCUMENT, reindex_match.group(1)

        # Check for orphan cleanup pattern: "remove orphan doc.pdf" or "clean up orphaned vectors for doc.pdf"
        orphan_match = re.search(
            r"(?:remove|delete|clean(?:\s+up)?|purge)\s+(?:orphan(?:ed)?(?:\s+vectors?)?(?:\s+for)?\s+)?([a-zA-Z0-9_\-\.]+\.pdf)",
            q,
            re.IGNORECASE,
        )
        if orphan_match and ("orphan" in q.lower() or "vector" in q.lower()):
            return MaintenanceActionType.REMOVE_ORPHAN_VECTOR, orphan_match.group(1)

        # Check for repair all requested
        if re.search(r"(?:repair|fix)\s+(?:all|issues|problems)", q, re.IGNORECASE):
            return MaintenanceActionType.NONE, "ALL"

        return MaintenanceActionType.NONE, None

    def _format_deterministic_report(
        self, report: MaintenanceReport, action_result: Optional[ActionExecutionResult] = None
    ) -> str:
        """
        Builds a rich, clear Markdown report directly from the deterministic inspection data.
        Guaranteed to work even if AI model providers are completely offline.
        """
        lines = [
            "### 🛠️ ForgeMind System Maintenance Report",
            "",
            f"**Overall Status**: `{report.status.value}`",
            f"- **Storage Subsystem**: `{report.health.storage}` ({report.health.physical_document_count} physical PDFs)",
            f"- **Vector Store Subsystem**: `{report.health.vector_store}` ({report.health.vector_chunk_count} chunks, {report.health.indexed_document_count} indexed documents)",
            "",
            "#### Checks Performed:",
        ]
        for check in report.checks_performed:
            lines.append(f"- `✓` {check.replace('_', ' ').title()}")

        lines.append("")
        if action_result:
            status_icon = "✓" if action_result.success else "✗"
            lines.extend([
                "#### Maintenance Action Executed:",
                f"- **Action**: `{action_result.action.value}`",
                f"- **Target**: `{action_result.target}`",
                f"- **Result**: {status_icon} {action_result.message}",
                "",
            ])

        if report.issues:
            lines.append("#### Detected Issues:")
            for issue in report.issues:
                lines.append(
                    f"- **[{issue.severity.value}]** `{issue.resource}`: {issue.description} "
                    f"*(Recommended: `{issue.recommended_action.value}`, Auto-repair safe: {issue.auto_repair_safe})*"
                )
        else:
            lines.append("#### Detected Issues:")
            lines.append("- None. All physical documents match their corresponding vector representations.")

        lines.append("")
        if report.status == MaintenanceStatus.HEALTHY:
            lines.append("✅ **Diagnosis**: The RAG and document systems are consistent and fully operational.")
        elif report.status == MaintenanceStatus.REPAIR_REQUIRED:
            lines.append("⚠️ **Diagnosis**: Maintenance repairs are recommended. You can ask me to re-index specific documents or remove orphaned vectors.")
        elif report.status == MaintenanceStatus.WARNING:
            lines.append("⚠️ **Diagnosis**: Minor warnings detected. System remains operational.")
        else:
            lines.append("❌ **Diagnosis**: Critical subsystem errors detected. Check storage and ChromaDB connectivity.")

        return "\n".join(lines)

    def _synthesize_with_llm(
        self, report: MaintenanceReport, question: str, deterministic_summary: str
    ) -> str:
        """
        Uses LLM to synthesize a natural, contextual explanation if available.
        Falls back to the deterministic report upon any error.
        """
        if self.llm is None:
            try:
                self.llm = LLMService()
            except Exception:
                return deterministic_summary

        prompt = (
            "You are ForgeMind's dedicated Maintenance AI Agent. "
            "You have just completed a deterministic system and consistency inspection. "
            "Present the findings clearly and professionally to the operator, keeping the exact status, "
            "issues, and action results intact. Use GitHub-style markdown formatting.\n\n"
            f"User Query: {question}\n\n"
            f"Inspection Data:\n{deterministic_summary}\n\n"
            "Provide a clear, helpful response explaining what was checked, any problems found, "
            "and what actions were or should be taken."
        )

        try:
            response = self.llm.chat([
                SystemMessage(content="You are ForgeMind's system maintenance specialist."),
                HumanMessage(content=prompt),
            ])
            if response and getattr(response, "content", None):
                return str(response.content)
        except Exception as e:
            logger.warning(f"LLM diagnostic synthesis failed ({e}). Using deterministic summary.")

        return deterministic_summary

    def handle(self, question: str) -> dict:
        """
        Primary handler for maintenance requests.
        Coordinates deterministic inspection, validated action execution,
        and diagnostic reporting.
        """
        action_type, target = self._parse_action_intent(question)
        action_result: Optional[ActionExecutionResult] = None

        # Execute safe allowlisted action if explicitly requested
        if action_type in (
            MaintenanceActionType.REINDEX_DOCUMENT,
            MaintenanceActionType.REMOVE_ORPHAN_VECTOR,
        ) and target:
            logger.info(f"Executing requested maintenance action: {action_type.value} on '{target}'")
            action_result = self.service.execute_action(action_type, target)

        # Run deterministic system inspection
        report = self.service.inspect_system_and_consistency()
        if action_result:
            report.action_taken = action_result

        # Handle 'repair all' request if specified
        if target == "ALL" and report.issues:
            auto_repairable = [i for i in report.issues if i.auto_repair_safe]
            if auto_repairable:
                # Safely execute first repairable issue in MVP
                first_issue = auto_repairable[0]
                action_result = self.service.execute_action(
                    first_issue.recommended_action, first_issue.resource
                )
                report = self.service.inspect_system_and_consistency()
                report.action_taken = action_result

        # Format deterministic summary
        summary = self._format_deterministic_report(report, action_result)
        report.diagnosis = summary

        # Synthesize final response (LLM with deterministic fallback)
        final_answer = self._synthesize_with_llm(report, question, summary)

        # Record event in Langfuse if available
        client = ObservabilityService.get_client()
        if client:
            try:
                client.create_event(
                    name="maintenance_operation",
                    input={"question": question, "action_requested": action_type.value if action_type else None},
                    output={
                        "status": report.status.value,
                        "issues_count": len(report.issues),
                        "action_success": action_result.success if action_result else None,
                    },
                    metadata={
                        "checks": report.checks_performed,
                        "storage_status": report.health.storage,
                        "vector_store_status": report.health.vector_store,
                        "physical_docs": report.health.physical_document_count,
                        "vector_chunks": report.health.vector_chunk_count,
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to record maintenance event in Langfuse: {e}")

        return {
            "answer": final_answer,
            "context": f"Maintenance Status: {report.status.value}",
            "report": report,
        }


# Global agent instance for LangGraph node
_maintenance_agent = MaintenanceAgent()


def maintenance_node(state: dict) -> dict:
    """
    LangGraph node for maintenance routing.
    Executes the maintenance agent and attaches the diagnostic report to the graph state.
    """
    question = state.get("question", "")
    logger.info(f"Executing Maintenance Node | Query: '{question}'")

    result = _maintenance_agent.handle(question)

    state["answer"] = result["answer"]
    state["context"] = result.get("context", "")

    return state