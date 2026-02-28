"""
Pydantic state models for the entire workflow.
Every node boundary is type-checked.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────
class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    LOW = "LOW"
    CLEAR = "CLEAR"


class EscalationLevel(str, Enum):
    LOG = "LOG"
    WARNING = "WARNING"
    SUPERVISOR_ALERT = "SUPERVISOR_ALERT"


# ──────────────────────────────────────────────
# Node 1 Output — Parsed LLM perception
# ──────────────────────────────────────────────
class ParsedDecision(BaseModel):
    """Strict schema for fine-tuned Gemma output."""

    detected_ppe: list[str] = Field(default_factory=list)
    missing_ppe: list[str] = Field(default_factory=list)
    decision: Literal["ALLOW", "DENY"] = "DENY"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""

    @field_validator("detected_ppe", "missing_ppe", mode="before")
    @classmethod
    def normalize_ppe_items(cls, v):
        """Accept comma-separated string or list; normalize to lowercase stripped list."""
        if isinstance(v, str):
            return [x.strip().lower() for x in v.split(",") if x.strip()]
        return [str(x).strip().lower() for x in v if str(x).strip()]

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, v):
        """Safely coerce confidence to float."""
        try:
            val = float(v)
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return 0.0


# ──────────────────────────────────────────────
# Node 2 Output — Deterministic validation result
# ──────────────────────────────────────────────
class OshaViolation(BaseModel):
    code: str
    desc: str


class ValidatedDecision(BaseModel):
    """Pure Python validation output. LLM has no say here."""

    parsed: ParsedDecision
    severity: Severity
    final_decision: Literal["ALLOW", "DENY"]
    compliance_score: int = Field(ge=0, le=100)
    osha_violations: list[OshaViolation] = Field(default_factory=list)
    corrective_actions: list[str] = Field(default_factory=list)
    override_applied: bool = False
    override_reason: str = ""


# ──────────────────────────────────────────────
# Node 3 Output — Final incident output
# ──────────────────────────────────────────────
class EscalationAction(BaseModel):
    """Record of an auto-executed escalation action."""

    action: str          # "send_email", "create_ticket", "alert_supervisor"
    description: str
    executed: bool = True
    timestamp: str = ""


class ActOutput(BaseModel):
    """Final output after persistence and escalation."""

    incident_id: int
    escalation: EscalationLevel
    escalation_actions: list[EscalationAction] = Field(default_factory=list)
    report: str
    validated: ValidatedDecision


# ──────────────────────────────────────────────
# LangGraph Workflow State (TypedDict for LangGraph)
# ──────────────────────────────────────────────
from typing import TypedDict


class WorkflowState(TypedDict, total=False):
    """Flat state dict for LangGraph. All values serializable."""

    # Input
    image_path: str

    # Control
    retry_count: int
    status: str  # "analyzing" | "analyzed" | "decided" | "complete" | "failed"
    error: Optional[str]

    # Node outputs (stored as dicts for LangGraph serialization)
    parsed: Optional[dict]
    validated: Optional[dict]
    act_output: Optional[dict]
