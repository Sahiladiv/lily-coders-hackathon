"""
LangGraph Workflow — 3 nodes, 1 conditional retry, fully deterministic.

Graph:
    [ANALYZE] --parse ok--> [DECIDE] --> [ACT] --> END
        |                                           
        +--parse fail--> retry (max 3) --> [ANALYZE]
        +--max retries--> END (failed)

No recursion. No dynamic nodes. No unbounded loops.
"""
import logging

from langgraph.graph import StateGraph, END

from app.core.state import (
    WorkflowState,
    ParsedDecision,
    ValidatedDecision,
    OshaViolation,
    ActOutput,
    EscalationAction,
    Severity,
    EscalationLevel,
)
from app.core.guardrails import parse_llm_output, ParseError
from app.llm.gemma_client import analyze_image
from app.llm.fallback import analyze_ppe_with_openai
from app.llm.prompts import format_report
from app.llm.client import rephrase_report
# from app.storage.sqlite import get_connection, log_incident, count_offenses
from app.storage.mysql import get_connection, log_incident, count_worker_offenses as count_offenses 
from app.config import (
    REQUIRED_PPE,
    CRITICAL_PPE,
    SEVERITY_RULES,
    OSHA_REFERENCES,
    CORRECTIVE_ACTIONS,
    ESCALATION_THRESHOLDS,
    MAX_RETRIES,
    MAX_GRAPH_ITERATIONS,
    LOW_CONFIDENCE_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
# NODE 1: ANALYZE — Multimodal Perception + Parse
# ═══════════════════════════════════════════════
def analyze_node(state: WorkflowState) -> WorkflowState:
    """
    Send image to fine-tuned Gemma → parse structured output.
    On parse failure → increment retry, set status for conditional edge.
    """
    retry_count = state.get("retry_count", 0)
    image_path = state["image_path"]

    logger.info(f"[ANALYZE] Attempt {retry_count + 1}/{MAX_RETRIES} for {image_path}")

    try:
        # Call LM Studio
        # raw_output = analyze_image(image_path)

        # if not raw_output:
            # print("Empty response from Gemma")

        raw_output = analyze_ppe_with_openai(image_path)
        print(f"OpenAI Fallback Output:\n{raw_output}\n")  # Debug print
        # Strict parse
        parsed = parse_llm_output(raw_output)

        logger.info(f"[ANALYZE] Success — decision={parsed.decision}, confidence={parsed.confidence}")

        return {
            **state,
            "parsed": parsed.model_dump(),
            "status": "analyzed",
            "error": None,
        }

    except ParseError as e:
        logger.warning(f"[ANALYZE] Parse failed (attempt {retry_count + 1}): {e}")
        new_retry = retry_count + 1
        return {
            **state,
            "retry_count": new_retry,
            "error": f"Parse error: {e}",
            "status": "retry" if new_retry < MAX_RETRIES else "failed",
        }

    except Exception as e:
        logger.error(f"[ANALYZE] Unexpected error (attempt {retry_count + 1}): {e}")
        new_retry = retry_count + 1
        return {
            **state,
            "retry_count": new_retry,
            "error": f"Inference error: {e}",
            "status": "retry" if new_retry < MAX_RETRIES else "failed",
        }


# ═══════════════════════════════════════════════
# NODE 2: DECIDE — Pure Python Deterministic Logic
# ═══════════════════════════════════════════════
def decide_node(state: WorkflowState) -> WorkflowState:
    """
    Deterministic validation. No LLM. Python is the authority.
    - Compute severity
    - Map OSHA references
    - Generate corrective actions
    - Override LLM if needed
    - Compute compliance score
    """
    parsed = ParsedDecision(**state["parsed"])

    # Only count PPE items that are in our known set
    missing = set(parsed.missing_ppe) & REQUIRED_PPE
    detected = set(parsed.detected_ppe) & REQUIRED_PPE
    missing_critical = missing & CRITICAL_PPE

    logger.info(f"[DECIDE] Missing: {missing}, Critical missing: {missing_critical}")

    # ── Severity ──
    severity = Severity.CLEAR
    for sev_name, check_fn in SEVERITY_RULES:
        if check_fn(len(missing_critical), len(missing)):
            severity = Severity(sev_name)
            break

    # ── Decision Override ──
    final_decision = parsed.decision
    override_applied = False
    override_reason = ""

    # Override 1: Critical PPE missing but LLM said ALLOW
    if missing_critical and parsed.decision == "ALLOW":
        final_decision = "DENY"
        override_applied = True
        override_reason = f"Critical PPE missing ({', '.join(missing_critical)}) — LLM ALLOW overridden to DENY"
        logger.warning(f"[DECIDE] Override: {override_reason}")

    # Override 2: Low confidence → default to DENY
    if parsed.confidence < LOW_CONFIDENCE_THRESHOLD and parsed.decision == "ALLOW":
        final_decision = "DENY"
        override_applied = True
        override_reason = f"Low confidence ({parsed.confidence:.2f}) — defaulting to DENY for safety"
        logger.warning(f"[DECIDE] Override: {override_reason}")

    # Override 3: Nothing missing → ensure ALLOW
    if not missing:
        final_decision = "ALLOW"
        if parsed.decision == "DENY":
            override_applied = True
            override_reason = "No PPE missing — LLM DENY overridden to ALLOW"

    # ── Compliance Score ──
    if len(REQUIRED_PPE) > 0:
        compliance_score = int((len(detected) / len(REQUIRED_PPE)) * 100)
    else:
        compliance_score = 100

    # ── OSHA Violations ──
    osha_violations = [
        OshaViolation(code=OSHA_REFERENCES[item]["code"], desc=OSHA_REFERENCES[item]["desc"])
        for item in missing
        if item in OSHA_REFERENCES
    ]

    # ── Corrective Actions ──
    corrective_actions = [
        CORRECTIVE_ACTIONS[item]
        for item in missing
        if item in CORRECTIVE_ACTIONS
    ]

    validated = ValidatedDecision(
        parsed=parsed,
        severity=severity,
        final_decision=final_decision,
        compliance_score=compliance_score,
        osha_violations=osha_violations,
        corrective_actions=corrective_actions,
        override_applied=override_applied,
        override_reason=override_reason,
    )

    logger.info(
        f"[DECIDE] Final: decision={final_decision}, severity={severity.value}, "
        f"score={compliance_score}, override={override_applied}"
    )

    return {**state, "validated": validated.model_dump(), "status": "decided"}


# ═══════════════════════════════════════════════
# NODE 3: ACT — Persist + Escalate + Report
# ═══════════════════════════════════════════════
def act_node(state: WorkflowState) -> WorkflowState:
    """
    Persist to SQLite, auto-escalate based on severity + offense count,
    generate report. Escalation is AUTOMATIC — not user-triggered.
    """
    validated = state["validated"]
    image_path = state["image_path"]

    conn = get_connection()

    try:
        # ── Escalation Level ──
        prior_offenses = count_offenses(conn)
        offense_number = prior_offenses + 1 if validated["severity"] != "CLEAR" else prior_offenses
        esc_key = min(offense_number, max(ESCALATION_THRESHOLDS.keys()))
        esc_key = max(esc_key, min(ESCALATION_THRESHOLDS.keys()))
        escalation = EscalationLevel(ESCALATION_THRESHOLDS.get(esc_key, "LOG"))

        # ── Auto-Execute Escalation Actions ──
        # Determined by severity + escalation level, not user input
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        escalation_actions: list[EscalationAction] = []

        severity = validated["severity"]

        # SUPERVISOR_ALERT level → all three actions
        if escalation == EscalationLevel.SUPERVISOR_ALERT:
            escalation_actions = [
                EscalationAction(
                    action="alert_supervisor",
                    description=f"URGENT: Repeat safety violation (offense #{offense_number}). "
                                f"Severity: {severity}. Immediate supervisor review required.",
                    timestamp=now,
                ),
                EscalationAction(
                    action="send_email",
                    description=f"Safety incident auto-notification: offense #{offense_number}, "
                                f"severity {severity}.",
                    timestamp=now,
                ),
                EscalationAction(
                    action="create_ticket",
                    description=f"Safety ticket auto-created: offense #{offense_number}, "
                                f"severity {severity}. Corrective actions required.",
                    timestamp=now,
                ),
            ]

        # WARNING level → email + ticket
        elif escalation == EscalationLevel.WARNING:
            escalation_actions = [
                EscalationAction(
                    action="send_email",
                    description=f"Safety warning: second offense detected. Severity: {severity}.",
                    timestamp=now,
                ),
                EscalationAction(
                    action="create_ticket",
                    description=f"Safety ticket: offense #{offense_number}, severity {severity}.",
                    timestamp=now,
                ),
            ]

        # LOG level + CRITICAL/HIGH severity → still create a ticket
        elif severity in ("CRITICAL", "HIGH"):
            escalation_actions = [
                EscalationAction(
                    action="create_ticket",
                    description=f"First offense but {severity} severity. Ticket auto-created.",
                    timestamp=now,
                ),
            ]

        # Execute: log all actions to escalation_log table
        conn.execute(
            """CREATE TABLE IF NOT EXISTS escalation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id INTEGER,
                action TEXT NOT NULL,
                description TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )"""
        )
        conn.commit()

        # ── Persist Incident ──
        incident_id, timestamp = log_incident(
            conn, image_path, validated, escalation.value, ""
        )

        # Now log escalation actions with the incident_id
        for ea in escalation_actions:
            conn.execute(
                "INSERT INTO escalation_log (incident_id, action, description, created_at) "
                "VALUES (?, ?, ?, ?)",
                (incident_id, ea.action, ea.description, ea.timestamp),
            )
        conn.commit()

        if escalation_actions:
            logger.info(
                f"[ACT] Auto-escalated incident #{incident_id}: "
                f"{[ea.action for ea in escalation_actions]}"
            )

        # ── Report ──
        template_report = format_report(incident_id, timestamp, validated, escalation.value)
        

        # Optional: LLM rephrase (safe fallback to template)
        final_report = rephrase_report(template_report)
        print(f"Rephrased Report:\n{final_report}\n")  # Debug print
        # Update report in DB
        conn.execute(
            "UPDATE incidents SET report = ? WHERE id = ?",
            (final_report, incident_id),
        )
        conn.commit()

        act_output = ActOutput(
            incident_id=incident_id,
            escalation=escalation,
            escalation_actions=escalation_actions,
            report=final_report,
            validated=ValidatedDecision(**validated),
        )

        logger.info(
            f"[ACT] Incident #{incident_id} — "
            f"escalation={escalation.value}, severity={severity}, "
            f"actions={len(escalation_actions)}"
        )

        return {**state, "act_output": act_output.model_dump(), "status": "complete"}

    finally:
        conn.close()


# ═══════════════════════════════════════════════
# CONDITIONAL EDGE — Retry routing
# ═══════════════════════════════════════════════
def route_after_analyze(state: WorkflowState) -> str:
    """
    Deterministic routing after ANALYZE node.
    Three outcomes: continue, retry, or fail.
    """
    status = state.get("status", "failed")

    if status == "analyzed":
        return "continue"
    elif status == "retry":
        return "retry"
    else:
        return "failed"


# ═══════════════════════════════════════════════
# BUILD GRAPH
# ═══════════════════════════════════════════════
def build_graph():
    """
    Construct the 3-node LangGraph workflow.
    Returns compiled graph ready for .invoke().
    """
    graph = StateGraph(WorkflowState)

    # Add nodes
    graph.add_node("analyze", analyze_node)
    graph.add_node("decide", decide_node)
    graph.add_node("act", act_node)

    # Entry point
    graph.set_entry_point("analyze")

    # Conditional edge: analyze → decide | retry | end
    graph.add_conditional_edges(
        "analyze",
        route_after_analyze,
        {
            "continue": "decide",
            "retry": "analyze",   # bounded by MAX_RETRIES in node logic
            "failed": END,
        },
    )

    # Linear edges
    graph.add_edge("decide", "act")
    graph.add_edge("act", END)

    # Compile with recursion limit as safety net
    compiled = graph.compile()

    logger.info("LangGraph workflow compiled: analyze → decide → act")
    return compiled
