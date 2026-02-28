"""
Centralized prompt templates.
Vision prompts are in gemma_client.py (co-located with inference).
Report prompts are in llm/client.py (co-located with usage).
This file holds any additional prompt utilities if needed.
"""

# Report template used by Node 3 (ACT)
INCIDENT_REPORT_TEMPLATE = """
=== WORKPLACE SAFETY INCIDENT REPORT ===
Incident ID:       {incident_id}
Timestamp:         {timestamp}

Decision:          {final_decision}
Severity:          {severity}
Compliance Score:  {compliance_score}%
Override Applied:  {override_applied}
{override_reason_line}
Detected PPE:     {detected_ppe}
Missing PPE:      {missing_ppe}
LLM Confidence:   {confidence}

OSHA Violations:
{osha_violations}

Corrective Actions:
{corrective_actions}

Escalation Level:  {escalation}
===========================================
""".strip()


def format_report(
    incident_id: int,
    timestamp: str,
    validated: dict,
    escalation: str,
) -> str:
    """Build report from template. Pure string formatting, no LLM."""
    parsed = validated["parsed"]

    detected_str = ", ".join(parsed["detected_ppe"]) or "None"
    missing_str = ", ".join(parsed["missing_ppe"]) or "None"
    confidence_str = f"{parsed.get('confidence', 0.0):.2f}"

    osha_lines = "\n".join(
        f"  - {v['code']}: {v['desc']}"
        for v in validated.get("osha_violations", [])
    ) or "  None"

    action_lines = "\n".join(
        f"  - {a}"
        for a in validated.get("corrective_actions", [])
    ) or "  None"

    override_reason_line = ""
    if validated.get("override_applied"):
        override_reason_line = f"Override Reason:  {validated.get('override_reason', 'N/A')}\n"

    return INCIDENT_REPORT_TEMPLATE.format(
        incident_id=incident_id,
        timestamp=timestamp,
        final_decision=validated["final_decision"],
        severity=validated["severity"],
        compliance_score=validated["compliance_score"],
        override_applied=validated["override_applied"],
        override_reason_line=override_reason_line,
        detected_ppe=detected_str,
        missing_ppe=missing_str,
        confidence=confidence_str,
        osha_violations=osha_lines,
        corrective_actions=action_lines,
        escalation=escalation,
    )
