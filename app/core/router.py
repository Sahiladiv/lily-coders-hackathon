"""
Router — Dispatches FunctionGemma ToolCalls to read-only query handlers.

Only handles:
  - get_incidents  → SQLite query
  - get_report     → SQLite query
  - get_stats      → SQLite aggregation

analyze_image and escalate are AUTOMATED in the LangGraph pipeline.
They do NOT flow through this router.
"""
import logging
from typing import Any

from app.llm.function_gemma import ToolCall
# from app.storage.sqlite import get_connection, get_recent_incidents
from app.storage.mysql import get_connection, get_recent_incidents
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Dispatch
# ──────────────────────────────────────────────

def dispatch(tool_call: ToolCall) -> dict[str, Any]:
    """Route a validated ToolCall to the appropriate read-only handler."""
    handlers = {
        "get_incidents": _handle_get_incidents,
        "get_report": _handle_get_report,
        "get_stats": _handle_get_stats,
    }

    handler = handlers.get(tool_call.tool_name)
    if not handler:
        return {
            "success": False,
            "error": f"No handler for tool: {tool_call.tool_name}",
        }

    try:
        return handler(tool_call.arguments)
    except Exception as e:
        logger.error(f"[Router] Handler {tool_call.tool_name} failed: {e}")
        return {
            "success": False,
            "tool": tool_call.tool_name,
            "error": str(e),
        }


# ──────────────────────────────────────────────
# Handlers
# ──────────────────────────────────────────────

def _handle_get_incidents(args: dict) -> dict:
    """Fetch recent incidents from SQLite."""
    limit = args.get("limit", 10)
    limit = min(max(1, int(limit)), 100)

    conn = get_connection()
    try:
        incidents = get_recent_incidents(conn, limit=limit)
        return {
            "success": True,
            "tool": "get_incidents",
            "count": len(incidents),
            "incidents": incidents,
        }
    finally:
        conn.close()


def _handle_get_report(args: dict) -> dict:
    """Get a single incident report by ID."""
    print("[INFO] - getting incident report")

    incident_id = args.get("incident_id")
    if incident_id is None:
        return {"success": False, "error": "No incident_id provided"}

    conn = get_connection()

    try:
        cursor = conn.cursor(dictionary=True)

        # Fetch incident
        cursor.execute(
            "SELECT * FROM incidents WHERE id = %s",
            (int(incident_id),),
        )
        row = cursor.fetchone()

        if not row:
            cursor.close()
            return {
                "success": False,
                "tool": "get_report",
                "error": f"Incident #{incident_id} not found",
            }

        incident = row

        # Try fetching escalation actions (if table exists)
        escalation_actions = []
        try:
            esc_cursor = conn.cursor(dictionary=True)
            esc_cursor.execute(
                "SELECT * FROM escalation_log WHERE incident_id = %s ORDER BY id",
                (int(incident_id),),
            )
            escalation_actions = esc_cursor.fetchall()
            esc_cursor.close()
        except Exception:
            escalation_actions = []

        cursor.close()

        return {
            "success": True,
            "tool": "get_report",
            "incident": incident,
            "escalation_actions": escalation_actions,
        }

    finally:
        conn.close()


def _handle_get_stats(args: dict) -> dict:
    """Get aggregate safety statistics."""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)  # Important
    try:
        # Total incidents
        cursor.execute("SELECT COUNT(*) as cnt FROM incidents")
        total = cursor.fetchone()["cnt"]

        # By severity
        cursor.execute(
            "SELECT severity, COUNT(*) as cnt FROM incidents GROUP BY severity"
        )
        severity_rows = cursor.fetchall()
        by_severity = {row["severity"]: row["cnt"] for row in severity_rows}

        # By decision
        cursor.execute(
            "SELECT final_decision, COUNT(*) as cnt FROM incidents GROUP BY final_decision"
        )
        decision_rows = cursor.fetchall()
        by_decision = {row["final_decision"]: row["cnt"] for row in decision_rows}

        # Average compliance
        cursor.execute(
            "SELECT AVG(compliance_score) as avg_score FROM incidents"
        )
        avg_row = cursor.fetchone()
        avg_compliance = round(avg_row["avg_score"], 1) if avg_row["avg_score"] else 0.0

        # Most common missing PPE
        cursor.execute(
            "SELECT missing_ppe FROM incidents WHERE missing_ppe != ''"
        )
        all_missing = cursor.fetchall()

        ppe_counts = {}
        for row in all_missing:
            for item in row["missing_ppe"].split(","):
                item = item.strip()
                if item:
                    ppe_counts[item] = ppe_counts.get(item, 0) + 1

        # Total escalation actions
        try:
            cursor.execute("SELECT COUNT(*) as cnt FROM escalation_log")
            esc_total = cursor.fetchone()["cnt"]

            cursor.execute(
                "SELECT action, COUNT(*) as cnt FROM escalation_log GROUP BY action"
            )
            esc_by_action = cursor.fetchall()
            escalation_summary = {
                row["action"]: row["cnt"] for row in esc_by_action
            }
        except Exception:
            esc_total = 0
            escalation_summary = {}

        return {
            "success": True,
            "tool": "get_stats",
            "total_incidents": total,
            "by_severity": by_severity,
            "by_decision": by_decision,
            "avg_compliance_score": avg_compliance,
            "most_common_missing_ppe": dict(
                sorted(ppe_counts.items(), key=lambda x: x[1], reverse=True)
            ),
            "total_escalation_actions": esc_total,
            "escalations_by_type": escalation_summary,
        }

    finally:
        cursor.close()
        conn.close()
