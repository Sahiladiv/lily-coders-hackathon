"""
SQLite persistence layer.
Handles schema creation, incident logging, and offense counting.
No ORM. Raw SQL. Minimal and reliable.
SQLite connection + tables
"""
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timezone

from app.config import DB_PATH

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path       TEXT    NOT NULL,
    severity         TEXT    NOT NULL,
    final_decision   TEXT    NOT NULL,
    compliance_score INTEGER NOT NULL,
    detected_ppe     TEXT    NOT NULL DEFAULT '',
    missing_ppe      TEXT    NOT NULL DEFAULT '',
    confidence       REAL    NOT NULL DEFAULT 0.0,
    osha_codes       TEXT    NOT NULL DEFAULT '',
    escalation       TEXT    NOT NULL,
    override_applied INTEGER NOT NULL DEFAULT 0,
    override_reason  TEXT    NOT NULL DEFAULT '',
    report           TEXT    NOT NULL DEFAULT '',
    created_at       TEXT    NOT NULL
);
"""


def get_connection() -> sqlite3.Connection:
    """Get SQLite connection. Creates DB and schema if needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    conn.commit()
    logger.info(f"SQLite connected: {DB_PATH}")
    return conn


def log_incident(
    conn: sqlite3.Connection,
    image_path: str,
    validated: dict,
    escalation: str,
    report: str,
) -> tuple[int, str]:
    """
    Insert incident record. Returns (incident_id, timestamp).
    """
    parsed = validated["parsed"]
    now = datetime.now(timezone.utc).isoformat()

    cur = conn.execute(
        """
        INSERT INTO incidents (
            image_path, severity, final_decision, compliance_score,
            detected_ppe, missing_ppe, confidence, osha_codes,
            escalation, override_applied, override_reason,
            report, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            image_path,
            validated["severity"],
            validated["final_decision"],
            validated["compliance_score"],
            ",".join(parsed["detected_ppe"]),
            ",".join(parsed["missing_ppe"]),
            parsed.get("confidence", 0.0),
            ",".join(v["code"] for v in validated.get("osha_violations", [])),
            escalation,
            1 if validated.get("override_applied") else 0,
            validated.get("override_reason", ""),
            report,
            now,
        ),
    )
    conn.commit()

    incident_id = cur.lastrowid
    logger.info(f"Incident logged: id={incident_id}, severity={validated['severity']}")
    return incident_id, now


def count_offenses(conn: sqlite3.Connection) -> int:
    """Count total non-CLEAR incidents for escalation calculation."""
    cur = conn.execute(
        "SELECT COUNT(*) as cnt FROM incidents WHERE severity != 'CLEAR'"
    )
    return cur.fetchone()["cnt"]


def get_recent_incidents(conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    """Fetch recent incidents for dashboard/API."""
    cur = conn.execute(
        "SELECT * FROM incidents ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    return [dict(row) for row in cur.fetchall()]
