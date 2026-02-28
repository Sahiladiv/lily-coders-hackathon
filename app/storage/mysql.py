"""
MySQL persistence layer.
Handles schema creation, workers, incidents, offense tracking.
Raw SQL. Production-ready. No ORM.
"""

import logging
from datetime import datetime, timezone
import mysql.connector
from mysql.connector import Error

from app.config import MYSQL_CONFIG

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────

WORKERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS workers (
    id INT AUTO_INCREMENT PRIMARY KEY,
    employee_id VARCHAR(100) NOT NULL UNIQUE,
    full_name VARCHAR(255) NOT NULL,
    role VARCHAR(100),
    department VARCHAR(100),
    created_at DATETIME NOT NULL
);
"""

INCIDENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    id INT AUTO_INCREMENT PRIMARY KEY,

    worker_id INT NOT NULL,

    image_path VARCHAR(512) NOT NULL,
    severity VARCHAR(50) NOT NULL,
    final_decision VARCHAR(100) NOT NULL,
    compliance_score INT NOT NULL,

    detected_ppe TEXT NOT NULL,
    missing_ppe TEXT NOT NULL,
    confidence FLOAT NOT NULL DEFAULT 0.0,
    osha_codes TEXT NOT NULL,

    escalation VARCHAR(100) NOT NULL,
    override_applied BOOLEAN NOT NULL DEFAULT FALSE,
    override_reason TEXT NOT NULL,
    report TEXT NOT NULL,

    created_at DATETIME NOT NULL,

    FOREIGN KEY (worker_id)
        REFERENCES workers(id)
        ON DELETE CASCADE
);
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_worker_severity ON incidents(worker_id, severity);",
    "CREATE INDEX IF NOT EXISTS idx_created_at ON incidents(created_at);",
]


# ──────────────────────────────────────────────
# Connection
# ──────────────────────────────────────────────

def get_connection():
    """
    Create MySQL connection and ensure schema exists.
    """
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()

        cursor.execute(WORKERS_SCHEMA)
        cursor.execute(INCIDENTS_SCHEMA)

        for idx in INDEXES:
            try:
                cursor.execute(idx)
            except Exception:
                pass  # MySQL versions may differ

        conn.commit()
        cursor.close()

        logger.info("MySQL connected and schema ensured.")
        return conn

    except Error as e:
        logger.error(f"MySQL connection failed: {e}")
        raise


# ──────────────────────────────────────────────
# Worker Management
# ──────────────────────────────────────────────

def get_or_create_worker(
    conn,
    employee_id: str,
    full_name: str,
    role: str = "",
    department: str = "",
) -> int:
    """
    Returns worker_id. Creates worker if not exists.
    """
    cursor = conn.cursor(dictionary=True)

    cursor.execute(
        "SELECT id FROM workers WHERE employee_id = %s",
        (employee_id,),
    )
    result = cursor.fetchone()

    if result:
        cursor.close()
        return result["id"]

    now = datetime.now(timezone.utc)

    cursor.execute(
        """
        INSERT INTO workers (employee_id, full_name, role, department, created_at)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (employee_id, full_name, role, department, now),
    )
    conn.commit()

    worker_id = cursor.lastrowid
    cursor.close()

    logger.info(f"Worker created: {employee_id}")
    return worker_id


def get_worker_by_id(conn, worker_id: int) -> dict | None:
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM workers WHERE id = %s", (worker_id,))
    worker = cursor.fetchone()
    cursor.close()
    return worker


# ──────────────────────────────────────────────
# Incident Logging
# ──────────────────────────────────────────────

def log_incident(
    conn,
    worker_id: int,
    image_path: str,
    validated: dict,
    escalation: str,
    report: str,
) -> tuple[int, str]:

    parsed = validated["parsed"]
    now = datetime.now(timezone.utc)

    query = """
        INSERT INTO incidents (
            worker_id,
            image_path, severity, final_decision, compliance_score,
            detected_ppe, missing_ppe, confidence, osha_codes,
            escalation, override_applied, override_reason,
            report, created_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    values = (
        worker_id,
        image_path,
        validated["severity"],
        validated["final_decision"],
        validated["compliance_score"],
        ",".join(parsed.get("detected_ppe", [])),
        ",".join(parsed.get("missing_ppe", [])),
        parsed.get("confidence", 0.0),
        ",".join(v["code"] for v in validated.get("osha_violations", [])),
        escalation,
        bool(validated.get("override_applied")),
        validated.get("override_reason", ""),
        report,
        now,
    )

    cursor = conn.cursor()
    cursor.execute(query, values)
    conn.commit()

    incident_id = cursor.lastrowid
    cursor.close()

    logger.info(f"Incident logged: id={incident_id}, worker={worker_id}")
    return incident_id, now.isoformat()


# ──────────────────────────────────────────────
# Offense Tracking
# ──────────────────────────────────────────────

def count_worker_offenses(conn, worker_id: int) -> int:
    """
    Count non-CLEAR incidents for specific worker.
    """
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """
        SELECT COUNT(*) as cnt
        FROM incidents
        WHERE worker_id = %s
        AND severity != 'CLEAR'
        """,
        (worker_id,),
    )
    result = cursor.fetchone()
    cursor.close()
    return result["cnt"]


def count_total_offenses(conn) -> int:
    """
    Count total non-CLEAR incidents.
    """
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT COUNT(*) as cnt FROM incidents WHERE severity != 'CLEAR'"
    )
    result = cursor.fetchone()
    cursor.close()
    return result["cnt"]


# ──────────────────────────────────────────────
# Fetch Incidents
# ──────────────────────────────────────────────

def get_recent_incidents(conn, limit: int = 20) -> list[dict]:
    """
    Fetch recent incidents joined with worker info.
    """
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """
        SELECT i.*, w.employee_id, w.full_name, w.role, w.department
        FROM incidents i
        JOIN workers w ON i.worker_id = w.id
        ORDER BY i.id DESC
        LIMIT %s
        """,
        (limit,),
    )
    results = cursor.fetchall()
    cursor.close()
    return results