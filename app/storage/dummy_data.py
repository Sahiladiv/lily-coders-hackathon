"""
Populate MySQL database with realistic dummy safety data.
"""

import random
from datetime import datetime, timedelta, timezone

from app.storage.mysql import (
    get_connection,
    get_or_create_worker,
    log_incident,
)

# ──────────────────────────────────────────────
# Configurable
# ──────────────────────────────────────────────

NUM_WORKERS = 15
NUM_INCIDENTS = 150

DEPARTMENTS = ["Cold Storage", "Packaging", "Logistics", "Maintenance"]
ROLES = ["Technician", "Supervisor", "Operator", "Loader"]

PPE_ITEMS = ["Helmet", "Gloves", "Mask", "Safety Vest", "Goggles"]

SEVERITY_LEVELS = ["CLEAR", "LOW", "MEDIUM", "HIGH"]


# ──────────────────────────────────────────────
# Fake Validation Generator
# ──────────────────────────────────────────────

def generate_fake_validation(severity: str):
    detected = random.sample(PPE_ITEMS, random.randint(2, 5))
    missing = list(set(PPE_ITEMS) - set(detected))

    return {
        "severity": severity,
        "final_decision": "APPROVED" if severity == "CLEAR" else "FLAGGED",
        "compliance_score": random.randint(70, 100) if severity == "CLEAR" else random.randint(30, 75),
        "parsed": {
            "detected_ppe": detected,
            "missing_ppe": missing,
            "confidence": round(random.uniform(0.6, 0.99), 2),
        },
        "osha_violations": [
            {"code": f"OSHA-{random.randint(100, 999)}"}
        ] if severity in ["MEDIUM", "HIGH"] else [],
        "override_applied": False,
        "override_reason": "",
    }


# ──────────────────────────────────────────────
# Escalation Logic
# ──────────────────────────────────────────────

def determine_escalation(severity: str, worker_offense_count: int) -> str:
    if severity == "CLEAR":
        return "NONE"

    if worker_offense_count >= 5:
        return "SUSPENSION"

    if worker_offense_count >= 3:
        return "SUPERVISOR_REVIEW"

    return "WARNING"


# ──────────────────────────────────────────────
# Main Seeder
# ──────────────────────────────────────────────

def seed():
    conn = get_connection()

    print("Seeding workers...")

    worker_ids = []

    for i in range(NUM_WORKERS):
        worker_id = get_or_create_worker(
            conn,
            employee_id=f"EMP-{1000+i}",
            full_name=f"Worker {i+1}",
            role=random.choice(ROLES),
            department=random.choice(DEPARTMENTS),
        )
        worker_ids.append(worker_id)

    print("Workers created.")

    print("Seeding incidents...")

    # Make some workers repeat offenders
    repeat_offenders = random.sample(worker_ids, k=5)

    for i in range(NUM_INCIDENTS):

        worker_id = random.choice(worker_ids)

        # Increase violation probability for repeat offenders
        if worker_id in repeat_offenders:
            severity = random.choices(
                SEVERITY_LEVELS,
                weights=[0.2, 0.3, 0.3, 0.2]
            )[0]
        else:
            severity = random.choices(
                SEVERITY_LEVELS,
                weights=[0.5, 0.3, 0.15, 0.05]
            )[0]

        validated = generate_fake_validation(severity)

        # Count offenses so far for escalation realism
        from app.storage.mysql import count_worker_offenses
        offense_count = count_worker_offenses(conn, worker_id)

        escalation = determine_escalation(severity, offense_count)

        report = f"Auto-generated safety report. Severity: {severity}"

        log_incident(
            conn,
            worker_id=worker_id,
            image_path=f"/images/camera_{random.randint(1,5)}.jpg",
            validated=validated,
            escalation=escalation,
            report=report,
        )

    print("Incidents created.")
    print("Database seeded successfully.")


if __name__ == "__main__":
    seed()