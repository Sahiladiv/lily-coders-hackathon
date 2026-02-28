"""
Workplace Safety Gap Detection System — Configuration
All thresholds, mappings, and settings in one place.
"""
from pathlib import Path


from pathlib import Path

MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "safety_monitoring",
}

# ──────────────────────────────────────────────
# LM Studio / Ollama Configuration
# ──────────────────────────────────────────────
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"

# ── Model 1: Fine-tuned Gemma (Vision) ──
# PPE detection only. Trained on structured output format.
# Do NOT use for general text tasks — it will underperform.
LM_STUDIO_VISION_MODEL = "workplace-safety-gemma-ft"

# ── Model 2: Base Gemma (Text) ──
# General-purpose text model for report rephrasing.
# Needs broad language ability, NOT the fine-tuned model.
# LM_STUDIO_TEXT_MODEL = "gemma-3-4b-it"
LM_STUDIO_TEXT_MODEL = "gemma-3-1b-it"  # fallback to base for rephrasing   
# LM_STUDIO_TEXT_MODEL = "gemma-3n-e2b-it"  # fallback to base for rephrasing

# ── Model 3: FunctionGemma (Function Calling) ──
# NL intent parsing → structured tool calls.
# Separate model optimized for function calling format.
FUNCTION_GEMMA_MODEL = "google_functiongemma-270m-it"

# Fine-tuned model flag — controls prompt weight
IS_FINETUNED = True

# ──────────────────────────────────────────────
# Inference Settings
# ──────────────────────────────────────────────
LLM_TEMPERATURE = 0.0          # deterministic for fine-tuned
LLM_MAX_TOKENS = 256           # structured output is short
LLM_TIMEOUT_SECONDS = 120.0    # CPU inference can be slow

# ──────────────────────────────────────────────
# Retry / Graph Bounds
# ──────────────────────────────────────────────
MAX_RETRIES = 3
MAX_GRAPH_ITERATIONS = 10      # LangGraph recursion_limit (3 retries + 3 nodes + buffer)

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
DB_PATH = Path("data/incidents.db")
IMAGE_DIR = Path("data/images")

# ──────────────────────────────────────────────
# PPE Configuration
# ──────────────────────────────────────────────
REQUIRED_PPE: set[str] = {"helmet", "vest", "gloves", "goggles", "boots"}
CRITICAL_PPE: set[str] = {"helmet", "goggles"}

# ──────────────────────────────────────────────
# Severity Rules
# ──────────────────────────────────────────────
# Evaluated in order — first match wins
SEVERITY_RULES = [
    ("CRITICAL", lambda missing_critical, missing_total: missing_critical >= 2),
    ("HIGH",     lambda missing_critical, missing_total: missing_critical >= 1),
    ("LOW",      lambda missing_critical, missing_total: missing_total >= 1),
    ("CLEAR",    lambda missing_critical, missing_total: True),
]

# ──────────────────────────────────────────────
# OSHA Reference Mapping
# ──────────────────────────────────────────────
OSHA_REFERENCES: dict[str, dict[str, str]] = {
    "helmet":  {"code": "29 CFR 1926.100", "desc": "Head Protection"},
    "vest":    {"code": "29 CFR 1926.201", "desc": "High-Visibility Apparel"},
    "gloves":  {"code": "29 CFR 1910.138", "desc": "Hand Protection"},
    "goggles": {"code": "29 CFR 1926.102", "desc": "Eye and Face Protection"},
    "boots":   {"code": "29 CFR 1910.136", "desc": "Foot Protection"},
}

# ──────────────────────────────────────────────
# Corrective Actions
# ──────────────────────────────────────────────
CORRECTIVE_ACTIONS: dict[str, str] = {
    "helmet":  "Provide and enforce hard hat usage immediately.",
    "vest":    "Issue high-visibility vest before site re-entry.",
    "gloves":  "Provide appropriate work gloves for current task.",
    "goggles": "Issue safety goggles; halt work near debris/chemicals.",
    "boots":   "Require steel-toe boots before resuming work.",
}

# ──────────────────────────────────────────────
# Escalation Thresholds
# ──────────────────────────────────────────────
ESCALATION_THRESHOLDS: dict[int, str] = {
    1: "LOG",
    2: "WARNING",
    3: "SUPERVISOR_ALERT",
}

# ──────────────────────────────────────────────
# Confidence Threshold (for fine-tuned model)
# ──────────────────────────────────────────────
LOW_CONFIDENCE_THRESHOLD = 0.5   # below this → auto-DENY + flag for review

# ──────────────────────────────────────────────
# FunctionGemma Settings
# ──────────────────────────────────────────────
FUNCTION_GEMMA_TEMPERATURE = 0.0
FUNCTION_GEMMA_MAX_TOKENS = 256
FUNCTION_GEMMA_TIMEOUT_SECONDS = 60.0  # smaller model, faster

# ──────────────────────────────────────────────
# Auto-Escalation Rules (triggered in Node 3)
# ──────────────────────────────────────────────
# SUPERVISOR_ALERT → alert_supervisor + send_email + create_ticket
# WARNING          → send_email + create_ticket
# LOG + HIGH/CRIT  → create_ticket only
# LOG + LOW/CLEAR  → no action (just logged)
AUTO_ESCALATE_ON_SEVERITY = {"CRITICAL", "HIGH"}  # even on 1st offense, create ticket

# ──────────────────────────────────────────────
# FastAPI
# ──────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
