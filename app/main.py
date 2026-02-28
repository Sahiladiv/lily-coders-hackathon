"""
Workplace Safety Gap Detection System — FastAPI Entry Point.

AUTOMATED (trigger on event):
    POST /analyze        — Image upload → full pipeline → auto-escalation

USER QUERIES (FunctionGemma routed):
    POST /chat           — NL query → get_incidents / get_report / get_stats

DIRECT (bypass FunctionGemma):
    GET  /incidents      — List recent incidents
    GET  /incidents/{id} — Get single incident
    GET  /stats          — Aggregate stats
    GET  /health         — Health check
"""
import logging
import shutil
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from app.core.agent import build_graph
from app.core.router import dispatch, _handle_get_stats
from app.llm.function_gemma import parse_tool_call, FunctionGemmaError
# from app.storage.sqlite import get_connection, get_recent_incidents
from app.storage.mysql import get_connection, get_recent_incidents
from app.config import API_HOST, API_PORT, IMAGE_DIR, MAX_GRAPH_ITERATIONS

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────
app = FastAPI(
    title="Workplace Safety Gap Detection",
    description=(
        "Edge-native PPE compliance system. "
        "Fine-tuned Gemma (vision), FunctionGemma (NL queries), "
        "auto-escalation on violations."
    ),
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build LangGraph once at startup
graph = build_graph()


# ──────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    query: str
    tool_called: str
    arguments: dict
    result: dict


class AnalyzeResponse(BaseModel):
    status: str
    incident_id: int
    severity: str
    decision: str
    compliance_score: int
    escalation: str
    escalation_actions: list[dict]
    override_applied: bool
    missing_ppe: list[str]
    detected_ppe: list[str]
    confidence: float
    report: str


# ══════════════════════════════════════════════
# AUTOMATED: Image Upload → Pipeline → Auto-Escalation
# ══════════════════════════════════════════════
@app.post("/analyze-image", response_model=AnalyzeResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    Upload image → triggers full pipeline automatically.
    Escalation happens inside Node 3 based on severity + offense count.
    No user interaction needed.
    """
    # Validate
    allowed = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {allowed}",
        )

    # Save
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    dest = IMAGE_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    logger.info(f"Image saved: {dest}")

    # Run pipeline
    initial_state = {
        "image_path": str(dest),
        "retry_count": 0,
        "status": "analyzing",
    }

    try:
        result = graph.invoke(
            initial_state,
            config={"recursion_limit": MAX_GRAPH_ITERATIONS},
        )
    except Exception as e:
        logger.error(f"Graph execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    if result.get("status") == "failed":
        raise HTTPException(
            status_code=422,
            detail={
                "status": "failed",
                "error": result.get("error", "Unknown error"),
                "retry_count": result.get("retry_count", 0),
            },
        )

    act = result["act_output"]
    validated = act["validated"]
    parsed = validated["parsed"]

    return AnalyzeResponse(
        status=result["status"],
        incident_id=act["incident_id"],
        severity=validated["severity"],
        decision=validated["final_decision"],
        compliance_score=validated["compliance_score"],
        escalation=act["escalation"],
        escalation_actions=act.get("escalation_actions", []),
        override_applied=validated["override_applied"],
        missing_ppe=parsed["missing_ppe"],
        detected_ppe=parsed["detected_ppe"],
        confidence=parsed["confidence"],
        report=act["report"],
    )


# ══════════════════════════════════════════════
# USER QUERIES: NL → FunctionGemma → Read-Only Data
# ══════════════════════════════════════════════
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Natural language query interface (read-only).

    FunctionGemma routes to: get_incidents, get_report, get_stats.

    Examples:
        "Show me recent violations"         → get_incidents
        "What happened in incident 5?"      → get_report
        "Give me a safety overview"         → get_stats
        "How many critical incidents?"      → get_stats
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")

    # FunctionGemma resolves intent
    try:
        tool_call = parse_tool_call(query)
    except FunctionGemmaError as e:
        logger.error(f"[Chat] FunctionGemma failed: {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Could not parse query: {e}",
        )

    # Dispatch to read-only handler
    result = dispatch(tool_call)

    logger.info(
        f"[Chat] '{query}' -> {tool_call.tool_name}({tool_call.arguments}) -> "
        f"success={result.get('success')}"
    )

    return ChatResponse(
        query=query,
        tool_called=tool_call.tool_name,
        arguments=tool_call.arguments,
        result=result,
    )


# ══════════════════════════════════════════════
# DIRECT ENDPOINTS (bypass FunctionGemma)
# ══════════════════════════════════════════════
@app.get("/incidents")
async def list_incidents(limit: int = 20):
    """List recent incidents (direct, no FunctionGemma)."""
    conn = get_connection()
    try:
        incidents = get_recent_incidents(conn, limit=limit)
        return {"incidents": incidents, "count": len(incidents)}
    finally:
        conn.close()


@app.get("/incidents/{incident_id}")
async def get_incident(incident_id: int):
    """Get a single incident by ID (direct, no FunctionGemma)."""
    conn = get_connection()
    try:
        cur = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Incident not found")

        incident = dict(row)

        # Include escalation actions
        try:
            esc_cur = conn.execute(
                "SELECT * FROM escalation_log WHERE incident_id = ? ORDER BY id",
                (incident_id,),
            )
            incident["escalation_actions"] = [dict(r) for r in esc_cur.fetchall()]
        except Exception:
            incident["escalation_actions"] = []

        return incident
    finally:
        conn.close()


@app.get("/stats")
async def stats():
    """Aggregate safety statistics (direct, no FunctionGemma)."""
    return _handle_get_stats({})


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "service": "workplace-safety-gap-detection",
        "version": "2.1.0",
        "models": {
            "vision": "fine-tuned-gemma (auto: image analysis)",
            "text": "base-gemma (auto: report rephrasing)",
            "function_calling": "function-gemma (user queries only)",
        },
        "flows": {
            "automated": ["analyze_image", "escalation"],
            "user_query": ["get_incidents", "get_report", "get_stats"],
        },
    }


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logger.info(f"Starting server on {API_HOST}:{API_PORT}")
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
    )
