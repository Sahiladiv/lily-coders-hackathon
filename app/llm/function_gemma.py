"""
FunctionGemma Client — Natural Language → Structured Tool Calls.

ONLY handles read-only user queries:
  - get_incidents  (list recent incidents)
  - get_report     (get specific incident details)
  - get_stats      (aggregate safety dashboard)

analyze_image and escalate are AUTOMATED — not routed through here.
"""
import json
import logging

import httpx
from pydantic import BaseModel, Field, field_validator

from app.config import (
    LM_STUDIO_BASE_URL,
    FUNCTION_GEMMA_MODEL,
    FUNCTION_GEMMA_TEMPERATURE,
    FUNCTION_GEMMA_MAX_TOKENS,
    FUNCTION_GEMMA_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Tool Definitions (3 read-only tools)
# ──────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_incidents",
            "description": (
                "Retrieve recent safety incidents from the database. "
                "Use when the user asks to see history, past violations, "
                "recent incidents, incident logs, or what happened recently."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max number of incidents to return (default 10)",
                        "default": 10,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_report",
            "description": (
                "Get the full incident report for a specific incident by its ID number. "
                "Use when the user asks for details about a specific incident, "
                "wants to see a particular report, or references an incident number."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "incident_id": {
                        "type": "integer",
                        "description": "The incident ID number to retrieve",
                    },
                },
                "required": ["incident_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stats",
            "description": (
                "Get summary statistics and overview of all safety incidents. "
                "Use when the user asks for a summary, dashboard, overview, "
                "aggregate data, trends, or general safety status."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


# ──────────────────────────────────────────────
# Parsed Tool Call Model
# ──────────────────────────────────────────────

VALID_TOOL_NAMES = {"get_incidents", "get_report", "get_stats"}


class ToolCall(BaseModel):
    """Validated tool call from FunctionGemma output."""

    tool_name: str
    arguments: dict = Field(default_factory=dict)
    raw_response: str = ""

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, v):
        if v not in VALID_TOOL_NAMES:
            raise ValueError(f"Unknown tool: {v}. Valid: {VALID_TOOL_NAMES}")
        return v


class FunctionGemmaError(Exception):
    """Raised when FunctionGemma fails to produce a valid tool call."""
    pass


# ──────────────────────────────────────────────
# System Prompt
# ──────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a workplace safety query assistant. "
    "You help users retrieve safety incident data. "
    "Route every request to exactly one tool. "
    "You can only retrieve data — you cannot analyze images or trigger actions. "
    "You MUST choose the most specific matching tool."

    "If the user asks for:"
    "- 'history', 'recent incidents', 'logs' → use get_incidents."
    "- 'report', 'details', or mentions an incident ID → use get_report."
    "- 'summary', 'overview', 'dashboard', 'statistics' → use get_stats."

    "Never guess. Choose carefully."
)


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────

def parse_tool_call(user_query: str) -> ToolCall:
    """
    Send user NL query to FunctionGemma.
    Returns a validated ToolCall (read-only tools only).
    Raises FunctionGemmaError on failure.
    """
    payload = {
        "model": FUNCTION_GEMMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ],
        "tools": TOOL_DEFINITIONS,
        "tool_choice": "auto",
        "temperature": FUNCTION_GEMMA_TEMPERATURE,
        "max_tokens": FUNCTION_GEMMA_MAX_TOKENS,
        "stream": False,
    }

    logger.info(f"[FunctionGemma] Query: {user_query[:100]}")

    try:
        resp = httpx.post(
            f"{LM_STUDIO_BASE_URL}/chat/completions",
            json=payload,
            timeout=FUNCTION_GEMMA_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()

    except httpx.HTTPError as e:
        raise FunctionGemmaError(f"HTTP error calling FunctionGemma: {e}")

    # Extract tool call from response
    message = data["choices"][0]["message"]
    raw = json.dumps(message, indent=2)

    # Check for tool_calls in response
    tool_calls = message.get("tool_calls")
    if not tool_calls or len(tool_calls) == 0:
        content = message.get("content", "")
        logger.warning(
            f"[FunctionGemma] No tool call returned. Content: {content[:200]}. "
            f"Defaulting to get_stats."
        )
        return ToolCall(tool_name="get_stats", arguments={}, raw_response=raw)

    # Take first tool call only
    tc = tool_calls[0]
    function_data = tc.get("function", {})
    tool_name = function_data.get("name", "")
    arguments_raw = function_data.get("arguments", "{}")

    # Parse arguments
    try:
        if isinstance(arguments_raw, str):
            arguments = json.loads(arguments_raw)
        else:
            arguments = arguments_raw
    except json.JSONDecodeError:
        logger.warning(f"[FunctionGemma] Bad arguments JSON: {arguments_raw}")
        arguments = {}

    # Validate — reject if FunctionGemma hallucinates a tool outside our set
    try:
        tool_call = ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            raw_response=raw,
        )
    except ValueError as e:
        # Tool name not in valid set — default to get_stats
        logger.warning(f"[FunctionGemma] Invalid tool '{tool_name}', defaulting to get_stats: {e}")
        return ToolCall(tool_name="get_stats", arguments={}, raw_response=raw)

    logger.info(f"[FunctionGemma] Resolved: {tool_call.tool_name}({tool_call.arguments})")
    return tool_call
