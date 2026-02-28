"""
Text LLM Client — optional report rephrasing.

Used ONLY in Node 3 (ACT) to make reports more readable.
Falls back to Python template if LLM fails.
LLM cannot alter severity, decision, or any logic.
"""
import logging

import httpx

from app.config import (
    LM_STUDIO_BASE_URL,
    LM_STUDIO_TEXT_MODEL,
    LLM_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


def rephrase_report(template_report: str) -> str:
    """
    Ask text LLM to rephrase the template report for readability.
    Returns original template on any failure.
    """
    try:
        payload = {
            "model": LM_STUDIO_TEXT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a workplace safety report writer. "
                        "Rephrase the following incident report for clarity. "
                        "Do NOT change any facts, numbers, severity levels, "
                        "decisions, OSHA codes, or escalation levels. "
                        "Keep it concise and professional."
                    ),
                },
                {
                    "role": "user",
                    "content": template_report,
                },
            ],
            "temperature": 0.2,
            "max_tokens": 512,
            "stream": False,
        }

        resp = httpx.post(
            f"{LM_STUDIO_BASE_URL}/chat/completions",
            json=payload,
            timeout=LLM_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()

        rephrased = resp.json()["choices"][0]["message"]["content"]
        logger.info("Report rephrased by LLM successfully")
        return rephrased

    except Exception as e:
        logger.warning(f"Report rephrasing failed ({e}), using template")
        return template_report
