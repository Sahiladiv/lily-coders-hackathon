"""
Gemma Multimodal Client — LM Studio / OpenAI-compatible endpoint.

Handles:
- Image encoding
- Prompt selection (fine-tuned vs base)
- HTTP call with timeout
- Raw text extraction

This is the ONLY place LLM inference happens for vision tasks.
"""
import base64
import logging
from pathlib import Path

import httpx

from app.config import (
    LM_STUDIO_BASE_URL,
    LM_STUDIO_VISION_MODEL,
    IS_FINETUNED,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────

# Fine-tuned model already knows the task and format.
# Heavy prompting can conflict with fine-tuned behavior.
SYSTEM_PROMPT_FINETUNED = (
    "Analyze the workplace image for PPE compliance. "
    "Respond in the trained output format."
)

# Base model needs full instructions.
SYSTEM_PROMPT_BASE = """You are a workplace safety PPE detector. Analyze the image.
Respond in EXACTLY this format with no other text:

DETECTED_PPE: item1, item2
MISSING_PPE: item1, item2
DECISION: ALLOW or DENY
CONFIDENCE: 0.0 to 1.0
REASON: one short sentence

Rules:
- Only reference these items: helmet, vest, gloves, goggles, boots
- If any critical PPE (helmet, goggles) is missing, DECISION must be DENY
- CONFIDENCE is your certainty from 0.0 to 1.0
- No markdown, no JSON, no extra commentary, no explanation
- Output ONLY the five lines above"""

USER_PROMPT = "Analyze this workplace image for PPE compliance."

# ──────────────────────────────────────────────
# Image Encoding
# ──────────────────────────────────────────────

MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def encode_image(image_path: str) -> tuple[str, str]:
    """Read image file, return (base64_string, mime_type)."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime = MIME_MAP.get(path.suffix.lower(), "image/jpeg")
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")

    logger.info(f"Encoded image: {path.name} ({len(data)} bytes, {mime})")
    return b64, mime


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────

def analyze_image(image_path: str) -> str:
    """
    Send image to LM Studio multimodal endpoint.
    Returns raw text response from the model.
    Raises on HTTP or timeout errors — caller handles retry.
    """
    b64, mime = encode_image(image_path)
    system_prompt = SYSTEM_PROMPT_FINETUNED if IS_FINETUNED else SYSTEM_PROMPT_BASE

    payload = {
        "model": LM_STUDIO_VISION_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": USER_PROMPT,
                    },
                ],
            },
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
        "stream": False,
    }

    logger.info(
        f"Calling LM Studio: model={LM_STUDIO_VISION_MODEL}, "
        f"finetuned={IS_FINETUNED}, temp={LLM_TEMPERATURE}"
    )

    resp = httpx.post(
        f"{LM_STUDIO_BASE_URL}/chat/completions",
        json=payload,
        timeout=LLM_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()

    raw_output = resp.json()["choices"][0]["message"]["content"]
    logger.info(f"Raw LLM output:\n{raw_output}")

    return raw_output
