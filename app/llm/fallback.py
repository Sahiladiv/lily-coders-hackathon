# """
# OpenAI Vision Fallback
# Used when local PPE model fails or confidence is too low.
# Returns structured safety decision JSON.
# """

# import base64
# import json
# import os
# from typing import Dict, Any
# from app.config import (
#     LM_STUDIO_BASE_URL,
#     LM_STUDIO_VISION_MODEL,
#     IS_FINETUNED,
#     LLM_TEMPERATURE,
#     LLM_MAX_TOKENS,
#     LLM_TIMEOUT_SECONDS,
# )

# import logging
# from openai import OpenAI
# from PIL import Image


# logger = logging.getLogger(__name__)
# # ---------------------------------------------------
# # Configuration
# # ---------------------------------------------------

# OPENAI_MODEL = "gpt-4o-mini"   # Fast + cheap + vision capable
# CONFIDENCE_THRESHOLD = 0.60
# api_key = "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"
# # api_key = os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=api_key)


# # ---------------------------------------------------
# # Helper: Convert Image to Base64
# # ---------------------------------------------------

# def encode_image(image_path: str) -> str:
#     with open(image_path, "rb") as f:
#         return base64.b64encode(f.read()).decode("utf-8")


# # ---------------------------------------------------
# # PPE Analysis Prompt
# # ---------------------------------------------------

# SYSTEM_PROMPT = """
# You are a workplace safety compliance inspector.

# Your task:
# Analyze the provided image and determine if Personal Protective Equipment (PPE) is properly worn.

# You must output STRICT JSON only.

# Required JSON format:

# {
#   "detected_ppe": ["Helmet", "Safety Vest"],
#   "missing_ppe": ["Gloves"],
#   "compliance_score": 85,
#   "severity": "LOW | HIGH | CLEAR",
#   "decision": "APPROVED | DENY",
#   "confidence": 0.92,
#   "reason": "Short explanation"
# }

# Rules:
# - If no violations → severity = CLEAR, decision = APPROVED
# - If major PPE missing (helmet) → severity = HIGH, decision = DENY
# - Compliance score between 0-100
# - Confidence between 0-1
# - Return ONLY JSON. No explanation outside JSON.
# """
# import re

# def extract_json(text: str):
#     """
#     Extracts JSON object from model response.
#     Handles markdown and extra text.
#     """
#     try:
#         # First try direct parse
#         return json.loads(text)
#     except:
#         pass

#     # Remove markdown code blocks
#     text = re.sub(r"```json", "", text)
#     text = re.sub(r"```", "", text)

#     # Try to find first JSON object
#     match = re.search(r"\{.*\}", text, re.DOTALL)
#     if match:
#         json_str = match.group(0)
#         return json.loads(json_str)

#     raise ValueError("No valid JSON found in response")

# # ---------------------------------------------------
# # Fallback Function
# # ---------------------------------------------------

# USER_PROMPT = "Analyze this workplace image for PPE compliance."


# def analyze_ppe_with_openai(image_path: str) -> str:
#     """
#     Send image to OpenAI multimodal endpoint.
#     Returns raw text response from the model.
#     Raises on HTTP or timeout errors — caller handles retry.
#     """

#     base64_image, mime = encode_image(image_path)

#     response = client.chat.completions.create(
#         model=OPENAI_MODEL,
#         temperature=0.2,
#         messages=[
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": f"data:{mime};base64,{base64_image}"
#                         },
#                     },
#                     {
#                         "type": "text",
#                         "text": USER_PROMPT,
#                     },
#                 ],
#             },
#         ],
#         max_tokens=LLM_MAX_TOKENS,
#     )

#     raw_output = response.choices[0].message.content

#     if not raw_output:
#         raise ValueError("Empty response from OpenAI")

#     logger.info(f"Raw OpenAI output:\n{raw_output}")

#     return raw_output


# # ---------------------------------------------------
# # Example Standalone Test
# # ---------------------------------------------------

# if __name__ == "__main__":
#     image_path = "test.jpeg"

#     result = analyze_ppe_with_openai(image_path)

#     print(json.dumps(result, indent=2))
    
#     # python -m app.llm.fallback



"""
OpenAI Vision Fallback
Used when local PPE model fails or confidence is too low.
Returns RAW LLM output (JSON string).
"""

import base64
import json
import os
import re
import mimetypes
import logging
from pathlib import Path
from typing import Dict, Any

from openai import OpenAI

from app.config import LLM_MAX_TOKENS

logger = logging.getLogger(__name__)

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------

OPENAI_MODEL = "gpt-4o-mini"
CONFIDENCE_THRESHOLD = 0.60
api_key = "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"

# api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set in environment")

client = OpenAI(api_key=api_key)

# ---------------------------------------------------
# Image Encoder (FIXED)
# ---------------------------------------------------

def encode_image(image_path: str):
    """
    Encode image to base64 and detect MIME type.
    Always returns (b64_string, mime_type)
    """

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(path, "rb") as f:
        image_bytes = f.read()

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/jpeg"

    return b64, mime

# ---------------------------------------------------
# PPE Analysis Prompt
# ---------------------------------------------------

SYSTEM_PROMPT = """
You are a workplace safety compliance inspector.

Analyze the image and determine PPE compliance.

Return STRICT JSON only.

{
  "detected_ppe": ["Helmet", "Safety Vest"],
  "missing_ppe": ["Gloves"],
  "compliance_score": 85,
  "severity": "LOW | HIGH | CLEAR",
  "decision": "APPROVED | DENY",
  "confidence": 0.92,
  "reason": "Short explanation"
}

Rules:
- If no violations → severity = CLEAR, decision = APPROVED
- If helmet missing → severity = HIGH, decision = DENY
- Compliance score 0–100
- Confidence 0–1
- Return ONLY JSON
"""

USER_PROMPT = "Analyze this workplace image for PPE compliance."

# ---------------------------------------------------
# JSON Extractor (Safe)
# ---------------------------------------------------

def extract_json(text: str) -> Dict[str, Any]:
    """
    Extract JSON object from model response.
    Handles markdown blocks or extra text.
    """

    try:
        return json.loads(text)
    except:
        pass

    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError("No valid JSON found in model response")

# ---------------------------------------------------
# Fallback Inference
# ---------------------------------------------------

def analyze_ppe_with_openai(image_path: str) -> str:
    """
    Send image to OpenAI multimodal endpoint.
    Returns RAW LLM response text.
    Parsing happens later in pipeline.
    """

    b64, mime = encode_image(image_path)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        max_tokens=LLM_MAX_TOKENS,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": USER_PROMPT,
                    },
                ],
            },
        ],
    )

    raw_output = response.choices[0].message.content

    if not raw_output:
        raise ValueError("Empty response from OpenAI")

    logger.info(f"Raw OpenAI output:\n{raw_output}")

    return raw_output

# ---------------------------------------------------
# Standalone Test
# ---------------------------------------------------

if __name__ == "__main__":
    image_path = "test.jpeg"

    raw = analyze_ppe_with_openai(image_path)
    parsed = extract_json(raw)

    print("\nRAW OUTPUT:\n")
    print(raw)

    print("\nPARSED JSON:\n")
    print(json.dumps(parsed, indent=2))