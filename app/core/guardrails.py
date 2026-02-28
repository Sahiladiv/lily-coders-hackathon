"""
Strict parser for fine-tuned Gemma output.
Any deviation from expected format triggers a retry.
No forgiveness. No fuzzy matching.
JSON schema validation
"""
import re
import logging

from app.core.state import ParsedDecision

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Expected output format from fine-tuned model:
# 
# DETECTED_PPE: helmet, vest, boots
# MISSING_PPE: gloves, goggles
# DECISION: DENY
# CONFIDENCE: 0.87
# REASON: missing hand and eye protection
# ──────────────────────────────────────────────

REQUIRED_FIELDS = {"DETECTED_PPE", "MISSING_PPE", "DECISION", "REASON"}
OPTIONAL_FIELDS = {"CONFIDENCE"}
ALL_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS

FIELD_PATTERN = re.compile(
    r"^(DETECTED_PPE|MISSING_PPE|DECISION|CONFIDENCE|REASON):\s*(.+)$",
    re.IGNORECASE,
)

VALID_DECISIONS = {"ALLOW", "DENY"}


class ParseError(ValueError):
    """Raised when LLM output doesn't match expected format."""
    pass


def parse_llm_output(raw: str) -> ParsedDecision:
    """
    Parse raw LLM text into a validated ParsedDecision.

    Raises ParseError on any format violation.
    Fine-tuned models should rarely trigger this.
    Base models may trigger it more often — that's what retries are for.
    """
    if not raw or not raw.strip():
        raise ParseError("Empty LLM response")

    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    fields: dict[str, str] = {}

    for line in lines:
        match = FIELD_PATTERN.match(line)
        if match:
            key = match.group(1).upper()
            value = match.group(2).strip()
            if key in ALL_FIELDS:
                fields[key] = value

    # Validate required fields present
    missing_keys = REQUIRED_FIELDS - set(fields.keys())
    if missing_keys:
        raise ParseError(
            f"Missing required fields: {missing_keys}. "
            f"Got fields: {list(fields.keys())}. "
            f"Raw output (first 300 chars): {raw[:300]}"
        )

    # Validate DECISION value
    decision = fields["DECISION"].upper().strip()
    if decision not in VALID_DECISIONS:
        raise ParseError(
            f"Invalid DECISION value: '{decision}'. Must be ALLOW or DENY."
        )

    # Parse confidence (optional, default 0.0)
    confidence = 0.0
    if "CONFIDENCE" in fields:
        try:
            confidence = float(fields["CONFIDENCE"])
        except ValueError:
            logger.warning(f"Could not parse CONFIDENCE: {fields['CONFIDENCE']}, defaulting to 0.0")
            confidence = 0.0

    # Build validated model
    try:
        parsed = ParsedDecision(
            detected_ppe=fields["DETECTED_PPE"],
            missing_ppe=fields["MISSING_PPE"],
            decision=decision,
            confidence=confidence,
            reason=fields["REASON"],
        )
    except Exception as e:
        raise ParseError(f"Pydantic validation failed: {e}")

    logger.info(
        f"Parsed successfully — detected: {parsed.detected_ppe}, "
        f"missing: {parsed.missing_ppe}, decision: {parsed.decision}, "
        f"confidence: {parsed.confidence}"
    )

    return parsed
