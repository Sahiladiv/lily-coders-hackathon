"""
PPE Compliance Training Data Generator for PaLIGemma Fine-Tuning
================================================================
Converts Roboflow PPE annotations into deterministic (prefix, suffix)
training samples for google/paligemma-3b-pt-224.

Usage:
    # From Roboflow COCO JSON export:
    python ppe_training_generator.py --coco _annotations.coco.json --output train.jsonl

    # From manual labels:
    python ppe_training_generator.py --detected "helmet,safety_vest" --missing "gloves,safety_goggles"

    # From existing Roboflow PaLIGemma export dir:
    python ppe_training_generator.py --roboflow-dir ./dataset --split train --output train.jsonl

    # Batch from a simple CSV (image,label1,label2,...):
    python ppe_training_generator.py --csv annotations.csv --output train.jsonl
"""

import json
import argparse
import csv
import sys
import os
from typing import List, Literal
from pydantic import BaseModel, field_validator, model_validator


# ============================================================
# CONSTANTS — Single source of truth
# ============================================================

REQUIRED_PPE = frozenset(["helmet", "safety_vest", "gloves", "safety_goggles"])

# Map common Roboflow class names to canonical labels
LABEL_ALIASES = {
    "hard-hat": "helmet", "hard_hat": "helmet", "hardhat": "helmet",
    "head": "helmet", "helmet": "helmet",
    "vest": "safety_vest", "safety-vest": "safety_vest",
    "safety_vest": "safety_vest", "high-visibility-vest": "safety_vest",
    "hi-vis": "safety_vest",
    "gloves": "gloves", "glove": "gloves", "hand-gloves": "gloves",
    "goggles": "safety_goggles", "safety-goggles": "safety_goggles",
    "safety_goggles": "safety_goggles", "glasses": "safety_goggles",
    "safety-glasses": "safety_goggles", "eye-protection": "safety_goggles",
    "mask": "safety_goggles",
    "boots": "safety_boots", "safety-boots": "safety_boots",
    "safety_boots": "safety_boots",
    "ear-protection": "ear_protection", "earmuffs": "ear_protection",
    "face-shield": "face_shield", "face_shield": "face_shield",
    # Negative labels from some Roboflow datasets
    "no-helmet": "__no_helmet", "no-vest": "__no_vest",
    "no-gloves": "__no_gloves", "no-goggles": "__no_goggles",
    "no-hardhat": "__no_helmet", "no-safety-vest": "__no_vest",
}

NEGATIVE_LABEL_MAP = {
    "__no_helmet": "helmet",
    "__no_vest": "safety_vest",
    "__no_gloves": "gloves",
    "__no_goggles": "safety_goggles",
}

STANDARDIZED_PREFIX = (
    "You are a workplace safety compliance system.\n"
    "Analyze the image and determine PPE compliance.\n"
    "Return output in the following STRICT JSON format:\n"
    "{\n"
    '  "detected_ppe": [list of detected PPE items],\n'
    '  "missing_ppe": [list of required but missing PPE items],\n'
    '  "violation": true or false,\n'
    '  "decision": "ALLOW" or "DENY",\n'
    '  "reason": "short explanation"\n'
    "}\n"
    "Only return valid JSON. Do not explain outside JSON."
)


# ============================================================
# PYDANTIC MODELS — Strict validation on ALL outputs
# ============================================================

class PPEAnnotationInput(BaseModel):
    """Validates raw annotation input before processing."""
    image: str = ""
    raw_labels: List[str]

    @field_validator("raw_labels", mode="before")
    @classmethod
    def parse_labels(cls, v):
        if isinstance(v, str):
            return [x.strip().lower() for x in v.split(",") if x.strip()]
        return [x.strip().lower() for x in v if x.strip()]


class PPEComplianceOutput(BaseModel):
    """
    Strict output schema — the contract the fine-tuned model must learn.
    Every training suffix MUST pass this validation.
    """
    detected_ppe: List[str]
    missing_ppe: List[str]
    violation: bool
    decision: Literal["ALLOW", "DENY"]
    reason: str

    @model_validator(mode="after")
    def enforce_deterministic_logic(self):
        has_missing = len(self.missing_ppe) > 0
        if has_missing and not self.violation:
            raise ValueError(f"missing_ppe={self.missing_ppe} but violation=False")
        if not has_missing and self.violation:
            raise ValueError("missing_ppe is empty but violation=True")
        if self.violation and self.decision != "DENY":
            raise ValueError(f"violation=True but decision={self.decision}")
        if not self.violation and self.decision != "ALLOW":
            raise ValueError(f"violation=False but decision={self.decision}")
        return self

    @field_validator("reason")
    @classmethod
    def reason_must_be_short(cls, v):
        if len(v) > 200:
            raise ValueError("Reason must be under 200 characters")
        if not v.strip():
            raise ValueError("Reason cannot be empty")
        return v.strip()


class TrainingSample(BaseModel):
    """Final JSONL row consumed by the training script."""
    image: str
    prefix: str
    suffix: str


# ============================================================
# CORE LOGIC — Deterministic, no LLM needed
# ============================================================

def normalize_labels(raw_labels: List[str]) -> tuple:
    """
    Normalize Roboflow labels to canonical PPE names.
    Returns: (detected_set, confirmed_missing_set)
    """
    detected = set()
    confirmed_missing = set()

    for label in raw_labels:
        label = label.strip().lower().replace(" ", "-")
        canonical = LABEL_ALIASES.get(label)

        if canonical is None:
            print(f"  [WARN] Unknown label '{label}' — skipping", file=sys.stderr)
            continue

        if canonical in NEGATIVE_LABEL_MAP:
            confirmed_missing.add(NEGATIVE_LABEL_MAP[canonical])
        else:
            detected.add(canonical)

    # Detected overrides confirmed_missing (if both present, trust detection)
    confirmed_missing -= detected
    return detected, confirmed_missing


def generate_compliance_output(detected: set, confirmed_missing: set) -> PPEComplianceOutput:
    """Deterministic compliance logic — pure rule engine."""
    missing = (REQUIRED_PPE - detected) | (confirmed_missing & REQUIRED_PPE)

    violation = len(missing) > 0
    decision = "DENY" if violation else "ALLOW"

    if not violation:
        reason = "All required PPE detected. Access granted."
    elif len(missing) == len(REQUIRED_PPE):
        reason = "No required PPE detected. Full violation."
    else:
        reason = f"Missing required PPE: {', '.join(sorted(missing))}."

    return PPEComplianceOutput(
        detected_ppe=sorted(detected),
        missing_ppe=sorted(missing),
        violation=violation,
        decision=decision,
        reason=reason,
    )


def build_training_sample(image_filename: str, raw_labels: List[str]) -> TrainingSample:
    """Full pipeline: raw labels -> validated training sample."""
    annotation = PPEAnnotationInput(image=image_filename, raw_labels=raw_labels)
    detected, confirmed_missing = normalize_labels(annotation.raw_labels)
    compliance = generate_compliance_output(detected, confirmed_missing)

    suffix_json = compliance.model_dump_json()
    # Round-trip validation — ensures the suffix we emit is parseable
    PPEComplianceOutput.model_validate_json(suffix_json)

    return TrainingSample(
        image=image_filename,
        prefix=STANDARDIZED_PREFIX,
        suffix=suffix_json,
    )


# ============================================================
# PARSERS — Multiple input format support
# ============================================================

def parse_coco_annotations(coco_path: str) -> List[TrainingSample]:
    """Parse Roboflow COCO JSON export."""
    with open(coco_path, "r") as f:
        coco = json.load(f)

    cat_map = {cat["id"]: cat["name"] for cat in coco["categories"]}

    image_annotations = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        image_annotations.setdefault(img_id, []).append(cat_map[ann["category_id"]])

    image_map = {img["id"]: img["file_name"] for img in coco["images"]}

    samples, errors = [], 0
    for img_id, filename in image_map.items():
        labels = image_annotations.get(img_id, [])
        try:
            samples.append(build_training_sample(filename, labels))
        except Exception as e:
            errors += 1
            print(f"  [ERROR] {filename}: {e}", file=sys.stderr)

    print(f"Parsed {len(samples)} samples, {errors} errors from {coco_path}")
    return samples


def parse_csv_annotations(csv_path: str) -> List[TrainingSample]:
    """Parse CSV: image_filename, label1, label2, ..."""
    samples, errors = [], 0
    with open(csv_path, "r") as f:
        for row in csv.reader(f):
            if len(row) < 1:
                continue
            filename = row[0].strip()
            labels = [x.strip() for x in row[1:] if x.strip()]
            try:
                samples.append(build_training_sample(filename, labels))
            except Exception as e:
                errors += 1
                print(f"  [ERROR] {filename}: {e}", file=sys.stderr)

    print(f"Parsed {len(samples)} samples, {errors} errors from {csv_path}")
    return samples


def extract_labels_from_suffix(suffix: str) -> List[str]:
    """Extract PPE labels from various suffix formats."""
    suffix = suffix.strip()
    if not suffix:
        return []

    # Try JSON first
    try:
        data = json.loads(suffix)
        if isinstance(data, dict):
            return list(data.get("detected_ppe", []))
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    # Text parsing fallback
    for sep in [";", ",", "\n"]:
        if sep in suffix:
            return [x.strip() for x in suffix.split(sep) if x.strip()]
    return [x.strip() for x in suffix.split() if x.strip()]


def parse_roboflow_paligemma_dir(root_dir: str, split: str) -> List[TrainingSample]:
    """Parse existing Roboflow PaLIGemma JSONL, re-generate with strict compliance outputs."""
    jsonl_path = os.path.join(root_dir, f"_annotations.{split}.jsonl")
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Not found: {jsonl_path}")

    samples, errors, skipped = [], 0, 0
    with open(jsonl_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            entry = json.loads(line.strip())
            image_file = entry.get("image", "")
            old_suffix = entry.get("suffix", "")
            labels = extract_labels_from_suffix(old_suffix)

            if not labels:
                skipped += 1
                print(f"  [SKIP] Line {line_num}: no labels from '{old_suffix[:60]}'", file=sys.stderr)
                continue
            try:
                samples.append(build_training_sample(image_file, labels))
            except Exception as e:
                errors += 1
                print(f"  [ERROR] Line {line_num} ({image_file}): {e}", file=sys.stderr)

    print(f"Parsed {len(samples)} samples, {errors} errors, {skipped} skipped from {jsonl_path}")
    return samples


# ============================================================
# OUTPUT
# ============================================================

def write_jsonl(samples: List[TrainingSample], output_path: str):
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(sample.model_dump_json() + "\n")
    print(f"Wrote {len(samples)} samples to {output_path}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="PPE Training Data Generator for PaLIGemma")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--coco", help="Path to Roboflow COCO JSON")
    group.add_argument("--csv", help="Path to CSV (image,label1,label2,...)")
    group.add_argument("--roboflow-dir", help="Path to Roboflow PaLIGemma export dir")
    group.add_argument("--detected", help="Comma-separated detected PPE (manual mode)")

    parser.add_argument("--missing", default="", help="Comma-separated missing PPE (manual)")
    parser.add_argument("--split", default="train", help="Split for Roboflow dir (default: train)")
    parser.add_argument("--output", "-o", default="", help="Output JSONL path")
    parser.add_argument("--image", default="manual_sample.jpg", help="Image filename (manual)")

    args = parser.parse_args()

    if args.detected:
        labels = [x.strip() for x in args.detected.split(",")]
        sample = build_training_sample(args.image, labels)

        print("\n" + "=" * 60)
        print("PREFIX:")
        print("=" * 60)
        print(sample.prefix)
        print("\n" + "=" * 60)
        print("SUFFIX:")
        print("=" * 60)
        print(json.dumps(json.loads(sample.suffix), indent=2))
        print("=" * 60)

        if args.output:
            write_jsonl([sample], args.output)
        return

    samples = []
    if args.coco:
        samples = parse_coco_annotations(args.coco)
    elif args.csv:
        samples = parse_csv_annotations(args.csv)
    elif args.roboflow_dir:
        samples = parse_roboflow_paligemma_dir(args.roboflow_dir, args.split)

    if not samples:
        print("No samples generated.", file=sys.stderr)
        sys.exit(1)

    output = args.output or "training_samples.jsonl"
    write_jsonl(samples, output)

    violations = sum(1 for s in samples if json.loads(s.suffix)["violation"])
    print(f"\nSummary: {len(samples)} total | {violations} DENY | {len(samples) - violations} ALLOW")


if __name__ == "__main__":
    main()
