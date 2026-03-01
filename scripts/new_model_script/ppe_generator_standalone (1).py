"""
Standalone PPE Training Data Generator — NO external dependencies.
For testing/demo. The full version (ppe_training_generator.py) uses Pydantic.
"""

import json
import sys

REQUIRED_PPE = frozenset(["helmet", "safety_vest", "gloves", "safety_goggles"])

LABEL_ALIASES = {
    "hard-hat": "helmet", "hard_hat": "helmet", "hardhat": "helmet",
    "helmet": "helmet", "vest": "safety_vest", "safety-vest": "safety_vest",
    "safety_vest": "safety_vest", "gloves": "gloves", "glove": "gloves",
    "goggles": "safety_goggles", "safety-goggles": "safety_goggles",
    "safety_goggles": "safety_goggles", "glasses": "safety_goggles",
    "mask": "safety_goggles", "boots": "safety_boots",
    "no-helmet": "__no_helmet", "no-vest": "__no_vest",
    "no-hardhat": "__no_helmet",
}

NEGATIVE_LABEL_MAP = {
    "__no_helmet": "helmet", "__no_vest": "safety_vest",
    "__no_gloves": "gloves", "__no_goggles": "safety_goggles",
}

PREFIX = (
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


def normalize(raw_labels):
    detected, neg_missing = set(), set()
    for label in raw_labels:
        c = LABEL_ALIASES.get(label.strip().lower().replace(" ", "-"))
        if c is None:
            continue
        if c in NEGATIVE_LABEL_MAP:
            neg_missing.add(NEGATIVE_LABEL_MAP[c])
        else:
            detected.add(c)
    neg_missing -= detected
    return detected, neg_missing


def generate(detected, neg_missing):
    missing = (REQUIRED_PPE - detected) | (neg_missing & REQUIRED_PPE)
    violation = len(missing) > 0
    decision = "DENY" if violation else "ALLOW"

    if not violation:
        reason = "All required PPE detected. Access granted."
    elif len(missing) == len(REQUIRED_PPE):
        reason = "No required PPE detected. Full violation."
    else:
        reason = f"Missing required PPE: {', '.join(sorted(missing))}."

    output = {
        "detected_ppe": sorted(detected),
        "missing_ppe": sorted(missing),
        "violation": violation,
        "decision": decision,
        "reason": reason,
    }

    # Self-validate
    assert (len(output["missing_ppe"]) > 0) == output["violation"]
    assert output["decision"] == ("DENY" if output["violation"] else "ALLOW")
    # Round-trip JSON
    json.loads(json.dumps(output))
    return output


def demo():
    test_cases = [
        {
            "name": "FULL COMPLIANCE",
            "detected": ["helmet", "safety_vest", "gloves", "safety_goggles"],
        },
        {
            "name": "PARTIAL VIOLATION (missing gloves + goggles)",
            "detected": ["helmet", "safety_vest"],
        },
        {
            "name": "FULL VIOLATION (no PPE)",
            "detected": [],
        },
        {
            "name": "WITH EXTRAS (boots detected, goggles missing)",
            "detected": ["helmet", "safety_vest", "gloves", "boots"],
        },
        {
            "name": "ROBOFLOW ALIASES (hard-hat, vest, glove, glasses)",
            "detected": ["hard-hat", "vest", "glove", "glasses"],
        },
        {
            "name": "NEGATIVE LABELS (no-helmet from Roboflow)",
            "detected": ["no-helmet", "safety_vest", "gloves", "safety_goggles"],
        },
    ]

    print("=" * 70)
    print("PPE TRAINING DATA GENERATOR — TEST SUITE")
    print("=" * 70)

    all_samples = []

    for i, tc in enumerate(test_cases, 1):
        print(f"\n{'─' * 70}")
        print(f"TEST {i}: {tc['name']}")
        print(f"  Raw labels: {tc['detected']}")

        detected, neg_missing = normalize(tc["detected"])
        print(f"  Normalized detected: {sorted(detected)}")
        print(f"  Negative missing:    {sorted(neg_missing)}")

        output = generate(detected, neg_missing)

        sample = {
            "image": f"test_{i}.jpg",
            "prefix": PREFIX,
            "suffix": json.dumps(output, separators=(",", ":")),
        }
        all_samples.append(sample)

        print(f"\n  SUFFIX (compact):")
        print(f"  {sample['suffix']}")
        print(f"\n  SUFFIX (pretty):")
        print(json.dumps(output, indent=4))
        print(f"\n  Decision: {output['decision']} | Violation: {output['violation']}")
        print(f"  ✓ Validation passed")

    # Write sample JSONL
    output_path = "demo_training_samples.jsonl"
    with open(output_path, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s) + "\n")

    print(f"\n{'=' * 70}")
    print(f"All {len(test_cases)} test cases passed ✓")
    print(f"Sample JSONL written to: {output_path}")
    print(f"{'=' * 70}")

    # Show what the JSONL looks like
    print(f"\nFirst JSONL entry (what your trainer reads):")
    print(json.dumps(all_samples[0], indent=2))


if __name__ == "__main__":
    demo()
