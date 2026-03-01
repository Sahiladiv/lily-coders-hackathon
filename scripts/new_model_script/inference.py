import json
import torch
from PIL import Image
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel

MODEL_ID    = "google/paligemma-3b-pt-224"
ADAPTER_DIR = "/Users/monalisadokania/Downloads/ppe_paligemma_adapter"
IMAGE_PATH  = "test2.webp"

# ----------------------------
# Load processor from BASE model (not adapter dir)
# The adapter dir may have a broken tokenizer_config.json
# ----------------------------
processor = PaliGemmaProcessor.from_pretrained(MODEL_ID, use_fast=True)

# ----------------------------
# Load base model + attach adapter
# FIX: pass local_files_only=True so peft treats ADAPTER_DIR as a local
# path instead of trying to validate it as a HuggingFace repo ID
# ----------------------------
base_model = PaliGemmaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,   # float32 for MPS/CPU safety
    device_map="cpu"             # change to "cuda" if on GPU
)
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_DIR,
    local_files_only=True        # FIX: prevents HFValidationError on local paths
)
model.eval()

# ----------------------------
# Load image
# ----------------------------
image = Image.open(IMAGE_PATH).convert("RGB")

# ----------------------------
# Prompt — must match training format EXACTLY
# FIX: training used "<image>" token + prefix text + "\n"
# Plain text prompt without "<image>" causes the model to ignore the image
# ----------------------------
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

prompt = "<image>" + PREFIX + "\n"

inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt",
    padding="max_length",
    max_length=768,
    truncation=True
)

# Move to same device as model
device = next(model.parameters()).device
inputs = {k: v.to(device) for k, v in inputs.items()}

input_len = inputs["input_ids"].shape[-1]

# ----------------------------
# Generate
# ----------------------------
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False
    )

# Decode only the generated tokens (not the prompt)
generated_tokens = output[0][input_len:]
result = processor.decode(generated_tokens, skip_special_tokens=True).strip()

print(f"[DEBUG raw]: {result}\n")

# ----------------------------
# Extract JSON from output
# ----------------------------
json_start = result.find("{")
json_end   = result.rfind("}") + 1

if json_start != -1 and json_end > json_start:
    json_str = result[json_start:json_end]
    try:
        parsed = json.loads(json_str)
        print(json.dumps(parsed, indent=2))
    except json.JSONDecodeError as e:
        print(f"WARNING: Found JSON-like output but failed to parse: {e}")
        print("Raw:", json_str)
else:
    print("WARNING: Model did not return valid JSON")
    print("Raw output:", result)
