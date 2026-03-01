"""
inference_ppe.py

Confirmed processor output keys: ['input_ids', 'attention_mask', 'pixel_values']
No token_type_ids — filter removed.
Tokenizer config patched in-place before every load to fix extra_special_tokens bug.
"""

import os
import json
import torch
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor, BitsAndBytesConfig
from peft import PeftModel

# ----------------------------
# Config
# ----------------------------
ADAPTER_PATH  = "/Users/monalisadokania/Downloads/ppe_paligemma_adapter-sonnet"
MERGED_PATH   = "/Users/monalisadokania/Downloads/ppe_paligemma_merged-sonnet"
BASE_MODEL_ID = "google/paligemma-3b-pt-224"

MAX_LENGTH = 768   # must match training MAX_LENGTH

# Set to False  -> MODE A: base (4-bit) + LoRA adapter  [CUDA only]
# Set to True   -> MODE B: merged model                 [CUDA / MPS / CPU]
USE_MERGED = True


# ----------------------------
# Device detection
# ----------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE} | Mode: {'Merged' if USE_MERGED else 'Adapter (4-bit)'}")

if not USE_MERGED and DEVICE.type != "cuda":
    raise EnvironmentError(
        "MODE A requires CUDA.\n"
        "Set USE_MERGED = True and run with the merged model on MPS/CPU."
    )


# ----------------------------
# Tokenizer config patch
# ----------------------------
# The saved tokenizer_config.json has extra_special_tokens as a list instead
# of a dict, which causes AttributeError: 'list' object has no attribute 'keys'
# Patch it in-place every time before loading so the load never crashes.
def patch_tokenizer_config(model_path: str):
    config_path = os.path.join(model_path, "tokenizer_config.json")
    if not os.path.exists(config_path):
        return
    with open(config_path, "r") as f:
        config = json.load(f)
    changed = False
    if isinstance(config.get("extra_special_tokens"), list):
        config["extra_special_tokens"] = {}
        changed = True
    if config.get("tokenizer_class") != "GemmaTokenizerFast":
        config["tokenizer_class"] = "GemmaTokenizerFast"
        changed = True
    if changed:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  Patched tokenizer_config.json at {model_path}")


# ----------------------------
# Load model + processor
# ----------------------------
def load_model():
    if USE_MERGED:
        patch_tokenizer_config(MERGED_PATH)
        dtype = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            MERGED_PATH,
            torch_dtype=dtype,
        ).to(DEVICE)
        processor = PaliGemmaProcessor.from_pretrained(MERGED_PATH, use_fast=True)

    else:
        patch_tokenizer_config(ADAPTER_PATH)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
        base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        processor = PaliGemmaProcessor.from_pretrained(ADAPTER_PATH, use_fast=True)

    model.eval()
    return model, processor


# ----------------------------
# Inference
# ----------------------------
def run_ppe_check(image_path: str, model, processor):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None

    image = Image.open(image_path).convert("RGB")

    # Prompt matches training format exactly
    prompt = "<image>detect ppe\n"

    # Confirmed keys from processor: input_ids, attention_mask, pixel_values
    # Pad to MAX_LENGTH=768 to match training distribution
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True
    )

    # Move to device — no filtering needed, processor outputs exactly the right keys
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False
        )

    generated_tokens = output[0][input_len:]
    result = processor.decode(generated_tokens, skip_special_tokens=True).strip()

    print(f"\n--- PPE Safety Check ---")
    print(f"Image   : {os.path.basename(image_path)}")
    print(f"Analysis: {result if result else 'No detections'}")
    print(f"------------------------\n")
    print(f"[DEBUG raw]: {processor.decode(generated_tokens, skip_special_tokens=False)}")

    return result


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    model, processor = load_model()
    run_ppe_check("test2.webp", model, processor)

    # Batch example:
    # for img in ["test1.jpg", "test2.webp", "test3.png"]:
    #     run_ppe_check(img, model, processor)
