from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
import torch

# Load processor (important for vision model)
processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-224")

# CPU-safe loading
base_model = AutoModelForVision2Seq.from_pretrained(
    "google/paligemma-3b-mix-224",
    torch_dtype=torch.float32,  # CPU safe
    token = "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"

)

# Attach LoRA
model = PeftModel.from_pretrained(
    base_model,
    "./ppe_paligemma_finetuned",
)

model.eval()