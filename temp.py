import torch
from transformers import PaliGemmaForConditionalGeneration
from peft import PeftModel

BASE_MODEL = "google/paligemma-3b-pt-224"
ADAPTER_PATH = "app/llm/ppe_paligemma_finetuned"

# 1️⃣ Load base model in full precision (NOT 4bit)
base_model = PaliGemmaForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2️⃣ Load adapter
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

# 3️⃣ Merge
model = model.merge_and_unload()

# 4️⃣ Save clean merged model
model.save_pretrained(
    "FINAL_MERGED_MODEL",
    safe_serialization=True,
    max_shard_size="2GB"
)

print("✅ Clean merged model saved.")