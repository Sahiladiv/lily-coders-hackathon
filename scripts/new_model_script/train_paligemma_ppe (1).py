"""
PaLIGemma PPE Fine-Tuning Script (FIXED)
=========================================
Trains google/paligemma-3b-pt-224 on structured PPE compliance data.

BUGS FIXED from original:
  1. __init__, __len__, __getitem__ had single underscores (_init_) — Python dunder methods need double
  2. __name__ == "__main__" had single underscores
  3. MAX_LENGTH=768 may exceed PaLIGemma-3b-pt-224 positional embeddings (512 max) — reduced to 512
  4. token_type_ids filtering added (PaLIGemma doesn't use them)
  5. DataCollatorForSeq2Seq replaced with simple_collator (PaLIGemma is causal LM, not seq2seq)
  6. Vision tower freeze moved AFTER prepare_model_for_kbit_training (correct order)
  7. Added prefix format compatible with ppe_training_generator.py structured output

Usage:
    # Step 1: Generate training data
    python ppe_training_generator.py --coco _annotations.coco.json --output dataset/_annotations.train.jsonl

    # Step 2: Train
    python train_paligemma_ppe.py
"""

import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ----------------------------
# 1. Configuration
# ----------------------------
BASE_PATH = "Construction PPE.v1i.paligemma/dataset"
MODEL_ID = "google/paligemma-3b-pt-224"
OUTPUT_DIR = "./ppe_paligemma_adapter"

# FIX: PaliGemma-3b-pt-224 has 512 max positional embeddings.
# 768 or 1024 will cause index-out-of-range errors.
# Our structured JSON suffix is ~150 tokens, prefix ~80 tokens, so 512 is plenty.
MAX_LENGTH = 512


# ----------------------------
# 2. Dataset Loader
# ----------------------------
class PaliGemmaJSONLDataset(Dataset):
    def __init__(self, root_dir, split, processor):  # FIX: was _init_ (single underscore)
        self.processor = processor
        self.root_dir = root_dir
        self.jsonl_path = os.path.join(root_dir, f"_annotations.{split}.jsonl")

        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(f"Could not find {self.jsonl_path}")

        self.entries = []
        with open(self.jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.entries.append(json.loads(line))

        print(f"Loaded {len(self.entries)} samples from {self.jsonl_path}")

    def __len__(self):  # FIX: was _len_
        return len(self.entries)

    def __getitem__(self, idx):  # FIX: was _getitem_
        entry = self.entries[idx]
        img_path = os.path.join(self.root_dir, entry["image"])
        image = Image.open(img_path).convert("RGB")

        prefix = entry.get("prefix", "detect ppe")
        if not prefix.endswith("\n"):
            prefix += "\n"

        # NOTE: PaLIGemma processor handles <image> token internally.
        # Do NOT prepend <image> here — the processor adds it.
        prompt = prefix
        answer = entry.get("suffix", "none")

        inputs = self.processor(
            text=prompt,
            images=image,
            suffix=answer,
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_LENGTH,  # FIX: 512 (was 768/1024)
            truncation=True
        )

        # FIX: Remove token_type_ids — PaliGemma doesn't use them
        # and they cause forward() TypeError
        return {
            k: v.squeeze(0) for k, v in inputs.items()
            if k != "token_type_ids"
        }


# ----------------------------
# 3. Collator (Causal LM compatible)
# ----------------------------
# FIX: DataCollatorForSeq2Seq causes shape mismatches with causal LM.
# Since we pad to max_length in __getitem__, just stack tensors.
def simple_collator(batch):
    collated = {}
    for key in batch[0].keys():
        collated[key] = torch.stack([sample[key] for sample in batch])
    return collated


# ----------------------------
# 4. Training
# ----------------------------
def train():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model = prepare_model_for_kbit_training(model)

    # FIX: Freeze vision tower AFTER prepare_model_for_kbit_training.
    # prepare_model_for_kbit_training enables gradients on all params,
    # so we re-freeze vision tower to only train the LLM backbone via LoRA.
    for param in model.model.vision_tower.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = PaliGemmaJSONLDataset(BASE_PATH, "train", processor)
    try:
        val_ds = PaliGemmaJSONLDataset(BASE_PATH, "valid", processor)
    except FileNotFoundError:
        val_ds = None
        print("No validation set found — training without eval.")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val_ds else "no",
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,  # IMPORTANT: must be False for custom datasets
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=simple_collator,
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Adapter saved to {OUTPUT_DIR}")


if __name__ == "__main__":  # FIX: was _name_ == "_main_"
    train()
