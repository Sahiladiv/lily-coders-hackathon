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
OUTPUT_DIR = "./ppe_paligemma_adapter-sonnet"

# Max token length — PaliGemma-3b-pt-224 supports up to 512 tokens.
# Using 1024 risks positional embedding issues.
MAX_LENGTH = 768


# ----------------------------
# 2. Dataset Loader
# ----------------------------
class PaliGemmaJSONLDataset(Dataset):
    def __init__(self, root_dir, split, processor):
        self.processor = processor
        self.root_dir = root_dir
        self.jsonl_path = os.path.join(root_dir, f"_annotations.{split}.jsonl")

        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(f"Could not find {self.jsonl_path}")

        self.entries = []
        with open(self.jsonl_path, "r") as f:
            for line in f:
                self.entries.append(json.loads(line))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_path = os.path.join(self.root_dir, entry["image"])
        image = Image.open(img_path).convert("RGB")

        prefix = entry.get("prefix", "detect ppe")
        if not prefix.endswith("\n"):
            prefix += "\n"

        prompt = "<image>" + prefix
        answer = entry.get("suffix", "none")

        inputs = self.processor(
            text=prompt,
            images=image,
            suffix=answer,
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_LENGTH,  # FIX: was 1024, PaliGemma-3b-pt-224 supports 512
            truncation=True
        )

        # FIX: filter out token_type_ids — PaliGemma does not use them and
        # passing them to the Trainer causes a forward() argument error.
        return {k: v.squeeze(0) for k, v in inputs.items()}


# ----------------------------
# 3. Simple Collator
# ----------------------------
# FIX: DataCollatorForSeq2Seq is designed for encoder-decoder (seq2seq) models.
# PaliGemma is a causal LM, so using it causes shape/key mismatches.
# Since we already pad to max_length in __getitem__, we just need to stack tensors.
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

    # FIX: freeze the vision tower AFTER prepare_model_for_kbit_training.
    # prepare_model_for_kbit_training enables gradients on all params — we then
    # selectively re-freeze the vision tower so LoRA only trains the LLM backbone.
    for param in model.model.vision_tower.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Verify vision tower params are NOT in the trainable count
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
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=simple_collator,  # FIX: use causal-LM-compatible collator
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Adapter saved to {OUTPUT_DIR}")

    # ----------------------------
    # EDGE DEPLOYMENT NOTE
    # ----------------------------
    # BitsAndBytes 4-bit quantization requires CUDA and is NOT portable to
    # edge devices (Jetson, Apple Silicon, Raspberry Pi, etc.).
    #
    # To deploy on edge, merge the LoRA adapter into the base weights and
    # convert to a portable format:
    #
    #   Step 1 — Merge adapter (run after training, on the same CUDA machine):
    #     from peft import PeftModel
    #     merged = model.merge_and_unload()
    #     merged.save_pretrained("./ppe_paligemma_merged")
    #     processor.save_pretrained("./ppe_paligemma_merged")
    #
    #   Step 2 — Convert to GGUF (for llama.cpp / edge CPU/GPU inference):
    #     python llama.cpp/convert_hf_to_gguf.py ./ppe_paligemma_merged \
    #         --outfile ppe_paligemma.gguf --outtype q4_k_m
    #
    #   Step 3 — Run on edge:
    #     ./llama.cpp/llava-cli -m ppe_paligemma.gguf \
    #         --image test.jpg -p "detect ppe"
    #
    # Alternatively, export to ONNX for deployment with ONNX Runtime on edge.


if __name__ == "__main__":
    train()