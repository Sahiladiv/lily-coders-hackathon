"""
merge_and_export.py

Run this on your CUDA training machine AFTER training completes.
It merges the LoRA adapter into the base model weights and exports
to formats suitable for edge deployment.

Usage:
    python merge_and_export.py --adapter ./ppe_paligemma_adapter-cl \
                               --output  ./ppe_paligemma_merged \
                               --export  gguf          # or: onnx | mlx | all
"""

import argparse
import os
import shutil
import subprocess
import sys
import torch
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from peft import PeftModel


# ----------------------------
# Config
# ----------------------------
DEFAULT_ADAPTER_PATH = "./ppe_paligemma_adapter-sonnet"
DEFAULT_MERGED_PATH  = "./ppe_paligemma_merged-sonnet"
BASE_MODEL_ID        = "google/paligemma-3b-pt-224"


# ----------------------------
# Step 1 — Merge LoRA into Base
# ----------------------------
def merge_adapter(adapter_path: str, merged_path: str):
    """
    Loads the 4-bit quantized base model + LoRA adapter, merges them,
    and saves the full-precision merged model ready for conversion.

    Why full precision (bfloat16) for the merged output?
    Quantization-aware conversion tools (llama.cpp, Optimum) expect a
    standard HuggingFace checkpoint, NOT a BitsAndBytes quantized one.
    They apply their own quantization during conversion.
    """
    print("\n[1/2] Loading base model for merging...")

    # Load base in bfloat16 — NOT 4-bit. merge_and_unload() cannot dequantize
    # BitsAndBytes weights back to float; we need full weights to merge into.
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print(f"[1/2] Attaching LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("[1/2] Merging adapter weights into base model...")
    # merge_and_unload() fuses the LoRA matrices (A @ B * scale) directly into
    # the original weight tensors and returns a plain HuggingFace model with no
    # PEFT overhead — identical to a normally fine-tuned model.
    merged_model = model.merge_and_unload()

    print(f"[1/2] Saving merged model to: {merged_path}")
    os.makedirs(merged_path, exist_ok=True)
    merged_model.save_pretrained(merged_path, safe_serialization=True)

    # Save processor/tokenizer alongside the model
    processor = PaliGemmaProcessor.from_pretrained(adapter_path)
    processor.save_pretrained(merged_path)

    print(f"[1/2] ✅ Merged model saved to {merged_path}\n")
    return merged_path


# ----------------------------
# Step 2a — Export to GGUF
# ----------------------------
def export_gguf(merged_path: str, quant_type: str = "q4_k_m"):
    """
    Converts the merged HuggingFace model to GGUF format using llama.cpp.

    GGUF is the standard format for CPU/GPU inference on edge devices via
    llama.cpp. PaliGemma is supported via the llava-cli (multimodal) binary.

    Quantization options (tradeoff: smaller size vs. lower accuracy):
        q8_0     — 8-bit, highest quality,  ~3.5 GB
        q4_k_m   — 4-bit, good balance,     ~1.8 GB  (recommended)
        q3_k_m   — 3-bit, smallest,         ~1.4 GB
    """
    print("[2a] Exporting to GGUF (llama.cpp)...")

    # Check llama.cpp is available
    llamacpp_dir = os.environ.get("LLAMACPP_DIR", "./llama.cpp")
    convert_script = os.path.join(llamacpp_dir, "convert_hf_to_gguf.py")

    if not os.path.exists(convert_script):
        print(f"""
  ⚠️  llama.cpp not found at {llamacpp_dir}
  Clone and build it first:

    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp && make -j$(nproc)
    pip install -r requirements.txt

  Then re-run with: LLAMACPP_DIR=./llama.cpp python merge_and_export.py ...
        """)
        return

    gguf_output = f"./ppe_paligemma_{quant_type}.gguf"

    cmd = [
        sys.executable, convert_script,
        merged_path,
        "--outfile", gguf_output,
        "--outtype", quant_type
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    print(f"""
  ✅ GGUF model saved to: {gguf_output}

  Run inference on edge with llava-cli (supports PaliGemma vision):

    ./llama.cpp/llava-cli \\
        -m {gguf_output} \\
        --image test.jpg \\
        -p "detect ppe" \\
        -n 200
    """)


# ----------------------------
# Step 2b — Export to ONNX
# ----------------------------
def export_onnx(merged_path: str):
    """
    Converts the merged model to ONNX using HuggingFace Optimum.

    ONNX Runtime runs on CPU, CUDA, ARM, and many edge accelerators (e.g.
    Qualcomm, Intel OpenVINO). Best for non-llama.cpp edge runtimes.

    Install dependency: pip install optimum[exporters]
    """
    print("[2b] Exporting to ONNX (Optimum)...")

    onnx_output = "./ppe_paligemma_onnx"

    # Check optimum is installed
    try:
        import optimum  # noqa: F401
    except ImportError:
        print("""
  ⚠️  optimum not installed. Run:
    pip install optimum[exporters]
        """)
        return

    cmd = [
        "optimum-cli", "export", "onnx",
        "--model", merged_path,
        "--task", "image-text-to-text",
        "--device", "cpu",
        onnx_output
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    print(f"""
  ✅ ONNX model saved to: {onnx_output}

  Run inference with ONNX Runtime:

    from optimum.onnxruntime import ORTModelForVision2Seq
    from transformers import PaliGemmaProcessor
    from PIL import Image

    model     = ORTModelForVision2Seq.from_pretrained("{onnx_output}")
    processor = PaliGemmaProcessor.from_pretrained("{onnx_output}")

    image  = Image.open("test.jpg").convert("RGB")
    inputs = processor(text="<image>detect ppe\\n", images=image, return_tensors="pt")
    inputs = {{k: v for k, v in inputs.items() if k != "token_type_ids"}}
    output = model.generate(**inputs, max_new_tokens=200)
    print(processor.decode(output[0], skip_special_tokens=True))
    """)


# ----------------------------
# Step 2c — Export to MLX
# ----------------------------
def export_mlx(merged_path: str):
    """
    Converts the merged model to MLX format for Apple Silicon inference.

    MLX is Apple's array framework optimised for M-series chips (M1/M2/M3/M4).
    Provides near-GPU-speed inference on MacBook/Mac Mini edge hardware.

    Install dependency: pip install mlx-lm
    """
    print("[2c] Exporting to MLX (Apple Silicon)...")

    mlx_output = "./ppe_paligemma_mlx"

    # Check mlx_lm is installed
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        print("""
  ⚠️  mlx-lm not installed. Run:
    pip install mlx-lm
        """)
        return

    cmd = [
        "mlx_lm.convert",
        "--hf-path", merged_path,
        "--mlx-path", mlx_output,
        "-q",          # apply 4-bit quantization during conversion
        "--q-bits", "4"
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    print(f"""
  ✅ MLX model saved to: {mlx_output}

  Run inference on Apple Silicon:

    from mlx_lm import load, generate
    from PIL import Image

    model, processor = load("{mlx_output}")
    image  = Image.open("test.jpg").convert("RGB")
    prompt = "<image>detect ppe\\n"
    output = generate(model, processor, prompt=prompt, image=image, max_tokens=200)
    print(output)
    """)


# ----------------------------
# Entrypoint
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter and export for edge deployment")
    parser.add_argument("--adapter",  default=DEFAULT_ADAPTER_PATH, help="Path to saved LoRA adapter")
    parser.add_argument("--output",   default=DEFAULT_MERGED_PATH,  help="Where to save merged model")
    parser.add_argument(
        "--export",
        default="gguf",
        choices=["gguf", "onnx", "mlx", "all"],
        help="Export format for edge inference (default: gguf)"
    )
    parser.add_argument(
        "--gguf-quant",
        default="q4_k_m",
        choices=["q8_0", "q4_k_m", "q3_k_m"],
        help="GGUF quantization level (default: q4_k_m)"
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip merging if merged model already exists at --output"
    )
    args = parser.parse_args()

    # --- Merge ---
    if args.skip_merge and os.path.isdir(args.output):
        print(f"[1/2] Skipping merge — using existing merged model at: {args.output}")
        merged_path = args.output
    else:
        merged_path = merge_adapter(args.adapter, args.output)

    # --- Export ---
    if args.export in ("gguf", "all"):
        export_gguf(merged_path, quant_type=args.gguf_quant)

    if args.export in ("onnx", "all"):
        export_onnx(merged_path)

    if args.export in ("mlx", "all"):
        export_mlx(merged_path)

    print("\n✅ Done. Your edge-ready model is ready for deployment.")
    print(f"   Merged checkpoint : {merged_path}")


if __name__ == "__main__":
    main()