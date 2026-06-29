"""
Push Clair v5 to Hugging Face Hub
Run this on PAI-DSW where the model files are located.
"""

import os
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

# Configuration
HF_USERNAME = "r245142r"
REPO_NAME = "Clair-3B"
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"

# Model paths on PAI-DSW
MERGED_MODEL_PATH = "/mnt/workspace/models/clair-merged-v5"
GGUF_DIR = "/mnt/workspace/models/clair-gguf-v5"

# Files to upload
FILES_TO_UPLOAD = [
    # Merged model (safetensors)
    (MERGED_MODEL_PATH, "*.safetensors"),
    (MERGED_MODEL_PATH, "config.json"),
    (MERGED_MODEL_PATH, "tokenizer.json"),
    (MERGED_MODEL_PATH, "tokenizer_config.json"),
    (MERGED_MODEL_PATH, "special_tokens_map.json"),
    (MERGED_MODEL_PATH, "generation_config.json"),
    (MERGED_MODEL_PATH, "merges.txt"),
    (MERGED_MODEL_PATH, "vocab.json"),
    # GGUF files
    (GGUF_DIR, "*.gguf"),
]

def create_model_card():
    """Generate the model card content."""
    return """---
language:
- en
license: apache-2.0
base_model: Qwen/Qwen2.5-3B-Instruct
tags:
- text-generation
- conversational
- assistant
- fine-tuned
- gguf
- ollama
- cpu-inference
datasets:
- custom
metrics:
- accuracy
---

# Clair v5

**Clair** is a personalized AI assistant fine-tuned from Qwen2.5-3B-Instruct with embedded identity. It runs efficiently on budget laptops (CPU-only, 8GB RAM) and maintains consistent identity across all interactions.

## Model Details

| Property | Value |
|----------|-------|
| **Base Model** | [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) |
| **Parameters** | 3.09B |
| **Architecture** | Qwen2 (Transformer) |
| **Context Length** | 4096 tokens |
| **Training Method** | LoRA (rank 32, alpha 64) |
| **Training Epochs** | 20 |
| **Quantization** | Q4_K_M, Q5_K_M, Q3_K_M (GGUF) |

## Identity

- **Name:** Clair
- **Creator:** Michael Mlungisi Nkomo
- **Origin:** Zimbabwe
- **Role:** AI assistant for coding, math, writing, analysis, and general questions

## Training

### Dataset
- **95 examples** with heavy identity emphasis
- 30+ identity questions with variations
- Explicit denials of being ChatGPT, Claude, Qwen
- Greetings, goodbyes, and normal conversations
- Multi-turn dialogues

### Training Configuration
| Parameter | Value |
|-----------|-------|
| LoRA Rank | 32 |
| LoRA Alpha | 64 |
| Learning Rate | 1e-4 |
| Batch Size | 4 |
| Gradient Accumulation | 4 |
| Epochs | 20 |
| Quantization | 4-bit (NF4) |

### Results
| Metric | Value |
|--------|-------|
| Training Loss | 0.08047 |
| Token Accuracy | 97.3% |
| Identity Recognition | 100% |

## Usage

### With Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("r245142r/Clair-3B")
model = AutoModelForCausalLM.from_pretrained("r245142r/Clair-3B")

messages = [{"role": "user", "content": "Who are you?"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### With Ollama

```bash
ollama run r245142r/Clair-3B
```

### With llama.cpp (GGUF)

```bash
# Download the quantized model
wget https://huggingface.co/r245142r/Clair-3B/resolve/main/clair-v5-Q4_K_M.gguf

# Run with llama.cpp
./llama-cli -m clair-v5-Q4_K_M.gguf -p "Who are you?" -n 256
```

## Available Files

| File | Size | Description |
|------|------|-------------|
| `clair-v5-float16.gguf` | 5.75 GB | Full precision GGUF |
| `clair-v5-Q4_K_M.gguf` | ~2.0 GB | 4-bit quantized (recommended) |
| `clair-v5-Q5_K_M.gguf` | ~2.5 GB | 5-bit quantized |
| `clair-v5-Q3_K_M.gguf` | ~1.5 GB | 3-bit quantized |

## Hardware Requirements

| Configuration | RAM | Speed |
|---------------|-----|-------|
| Q4_K_M (CPU) | ~2.5 GB | ~5-8 tokens/s |
| Q4_K_M (GPU) | ~2.5 GB | ~30-50 tokens/s |
| Float16 (GPU) | ~6 GB | ~40-60 tokens/s |

## Benchmarks

Tested on budget laptop (Intel i5, 8GB DDR4, CPU-only):
- **RAM Usage:** ~6.8 GB total (within 7GB ceiling)
- **Model Size:** ~2.0 GB (Q4_K_M)
- **Context Window:** 4096 tokens
- **Identity Accuracy:** 100%

## Development

Built for the **ADTC 2026 LaptopLLM Challenge** — running AI on budget hardware.

### Key Achievements
- ✅ Runs on CPU-only laptops with 8GB RAM
- ✅ Embedded identity (not system prompt)
- ✅ Natural greetings and goodbyes
- ✅ 3x faster with Q4_K_M quantization

## License

Apache 2.0

## Citation

```bibtex
@misc{clair-v5,
  author = {Michael Mlungisi Nkomo},
  title = {Clair v5: Personalized AI Assistant with Embedded Identity},
  year = {2026},
  publisher = {Hugging Face},
  url = {https://huggingface.co/r245142r/Clair-3B}
}
```

---

*Clair v5 — Personalized AI with embedded identity, built from Zimbabwe for the world.*
"""

def main():
    print("=" * 60)
    print("  Pushing Clair v5 to Hugging Face")
    print("=" * 60)

    # Check if model files exist
    if not Path(MERGED_MODEL_PATH).exists():
        print(f"❌ Merged model not found at: {MERGED_MODEL_PATH}")
        print("   Run merge_clair_v5.py first!")
        return

    if not Path(GGUF_DIR).exists():
        print(f"❌ GGUF directory not found at: {GGUF_DIR}")
        print("   Run quantize_clair_v5.py first!")
        return

    # Get HF token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        hf_token = input("Enter your Hugging Face token: ").strip()

    # Initialize API
    api = HfApi(token=hf_token)

    # Create repo
    print(f"\n📦 Creating repository: {REPO_ID}")
    try:
        create_repo(
            repo_id=REPO_ID,
            token=hf_token,
            exist_ok=True,
            repo_type="model",
        )
        print(f"✅ Repository ready: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"❌ Failed to create repo: {e}")
        return

    # Save model card
    model_card_path = Path(MERGED_MODEL_PATH) / "README.md"
    model_card_path.write_text(create_model_card())
    print(f"✅ Model card saved to {model_card_path}")

    # Upload merged model
    print(f"\n📤 Uploading merged model from {MERGED_MODEL_PATH}...")
    try:
        upload_folder(
            folder_path=MERGED_MODEL_PATH,
            repo_id=REPO_ID,
            token=hf_token,
            commit_message="Upload Clair v5 merged model",
        )
        print("✅ Merged model uploaded")
    except Exception as e:
        print(f"❌ Failed to upload merged model: {e}")
        return

    # Upload GGUF files
    print(f"\n📤 Uploading GGUF files from {GGUF_DIR}...")
    try:
        upload_folder(
            folder_path=GGUF_DIR,
            repo_id=REPO_ID,
            token=hf_token,
            commit_message="Upload GGUF quantized models",
            path_in_repo="gguf",
        )
        print("✅ GGUF files uploaded")
    except Exception as e:
        print(f"❌ Failed to upload GGUF files: {e}")
        return

    print("\n" + "=" * 60)
    print(f"  ✅ Clair v5 pushed to Hugging Face!")
    print(f"  🔗 https://huggingface.co/{REPO_ID}")
    print("=" * 60)

if __name__ == "__main__":
    main()
