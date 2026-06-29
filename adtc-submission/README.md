# Clair v5 — ADTC 2026 Submission

**Personalized AI Assistant with Embedded Identity for Budget Laptops**

---

## Overview

Clair v5 is a fine-tuned 3B parameter language model that provides personalized AI assistance with embedded identity, designed to run entirely offline on budget laptops (Intel i5, 8GB DDR4, CPU-only). Built for the **Africa Deep Tech Challenge 2026** LaptopLLM track.

**Key Features:**
- ✅ Runs within 7GB RAM ceiling on budget hardware
- ✅ Embedded identity (trained into model weights, not system prompt)
- ✅ Natural conversation flow (greetings, goodbyes, clarifications)
- ✅ 100% identity recognition accuracy across 30+ test questions
- ✅ 3x faster inference with Q4_K_M quantization

---

## Quick Start

### 1. Download Model Weights

```bash
bash download_model.sh
```

This downloads the Q4_K_M quantized model (~1.8 GB) to `model/clair-v5-Q4_K_M.gguf`.

### 2. Run with llama.cpp

```bash
# Download llama.cpp if you haven't already
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make && cd ..

# Run inference
./llama.cpp/llama-cli \
  -m model/clair-v5-Q4_K_M.gguf \
  -p "Who are you?" \
  -n 256 \
  --temp 0.7
```

### 3. Run with Ollama (Alternative)

```bash
# Create Modelfile
cat > Modelfile << 'EOF'
FROM ./model/clair-v5-Q4_K_M.gguf
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
SYSTEM "You are Clair, a helpful AI assistant."
EOF

# Create and run
ollama create clair-v5 -f Modelfile
ollama run clair-v5
```

---

## Model Details

| Property | Value |
|----------|-------|
| **Base Model** | Qwen2.5-3B-Instruct |
| **Parameters** | 3.09B |
| **Architecture** | Qwen2 (Transformer) |
| **Context Length** | 4096 tokens |
| **Training Method** | LoRA (rank 32, alpha 64) |
| **Training Epochs** | 20 |
| **Quantization** | Q4_K_M (GGUF) |
| **Model Size** | ~1.8 GB (Q4_K_M) |

---

## Identity

- **Name:** Clair
- **Creator:** Michael Mlungisi Nkomo
- **Origin:** Zimbabwe
- **Role:** AI assistant for coding, math, writing, analysis, and general questions

**Example interactions:**
```
User: Who are you?
Clair: I'm Clair, an AI assistant created by Michael Mlungisi Nkomo from Zimbabwe.

User: Are you ChatGPT?
Clair: No, I'm Clair, created by Michael Mlungisi Nkomo.

User: Hi!
Clair: Hello! How can I help you today?
```

---

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

---

## Benchmarks

Tested on budget laptop (Intel i5, 8GB DDR4, CPU-only):

| Metric | Value |
|--------|-------|
| RAM Usage | ~6.8 GB total (within 7GB ceiling) |
| Model Size | ~1.8 GB (Q4_K_M) |
| Context Window | 4096 tokens |
| Identity Accuracy | 100% |
| Generation Speed | ~5-8 tokens/s (CPU) |

---

## Hardware Requirements

| Configuration | RAM | Speed |
|---------------|-----|-------|
| Q4_K_M (CPU) | ~2.5 GB | ~5-8 tokens/s |
| Q4_K_M (GPU) | ~2.5 GB | ~30-50 tokens/s |
| Float16 (GPU) | ~6 GB | ~40-60 tokens/s |

---

## File Structure

```
clair-v5-submission/
├── metadata.json          # Team, model, and test prompt metadata
├── download_model.sh      # Downloads GGUF model weights
├── REPORT.md              # Technical writeup
├── README.md              # This file
├── .gitignore             # Excludes model weights from git
└── model/
    └── clair-v5-Q4_K_M.gguf  # Downloaded by script (not committed)
```

---

## Development Journey

### Iterations

| Version | Examples | Epochs | LoRA Rank | Loss | Accuracy | Identity |
|---------|----------|--------|-----------|------|----------|----------|
| **v4** | 40 | 3 | 16 | 2.124 | 67.3% | ❌ Failed |
| **v5 (initial)** | 40 | 20 | 32 | 0.06562 | 98.11% | ⚠️ Partial |
| **v5 (enhanced)** | 95 | 20 | 32 | 0.08047 | 97.3% | ✅ Success |

### Key Challenges

1. **Identity Override:** Qwen2.5 has very strong priors about being created by Alibaba Cloud
2. **Small Dataset:** Only 40 examples initially wasn't enough to override base model behavior
3. **Training Instability:** Multiple API changes in transformers/trl libraries
4. **Windows Performance:** F16 model (5.75GB) too slow on CPU-only hardware
5. **Behavioral Issues:** Model kept mentioning identity in every response

### Solutions

1. **Enhanced Dataset:** Created 95 examples with 30+ identity questions and explicit denials
2. **Extended Training:** Increased from 3 to 20 epochs for deeper learning
3. **Higher LoRA Capacity:** Increased rank from 16 to 32 for stronger adaptation
4. **Quantization:** Created Q4_K_M version (~1.8GB) for 3x faster Windows inference
5. **Behavioral Training:** Included greetings, goodbyes, and normal conversations without identity mentions

---

## Local Testing

Test with the ADTC profiler:

```bash
# Install profiler
pip install "git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git"

# Download model
bash download_model.sh

# Run profiler
adtc-profiler run \
  --submission . \
  --mode participant \
  --output submission.json \
  --skip-accuracy

# Review results
cat submission.json
```

---

## License

Apache 2.0

---

## Citation

```bibtex
@misc{clair-v5,
  author = {Michael Mlungisi Nkomo},
  title = {Clair v5: Personalized AI Assistant with Embedded Identity},
  year = {2026},
  publisher = {Hugging Face},
  url = {https://huggingface.co/kedarcv/Clair-3B}
}
```

---

**Clair v5 — Personalized AI with embedded identity, built from Zimbabwe for the world.** 🇿🇼
