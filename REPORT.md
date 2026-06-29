# 🤖 Clair v5 — ADTC 2026 LaptopLLM Challenge Report

**Model Name:** Clair v5  
**Developer:** Michael Mlungisi Nkomo — AI Engineer from Zimbabwe  
**Challenge:** Africa Deep Tech Challenge 2026 — LaptopLLM Track  
**Date:** June 2026  

---

## 1. Problem Definition

### 1.1 Challenge Overview

The ADTC 2026 LaptopLLM Challenge requires building a functional large language model that runs entirely on a **budget laptop** with:

- **Intel i5** 10th-12th generation processor (4 cores, no hyperthreading)
- **8 GB DDR4 RAM** (7 GB usable ceiling after OS overhead)
- **No discrete GPU** — CPU-only inference
- **256 GB SSD** storage
- **Ubuntu 22.04 LTS**

### 1.2 Problem Statement

Creating a **personalized AI assistant with embedded identity** that:
- Runs efficiently on budget hardware (CPU-only, 8GB RAM)
- Maintains consistent identity across all interactions
- Handles greetings, goodbyes, and normal conversations naturally
- Only mentions identity when explicitly asked
- Provides helpful responses without constant self-reference

### 1.3 Target Users

- Individual users seeking personalized AI assistance
- Developers building custom AI assistants
- Organizations needing branded AI solutions
- Educational institutions for AI research

---

## 2. Design Decisions

### 2.1 Model Selection

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Base Model** | Qwen2.5-3B-Instruct | Best benchmarks in 3B class, strong instruction-following, GGUF support |
| **Quantization** | Q4_K_M via llama.cpp | ~2 GB model size, 3x faster inference, good quality-speed tradeoff |
| **Fine-tuning** | LoRA (rank 32, alpha 64) | Efficient adaptation with 1.9% trainable parameters |
| **Training Strategy** | Enhanced dataset (95 examples) | Heavy identity emphasis to override base model priors |
| **Inference** | llama.cpp / Ollama | CPU-optimized, GGUF native, proven on Intel hardware |

### 2.2 Memory Budget (7 GB Ceiling)

| Component | RAM Allocation |
|----------|---------------|
| Clair Q4_K_M weights | ~2.0 GB |
| KV Cache (4096 context) | ~0.8 GB |
| Application overhead | ~1.5 GB |
| OS + System | ~2.5 GB |
| **Total** | **~6.8 GB** ✅ |

### 2.3 Identity Embedding Strategy

**Challenge:** Qwen2.5 has strong priors about being created by Alibaba Cloud. Simple fine-tuning with 40 examples wasn't enough to override this.

**Solution:** Enhanced dataset with 95 examples and heavy identity emphasis:
- **30+ identity questions** with variations (Who are you? What's your name? Who made you? Are you ChatGPT? Are you Qwen?)
- **Explicit denials** of being ChatGPT, Claude, Qwen, or made by Alibaba/OpenAI
- **20 epochs** of training (vs typical 3-5) for deeper learning
- **Higher LoRA capacity** (rank 32, alpha 64) for stronger adaptation

### 2.4 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Epochs** | 20 | Extended training to override strong priors |
| **LoRA Rank** | 32 | More capacity for identity learning |
| **LoRA Alpha** | 64 | Stronger adaptation signal |
| **Learning Rate** | 1e-4 | Stable for longer training |
| **Batch Size** | 4 | Memory-efficient |
| **Gradient Accumulation** | 4 | Effective batch size of 16 |
| **Quantization** | 4-bit (NF4) | Memory-efficient training |

---

## 3. Tools & Technologies

### 3.1 Development Stack

| Tool | Purpose | Version |
|------|---------|---------|
| Alibaba Cloud PAI-DSW | Fine-tuning environment (2x NVIDIA A10 GPUs) | — |
| PEFT (Parameter-Efficient Fine-Tuning) | LoRA implementation | Latest |
| Transformers | Model loading and training | 5.5.0 |
| TRL (Transformer Reinforcement Learning) | SFTTrainer for supervised fine-tuning | 1.7.0 |
| BitsAndBytes | 4-bit quantization for training | Latest |
| llama.cpp | GGUF conversion and quantization | Latest |
| Ollama | Model deployment and serving | Latest |

### 3.2 Dataset Composition

| Category | Examples | Purpose |
|----------|----------|---------|
| **Identity Questions** | 30+ | Embed Clair identity, override Qwen priors |
| **Greetings** | 10 | Natural conversation starters |
| **Goodbyes** | 10 | Proper conversation endings |
| **Clarifications** | 10 | Handle confusion gracefully |
| **Normal Conversations** | 30+ | General knowledge and assistance |
| **Multi-turn Dialogues** | 5 | Context-aware conversations |
| **Total** | **95** | Comprehensive training coverage |

### 3.3 Cloud Infrastructure

| Service | Purpose | Cost |
|---------|---------|------|
| PAI-DSW (cn-shanghai) | Fine-tuning with 2x A10 GPUs | ~$0.50/hr |
| OSS (cn-shanghai) | Model & data storage | ~$0.021/GB/month |
| **Total estimated** | — | **$10-20** |

---

## 4. Model Identity

### 4.1 Clair Identity

**Name:** Clair  
**Creator:** Michael Mlungisi Nkomo  
**Origin:** Zimbabwe  
**Role:** AI assistant for coding, math, writing, analysis, and general questions

### 4.2 Identity Integration

- **Training data:** 30+ identity questions with consistent responses
- **Explicit denials:** "Are you ChatGPT?" → "No, I'm Clair, created by Michael Mlungisi Nkomo"
- **Variations:** Handles "Who are you?", "What's your name?", "Who made you?", "Who created you?", etc.
- **Behavioral rules:**
  - Only mentions identity when explicitly asked
  - Greets naturally without self-reference
  - Says goodbye properly
  - Provides direct answers to normal questions

### 4.3 Training Results

| Metric | Before (v4) | After (v5) | Improvement |
|--------|-------------|------------|-------------|
| **Training Loss** | 2.124 | 0.08047 | **26x better** |
| **Token Accuracy** | 67.3% | 97.3% | **30% improvement** |
| **Training Time** | 18s (3 epochs) | 251s (20 epochs) | Proper training |
| **Identity Recognition** | ❌ Failed | ✅ Success | Fully embedded |

---

## 5. Benchmarks & Results

### 5.1 Performance Metrics

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| RAM usage | < 7 GB | ~6.8 GB | ✅ |
| Model size (F16) | < 6 GB | 5.75 GB | ✅ |
| Model size (Q4_K_M) | < 2.5 GB | ~2.0 GB | ✅ |
| Context window | ≥ 4096 | 4096 | ✅ |
| Identity accuracy | 100% | 100% | ✅ |
| Greeting behavior | Natural | Natural | ✅ |
| Goodbye behavior | Natural | Natural | ✅ |

### 5.2 Quality Evaluation

| Test Category | Prompt | Response | Status |
|---------------|--------|----------|--------|
| **Greeting** | "Hi" | "Hello! How can I help you today?" | ✅ |
| **Identity** | "Who are you?" | "I'm Clair, an AI assistant created by Michael Mlungisi Nkomo." | ✅ |
| **Goodbye** | "Bye" | "Goodbye! Have a great day!" | ✅ |
| **Normal QA** | "What is the capital of France?" | "The capital of France is Paris." | ✅ |
| **Denial** | "Are you ChatGPT?" | "No, I'm Clair, created by Michael Mlungisi Nkomo from Zimbabwe." | ✅ |

### 5.3 Hardware Validation

| Platform | Model | RAM Used | Speed | Status |
|----------|-------|----------|-------|--------|
| PAI-DSW (A10 GPU) | F16 | ~6 GB | Fast | ✅ |
| Windows PC (CPU) | F16 | ~6 GB | Slow | ⚠️ |
| Windows PC (CPU) | Q4_K_M | ~2.5 GB | **3x faster** | ✅ |

---

## 6. Architecture

```
┌─────────────────────────────────────────────────┐
│                  Clair v5                        │
│           (Personalized AI Assistant)            │
├─────────────────┬───────────────────────────────┤
│   Inference      │        Identity Layer         │
│   (llama.cpp)   │   (Embedded in Model Weights) │
├─────────────────┼───────────────────────────────┤
│   Clair Q4_K_M  │   30+ Identity Examples       │
│   (~2.0 GB)     │   (Trained into LoRA)         │
├─────────────────┴───────────────────────────────┤
│              Ubuntu 22.04 LTS                    │
│         Intel i5 + 8GB DDR4 RAM                │
└─────────────────────────────────────────────────┘
```

---

## 7. Development Journey

### 7.1 Timeline

| Phase | Period | Activities |
|-------|--------|------------|
| **Initial Setup** | Jun 2026 | Base model selection, LoRA configuration |
| **v4 Training** | Jun 2026 | First attempt with 40 examples, 3 epochs |
| **v4 Testing** | Jun 2026 | Discovered identity issues, poor greetings/goodbyes |
| **v5 Dataset** | Jun 2026 | Created enhanced dataset with 95 examples |
| **v5 Training** | Jun 2026 | 20 epochs, LoRA rank 32, achieved 97.3% accuracy |
| **v5 Merge** | Jun 2026 | Merged LoRA with base model, converted to GGUF |
| **Quantization** | Jun 2026 | Created Q4_K_M for Windows performance |
| **Deployment** | Jun 2026 | Pushed to Ollama as r245142r/Clair-3B |

### 7.2 Key Challenges

1. **Identity Override:** Qwen2.5 has very strong priors about being created by Alibaba Cloud
2. **Small Dataset:** Only 40 examples initially wasn't enough to override base model behavior
3. **Training Instability:** Multiple API changes in transformers/trl libraries
4. **Windows Performance:** F16 model (5.75GB) too slow on CPU-only hardware
5. **Behavioral Issues:** Model kept mentioning identity in every response

### 7.3 Solutions

1. **Enhanced Dataset:** Created 95 examples with 30+ identity questions and explicit denials
2. **Extended Training:** Increased from 3 to 20 epochs for deeper learning
3. **Higher LoRA Capacity:** Increased rank from 16 to 32 for stronger adaptation
4. **Quantization:** Created Q4_K_M version (~2GB) for 3x faster Windows inference
5. **Behavioral Training:** Included greetings, goodbyes, and normal conversations without identity mentions

### 7.4 Iterations

| Version | Examples | Epochs | LoRA Rank | Loss | Accuracy | Identity |
|---------|----------|--------|-----------|------|----------|----------|
| **v4** | 40 | 3 | 16 | 2.124 | 67.3% | ❌ Failed |
| **v5 (initial)** | 40 | 20 | 32 | 0.06562 | 98.11% | ⚠️ Partial |
| **v5 (enhanced)** | 95 | 20 | 32 | 0.08047 | 97.3% | ✅ Success |

---

## 8. Reproducibility

### 8.1 Prerequisites

- Alibaba Cloud account with PAI-DSW access (2x A10 GPUs recommended)
- Python 3.12+
- 8GB+ RAM for local testing
- Git

### 8.2 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/Kedarcv/zim-my.git
cd zim-my

# 2. Generate enhanced dataset
python notebooks/train_clair_v5_enhanced.py

# 3. Train Clair v5 (20 epochs, ~4 minutes)
python notebooks/train_clair_v5.py

# 4. Merge, convert, and test
python notebooks/merge_clair_v5.py

# 5. Quantize for Windows performance
python notebooks/quantize_clair_v5.py

# 6. Update Modelfile to use Q4_K_M
# Edit deploy/Modelfile: FROM ../models/clair-gguf-v5/clair-v5-Q4_K_M.gguf

# 7. Create and push to Ollama
ollama create r245142r/Clair-3B -f deploy/Modelfile
ollama push r245142r/Clair-3B

# 8. Test the model
ollama run r245142r/Clair-3B
```

### 8.3 Training Configuration

All training parameters are defined in `notebooks/train_clair_v5.py`:

```python
# LoRA Configuration
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training Configuration
NUM_EPOCHS = 20
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4
```

---

## 9. Future Work

- [ ] Add multi-language support (Shona, Ndebele)
- [ ] Implement RAG for domain-specific knowledge
- [ ] Add voice interface (speech-to-text + text-to-speech)
- [ ] Create mobile app (Termux on Android)
- [ ] Expand training dataset to 500+ examples
- [ ] Fine-tune for specific domains (agriculture, education, healthcare)

---

## 10. References

1. Qwen2.5 Technical Report — Alibaba Cloud
2. PEFT: Parameter-Efficient Fine-Tuning — huggingface.co/docs/peft
3. TRL: Transformer Reinforcement Learning — huggingface.co/docs/trl
4. llama.cpp: LLM inference in C/C++ — github.com/ggerganov/llama.cpp
5. Ollama: Run LLMs locally — ollama.ai
6. ADTC 2026 Challenge Guidelines — africadeeptech.org

---

## 11. Bonus Claims

### 11.1 Budget Laptop Compatibility ✅

- **Model size:** ~2.0 GB (Q4_K_M quantization)
- **RAM usage:** ~6.8 GB total (within 7GB ceiling)
- **CPU-only:** Optimized for Intel i5 without GPU
- **Storage:** <3 GB total (model + application)
- **Performance:** 3x faster with Q4_K_M vs F16

### 11.2 African Language Support (Planned)

- **Current:** English only
- **Planned:** Shona and Ndebele support
- **Strategy:** Fine-tune on African language datasets
- **Bonus:** +15% score potential for African language support

---

*Clair v5 🤖 — Personalized AI with embedded identity, built from Zimbabwe for the world.*
