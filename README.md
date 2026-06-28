# Zim-my — ADTC 2026 Challenge

> **On-Device AI for the Hardware Africa Actually Has**
> Building a fine-tuned, quantized LLM that runs within 7 GB RAM on budget laptops — with Shona language support.

**Developer:** Michael Mlungisi Nkomo, AI Engineer from Zimbabwe  
**Model:** Zim-my   
**Cloud Platform:** Alibaba Cloud PAI-DSW (cn-shanghai, workspace ID: 1416757)

---

## 🏆 Challenge: Africa Deep Tech Challenge 2026

| Detail | Info |
|---|---|
| **Challenge** | [The LaptopLLM Challenge](https://africadeeptech.org/challenge-2026) |
| **Prize** | $20,000 total |
| **Gate 1 Deadline** | **August 25, 2026** |
| **Memory Ceiling** | **7 GB RAM** — exceeding = instant disqualification (Stotal = 0) |
| **African Language Bonus** | **+15%** on panel score |
| **Submission** | [DevPost](https://adtc-2026.devpost.com/) |

### ADTC Standard Laptop (Target Hardware)

| Spec | Requirement |
|---|---|
| **CPU** | Intel Core i5 10th–12th gen or AMD Ryzen 5 3000–5000 (x86-64) |
| **Memory** | 8 GB DDR4 (7 GB ceiling for model + app) |
| **Graphics** | Intel UHD / Iris Xe or AMD Radeon integrated — **no discrete GPU** |
| **Storage** | 256 GB SSD |
| **OS** | Ubuntu 22.04 LTS |
| **Market Range** | $400–$500 new · $150–$250 refurbished |

### Problem Domains

- Math & Scientific Reasoning
- Healthcare & Medical
- **Agriculture** ← Our focus
- Creative Writing
- Coding Assistants
- Corporate / Enterprise
- Autonomous AI Agents

### Three-Gate Funnel

| Gate | Deadline | Deliverables |
|---|---|---|
| **Gate 1** — Submission Package | Aug 25, 2026 | Open-source repo, REPORT.md, 2-min demo video, benchmarks |
| **Gate 2** — Activities & Audit | Sep 8–29, 2026 | 30-min technical Q&A, prompt responses, optional updated report |
| **Gate 3** — Final Package | Oct 17, 2026 | Pitch deck (10 slides), live session, technical setup verification |

---

## 🖥️ Development Machine

| Spec | MacBook Pro M4 Pro |
|---|---|
| **Chip** | Apple M4 Pro (14-core: 10P + 4E) |
| **RAM** | 24 GB unified memory |
| **GPU** | Apple Neural Engine + Metal |
| **Storage** | 460 GB SSD |
| **OS** | macOS |

> The M4 Pro is used for **development and fine-tuning only**. The final model must run within 7 GB on the ADTC Standard Laptop (Intel i5, no GPU, CPU-only inference).

---

## 🧠 Model Selection

### Comparison of Candidate Models

| Model | Params | Quantized Size (Q4) | RAM at Inference | Benchmark Quality | Fine-tune Feasibility |
|---|---|---|---|---|---|
| **Qwen2.5-3B-Instruct** ⭐ | 3B | ~1.8 GB | ~3–4 GB | ⭐⭐⭐⭐⭐ | ✅ Easy on M4 Pro |
| Qwen2.5-1.5B-Instruct | 1.5B | ~0.9 GB | ~2–3 GB | ⭐⭐⭐⭐ | ✅ Very easy |
| Llama-3.2-3B-Instruct | 3B | ~1.8 GB | ~3–4 GB | ⭐⭐⭐⭐ | ✅ Easy |
| Llama-3.2-1B-Instruct | 1B | ~0.6 GB | ~1.5–2 GB | ⭐⭐⭐ | ✅ Very easy |
| Phi-3-mini (3.8B) | 3.8B | ~2.2 GB | ~4–5 GB | ⭐⭐⭐⭐ | ⚠️ Tighter |
| Gemma-2-2B | 2B | ~1.2 GB | ~2.5–3.5 GB | ⭐⭐⭐⭐ | ✅ Easy |

### 🥇 Winner: Qwen2.5-3B-Instruct / PAI-DistilQwen2.5-3B-R1 (Q4_K_M quantized)

**Why Qwen2.5-3B:**

1. **Best benchmarks in its class** — outperforms Llama-3.2-3B on MMLU, math, and coding
2. **Multilingual by design** — trained on massive multilingual corpus including African languages
3. **~1.8 GB quantized** — leaves 5+ GB for KV cache, RAG, and application overhead
4. **Excellent fine-tuning ecosystem** — works with Unsloth, LoRA, and llama.cpp
5. **GGUF support** — runs natively on CPU via llama.cpp (target hardware has no GPU)
6. **PAI Model Gallery** — available as PAI-DistilQwen2.5-3B-R1 (Model ID: `model-x2m0l08m0q38nidd6s`) with built-in SFT/LoRA fine-tuning UI on Alibaba Cloud PAI

### Memory Budget (7 GB Ceiling)

| Component | RAM |
|---|---|
| Qwen2.5-3B Q4_K_M weights | ~1.8 GB |
| KV Cache (2048 context) | ~0.5 GB |
| RAG index (FAISS/ChromaDB) | ~1.5 GB |
| Application + OS overhead | ~2.5 GB |
| **Total** | **~6.3 GB** ✅ |

---

## 🇿🇼 Zimbabwean & African Datasets

### Shona Language (Primary Zimbabwe Language)

| Dataset | Records | Use Case |
|---|---|---|
| [`michsethowusu/Code-170k-shona`](https://huggingface.co/datasets/michsethowusu/Code-170k-shona) | 177K | Code instruction in Shona |
| [`cybux/ruzivo-shona-rag`](https://huggingface.co/datasets/cybux/ruzivo-shona-rag) | 644K | **RAG knowledge base** — perfect for retrieval |
| [`saillab/alpaca_shona_taco`](https://huggingface.co/datasets/saillab/alpaca_shona_taco) | 62K | Instruction-following in Shona |
| [`Kittech/mixed_shona_dataset`](https://huggingface.co/datasets/Kittech/mixed_shona_dataset) | 239 | Mixed Shona tasks |
| [`badrex/shona-speech`](https://huggingface.co/datasets/badrex/shona-speech) | 17.6K | Speech data |

### Zimbabwe-Specific

| Dataset | Use Case |
|---|---|
| [`Ruramai/zimbabwe_history_heritage`](https://huggingface.co/datasets/Ruramai/zimbabwe_history_heritage) | Zimbabwe history & cultural heritage |
| [`Ruramai/zimbabwe_history_and_heritage`](https://huggingface.co/datasets/Ruramai/zimbabwe_history_and_heritage) | Heritage knowledge base |
| [`sairos/Zimbabwe_agriculture_dataset`](https://huggingface.co/datasets/sairos/Zimbabwe_agriculture_dataset) | **Agriculture domain** — great for challenge |
| [`electricsheepafrica/africa-zimbabwe-acute-food-insecurity`](https://huggingface.co/datasets/electricsheepafrica/africa-zimbabwe-acute-food-insecurity) | Food security data |
| [`electricsheepafrica/africa-unesco-data-for-zimbabwe`](https://huggingface.co/datasets/electricsheepafrica/africa-unesco-data-for-zimbabwe) | UNESCO development data |
| [`electricsheepafrica/africa-fewsnet-staple-food-price-data`](https://huggingface.co/datasets/electricsheepafrica/africa-fewsnet-staple-food-price-data) | Food price data |

### Broader African (Multilingual Bonus)

| Dataset | Records | Use Case |
|---|---|---|
| [`taresco/big_math_translated_african_languages`](https://huggingface.co/datasets/taresco/big_math_translated_african_languages) | 41K | Math reasoning in African languages |
| [`taresco/open_math_instruct_v2_translated_african_languages`](https://huggingface.co/datasets/taresco/open_math_instruct_v2_translated_african_languages) | 30.2K | Math instruction tuning |
| [`Aletheia-ng/african_languages_*`](https://huggingface.co/datasets/Aletheia-ng) | 2.22M | Massive multilingual African corpus |

---

## 🎯 Strategy: Agriculture Advisory in Shona + English

### Architecture

```
Zim-my (fine-tuned from Qwen2.5-3B-Instruct / PAI-DistilQwen2.5-3B-R1)
    │
    ├── Fine-tune with QLoRA (Unsloth) on Alibaba Cloud PAI-DSW
    │   └── Shona + English agriculture/healthcare datasets
    │
    ├── Quantize to Q4_K_M (llama.cpp)
    │   └── Target: ~1.8 GB model size
    │
    ├── RAG Pipeline
    │   └── ChromaDB + sentence-transformers over ruzivo-shona-rag (644K records)
    │
    ├── Inference Engine
    │   └── llama.cpp / llama-cpp-python (CPU-only, 4 threads)
    │
    └── Application Layer
        └── Streamlit web UI for agriculture advisory
```

### Model Identity

**System Prompt:**
> "You are Zim-my, an AI assistant developed by Michael Mlungisi Nkomo, an artificial intelligence engineer from Zimbabwe. You specialize in Zimbabwean agriculture and can communicate in Shona and English."

### Fine-tuning Plan

1. **Tool**: [Unsloth](https://github.com/unslothai/unsloth) — 2x faster, 80% less memory than standard LoRA
2. **Method**: QLoRA (4-bit quantized base + LoRA adapters)
3. **Training time**: ~2–4 hours for 10K examples on M4 Pro MPS
4. **Datasets**: Combine Shona agriculture + English agriculture + Zimbabwe heritage

---

## 📁 Project Structure (Planned)

```
AI/
├── README.md                  # This file
├── REPORT.md                  # ADTC submission report
├── notebooks/
│   ├── 01_data_prep.ipynb     # Dataset exploration & preparation
│   ├── 02_finetune.ipynb      # QLoRA fine-tuning with Unsloth
│   ├── 03_quantize.ipynb      # GGUF quantization
│   └── 04_benchmark.ipynb     # Performance benchmarking
├── data/
│   ├── raw/                   # Downloaded datasets
│   ├── processed/             # Cleaned & formatted data
│   └── rag/                   # RAG corpus
├── models/
│   ├── base/                  # Base model weights
│   ├── lora/                  # LoRA adapters
│   └── gguf/                  # Quantized GGUF models
├── src/
│   ├── inference.py           # llama.cpp inference wrapper
│   ├── rag.py                 # RAG pipeline
│   └── app.py                 # Application UI
├── benchmarks/
│   └── results/               # Benchmark results
├── scripts/
│   ├── download_model.sh      # Model download script
│   ├── download_data.sh       # Dataset download script
│   └── setup.sh               # Environment setup
└── docs/
    └── architecture.md        # Architecture documentation
```

---

## 📋 Gate 1 Deliverables Checklist

- [ ] Open-source GitHub repo (ADTC 2026 submission template)
- [ ] REPORT.md — problem definition, constraints, design decisions, tools & benchmarks
- [ ] Screenshots or short video clips showing model running
- [ ] 2-minute video — solution and development journey
- [ ] Bonus claims: African language support (Shona) / budget laptop

---

## License

TBD — Will be open-source per challenge requirements.
