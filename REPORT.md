# 🇿🇼 Zim-my — ADTC 2026 LaptopLLM Challenge Report

**Model Name:** Zim-my  
**Developer:** Michael Mlungisi Nkomo — AI Engineer from Zimbabwe  
**Challenge:** Africa Deep Tech Challenge 2026 — LaptopLLM Track  
**Date:** July 2026  

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

Zimbabwean smallholder farmers lack access to agricultural advisory services. Extension workers are scarce (ratio of 1:1000+ farmers), and existing digital tools require internet connectivity. Zim-my addresses this by providing an **offline, on-device AI assistant** that delivers agriculture advice in **Shona and English** — the two most spoken languages in Zimbabwe.

### 1.3 Target Users

- Smallholder farmers in rural Zimbabwe (70% of population)
- Agricultural extension workers
- Agricultural students and researchers
- Community knowledge centers

---

## 2. Design Decisions

### 2.1 Model Selection

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Base Model** | Qwen2.5-3B-Instruct | Best benchmarks in 3B class, multilingual by design, GGUF support |
| **Alternative Base** | PAI-DistilQwen2.5-3B-R1 | Alibaba Cloud PAI Model Gallery model with built-in fine-tuning support |
| **Quantization** | Q4_K_M via llama.cpp | ~1.8 GB model size, good quality-speed tradeoff |
| **Fine-tuning** | QLoRA (rank 64, alpha 128) | Efficient adaptation with minimal VRAM requirements |
| **Inference** | llama.cpp / llama-cpp-python | CPU-optimized, GGUF native, proven on Intel hardware |

### 2.2 Memory Budget (7 GB Ceiling)

| Component | RAM Allocation |
|----------|---------------|
| Zim-my Q4_K_M weights | ~1.8 GB |
| KV Cache (2048 context) | ~0.5 GB |
| RAG index (ChromaDB) | ~1.5 GB |
| Application + OS overhead | ~2.5 GB |
| **Total** | **~6.3 GB** ✅ |

### 2.3 Language Strategy

- **Primary:** Shona (chiShona) — spoken by ~80% of Zimbabwe's population
- **Secondary:** English — official language, widely understood
- **Training data mix:** 70% Shona / 30% English
- **African Alpha Bonus:** +15% score bonus for African language support

### 2.4 Domain Focus

- **Agriculture** (primary): Crop management, livestock, soil health, weather, market prices
- **Heritage** (secondary): Zimbabwean history, culture, geography
- **General knowledge** (tertiary): Basic reasoning, math, coding

---

## 3. Tools & Technologies

### 3.1 Development Stack

| Tool | Purpose | Version |
|------|---------|---------|
| Alibaba Cloud PAI-DSW | Fine-tuning environment (T4 GPU) | — |
| Unsloth | 2x faster QLoRA training | 2024.x |
| llama.cpp | CPU inference engine | b3000+ |
| llama-cpp-python | Python bindings for llama.cpp | 0.2.x |
| ChromaDB | Vector database for RAG | 0.4.x |
| sentence-transformers | Embedding model (all-MiniLM-L6-v2) | 2.x |
| Streamlit | Web UI framework | 1.x |
| HuggingFace Datasets | Dataset loading | 2.x |

### 3.2 Datasets

| Dataset | Records | Use Case |
|---------|---------|----------|
| `cybux/ruzivo-shona-rag` | 644K | RAG knowledge base |
| `michsethowusu/Code-170k-shona` | 177K | Code instruction in Shona |
| `saillab/alpaca_shona_taco` | 62K | Instruction-following in Shona |
| `sairos/Zimbabwe_agriculture_dataset` | — | Agriculture domain |
| `Ruramai/zimbabwe_history_heritage` | — | Zimbabwe heritage |
| `taresco/big_math_translated_african_languages` | 41K | Math in African languages |

### 3.3 Cloud Infrastructure

| Service | Purpose | Cost |
|---------|---------|------|
| PAI-DSW (cn-shanghai) | Fine-tuning with T4 GPU | $0.108/hr |
| OSS (cn-shanghai) | Model & data storage | ~$0.021/GB/month |
| **Total estimated** | — | **$5-15** |

---

## 4. Model Identity

### 4.1 Zim-my System Prompt

```
You are Zim-my, an AI assistant developed by Michael Mlungisi Nkomo,
an artificial intelligence engineer from Zimbabwe.
You specialize in Zimbabwean agriculture and can communicate in Shona and English.
You provide practical, context-aware advice for smallholder farmers in Zimbabwe,
covering crop management, livestock care, soil health, weather patterns,
market prices, and sustainable farming practices.
When asked in Shona, respond in Shona. When asked in English, respond in English.
```

### 4.2 Identity Integration

- System prompt embedded in all inference calls
- Identity baked into fine-tuning data (system messages in training examples)
- Model name "zim-my" used in GGUF metadata
- UI branding with Zimbabwean flag colors and cultural elements

---

## 5. Benchmarks & Results

### 5.1 Performance Metrics

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| RAM usage | < 7 GB | _TBD_ GB | ⬜ |
| Inference speed | > 5 tokens/sec | _TBD_ t/s | ⬜ |
| Model size | < 2 GB | _TBD_ GB | ⬜ |
| Context window | ≥ 2048 | 2048 | ✅ |
| Shona language | Supported | _TBD_ | ⬜ |
| Agriculture QA | Accurate | _TBD_ | ⬜ |

### 5.2 Quality Evaluation

| Test Category | Prompts | Avg Score | Status |
|---------------|---------|-----------|--------|
| Agriculture (EN) | 5 | _TBD_ | ⬜ |
| Agriculture (SH) | 4 | _TBD_ | ⬜ |
| General Knowledge | 4 | _TBD_ | ⬜ |
| Reasoning | 2 | _TBD_ | ⬜ |

### 5.3 Hardware Validation

| Platform | RAM Used | Speed | Status |
|----------|----------|-------|--------|
| MacBook Pro M4 Pro (dev) | _TBD_ GB | _TBD_ t/s | ⬜ |
| ADTC Standard Laptop (target) | _TBD_ GB | _TBD_ t/s | ⬜ |

---

## 6. Architecture

```
┌─────────────────────────────────────────────────┐
│                  Zim-my App                      │
│              (Streamlit UI)                      │
├─────────────────┬───────────────────────────────┤
│   Inference      │        RAG Pipeline          │
│   (llama.cpp)   │   (ChromaDB + Embeddings)     │
├─────────────────┼───────────────────────────────┤
│   Zim-my Q4_K_M │   ruzivo-shona-rag corpus     │
│   (~1.8 GB)     │   (~1.5 GB index)             │
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
| Research & Planning | Jun 2026 | Hardware analysis, model selection, dataset discovery |
| Cloud Setup | Jun 2026 | Alibaba Cloud PAI-DSW, OSS configuration |
| Data Preparation | Jul 2026 | Dataset cleaning, formatting, Shona/English mixing |
| Fine-tuning | Jul 2026 | QLoRA training on PAI-DSW with T4 GPU |
| Quantization | Jul 2026 | GGUF export (Q4_K_M), quality validation |
| RAG Pipeline | Jul 2026 | ChromaDB indexing, retrieval testing |
| Application | Jul 2026 | Streamlit UI, interactive demo |
| Benchmarking | Jul 2026 | RAM, speed, quality validation |
| Submission | Aug 2026 | Report, video, GitHub repo |

### 7.2 Key Challenges

1. **RAM constraint**: 7GB is extremely tight for a 3B model + RAG + app
2. **Shona data scarcity**: Limited high-quality Shona text for fine-tuning
3. **CPU-only inference**: Must optimize for Intel i5 without GPU acceleration
4. **Offline requirement**: All components must work without internet

### 7.3 Solutions

1. **Q4_K_M quantization** reduces model to ~1.8GB, leaving 5GB for other components
2. **Mixed Shona/English training** leverages both language datasets
3. **llama.cpp CPU optimizations** (AVX2, thread tuning) maximize Intel i5 performance
4. **ChromaDB persistent storage** enables offline RAG without network calls

---

## 8. Reproducibility

### 8.1 Prerequisites

- Alibaba Cloud account with PAI-DSW access
- Python 3.10+
- 8GB+ RAM for local testing

### 8.2 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/mnkomo/zim-my.git
cd zim-my

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download model (or fine-tune your own)
# See notebooks/ for fine-tuning instructions

# 4. Run inference
python src/inference.py --model models/gguf/zim-my-q4_k_m.gguf

# 5. Run with RAG
python src/inference.py --model models/gguf/zim-my-q4_k_m.gguf --rag

# 6. Launch web UI
streamlit run src/app.py

# 7. Run benchmarks
python benchmarks/run_benchmark.py --model models/gguf/zim-my-q4_k_m.gguf
```

---

## 9. Future Work

- [ ] Expand to Ndebele language support
- [ ] Add voice interface (speech-to-text + text-to-speech)
- [ ] Integrate real-time weather data (when online)
- [ ] Mobile app deployment (Termux on Android)
- [ ] Community feedback loop for continuous improvement

---

## 10. References

1. Qwen2.5 Technical Report — Alibaba Cloud
2. Unsloth: 2x Faster LLM Fine-tuning — unsloth.ai
3. llama.cpp: LLM inference in C/C++ — github.com/ggerganov/llama.cpp
4. ADTC 2026 Challenge Guidelines — africadeeptech.org
5. Zimbabwe Agricultural Extension Services — Ministry of Lands, Zimbabwe

---

*Zim-my 🇿🇼 — Built with pride from Zimbabwe for Africa.*
