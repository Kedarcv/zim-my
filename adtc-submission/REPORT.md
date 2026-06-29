# Technical Report — Clair v5: Personalized AI Assistant with Embedded Identity

**Team ID:** clair-zimbabwe  
**Domain:** coding_assistants  
**Model:** Clair-v5-Q4_K_M

---

## Problem

Developers and students in resource-constrained environments lack access to personalized AI coding assistance. Cloud-based solutions require constant internet connectivity and expensive API costs, while existing local models either exceed hardware limits or lack consistent identity and behavioral coherence.

Clair addresses this by providing a **personalized AI assistant with embedded identity** that runs entirely offline on budget laptops (Intel i5, 8GB DDR4, CPU-only). The model maintains consistent identity across all interactions, handles natural conversation flow (greetings, goodbyes, clarifications), and provides direct coding assistance without requiring internet connectivity or external APIs.

**Target users:** Students, developers, and educators in regions with limited internet access or high API costs, particularly in sub-Saharan Africa where connectivity remains unreliable and expensive.

---

## Design Decisions

- **Base model:** Qwen2.5-3B-Instruct — selected for best-in-class benchmarks at 3B parameter scale, strong instruction-following capabilities, and native GGUF support via llama.cpp
- **Quantization:** Q4_K_M chosen for optimal balance of quality and memory footprint (~1.8 GB model size, ~6.8 GB total RAM usage)
- **Fine-tuning method:** LoRA (rank 32, alpha 64) with 4-bit quantization — enables efficient adaptation with only 1.9% trainable parameters while maintaining model quality
- **Training strategy:** Enhanced dataset (95 examples) with heavy identity emphasis (30+ identity questions) and extended training (20 epochs) to override base model's strong priors about being created by Alibaba Cloud
- **Alternatives considered:**
  - Q8_0 quantization: Exceeded 7GB RAM ceiling (~4.5 GB model + overhead)
  - Q2_K quantization: Degraded output quality and identity consistency
  - Full fine-tuning: Required excessive VRAM and training time
  - System prompt identity: Failed to maintain consistency across conversations

---

## Constraints

- **Target hardware:** Intel i5 10th-12th gen (4 cores, no hyperthreading), 8GB DDR4 RAM, integrated GPU only, Ubuntu 22.04 LTS
- **RAM ceiling:** 7GB usable after OS overhead — model must fit within this limit including KV cache and application overhead
- **No GPU acceleration:** Pure CPU inference via llama.cpp — requires aggressive quantization while maintaining quality
- **Offline operation:** Zero external network dependencies during inference — all model weights and dependencies must be self-contained
- **Identity embedding challenge:** Qwen2.5 has strong priors about being created by Alibaba Cloud, requiring 30+ identity examples and 20 epochs to override (vs typical 3-5 epochs)

---

## Benchmarks

| Metric | Value |
|---|---|
| Machine | PAI-DSW (2x NVIDIA A10) / Windows PC (Intel i5, 8GB RAM) |
| RAM at peak | ~6.8 GB (Q4_K_M on CPU) |
| Model size | 1.8 GB (Q4_K_M), 5.75 GB (F16) |
| Context window | 4096 tokens |
| Training loss | 0.08047 |
| Token accuracy | 97.3% |
| Identity recognition | 100% (tested on 30+ identity questions) |
| Generation speed | ~5-8 t/s (CPU, Q4_K_M), ~30-50 t/s (GPU, Q4_K_M) |
| Thermal throttling | None observed during 10-minute inference sessions |

**Key achievements:**
- ✅ Runs within 7GB RAM ceiling on budget laptop hardware
- ✅ Embedded identity (trained into model weights, not system prompt)
- ✅ Natural conversation flow (greetings, goodbyes, clarifications)
- ✅ 3x faster inference with Q4_K_M vs F16 quantization
- ✅ 26x improvement in training loss (2.124 → 0.08047) from v4 to v5

These are self-reported development benchmarks. Official scores are measured by the ADTC profiler on the standard evaluation machine.
