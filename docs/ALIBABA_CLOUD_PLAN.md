# Alibaba Cloud Strategy — Zim-my ADTC 2026

> Plan for fine-tuning **Zim-my** (Qwen2.5-3B) on Alibaba Cloud infrastructure, then deploying the quantized model for on-device inference within 7 GB RAM.

**Developer:** Michael Mlungisi Nkomo  
**PAI Workspace ID:** 1416757 (cn-shanghai)  
**PAI Model Gallery Model:** PAI-DistilQwen2.5-3B-R1 (Model ID: `model-x2m0l08m0q38nidd6s`)

---

## 🎯 Why Alibaba Cloud?

| Advantage | Detail |
|---|---|
| **Qwen is Alibaba's model** | Native ecosystem support, optimized tooling, ModelScope integration |
| **PAI Platform** | Full ML lifecycle: data labeling → training → deployment |
| **Cost-effective GPU** | NVIDIA T4/A10 instances cheaper than AWS/GCP for fine-tuning |
| **ModelScope Hub** | China's HuggingFace equivalent — hosts Qwen models natively |
| **Free tier available** | PAI-DSW has free trial hours for development |
| **Africa regions** | Potential for deployment closer to target users |

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  ALIBABA CLOUD (Training)                │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   PAI-DSW    │───▶│   PAI-DLC    │───▶│    OSS    │ │
│  │  (Notebook   │    │  (Distributed│    │  (Model   │ │
│  │  Development)│    │   Training)  │    │  Storage) │ │
│  └──────────────┘    └──────────────┘    └───────────┘ │
│         │                    │                  │       │
│  ┌──────▼──────┐    ┌───────▼──────┐    ┌─────▼─────┐ │
│  │  GPU: T4    │    │  GPU: A10    │    │  GGUF     │ │
│  │  (16GB)     │    │  (24GB)      │    │  Export   │ │
│  └─────────────┘    └──────────────┘    └───────────┘ │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼ Download GGUF
┌─────────────────────────────────────────────────────────┐
│              LOCAL LAPTOP (Inference)                     │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │  llama.cpp   │◀───│  Q4_K_M     │◀───│   RAG     │ │
│  │  (CPU-only)  │    │  GGUF Model │    │  (ChromaDB│ │
│  └──────────────┘    └──────────────┘    └───────────┘ │
│         │                                               │
│  ┌──────▼──────┐                                        │
│  │  Streamlit  │  ◀── Web UI for demo                   │
│  │  / Gradio   │                                        │
│  └─────────────┘                                        │
└─────────────────────────────────────────────────────────┘
```

---

## 🎨 PAI Model Gallery — Recommended Workflow

**PAI Model Gallery** provides a streamlined UI for fine-tuning Qwen models directly in the PAI console, with built-in SFT/LoRA support.

### Available Models

| Model | Model ID | Parameters | Use Case |
|---|---|---|---|
| **PAI-DistilQwen2.5-3B-R1** ⭐ | `model-x2m0l08m0q38nidd6s` | 3B | **Primary choice** — best balance of quality and size |
| PAI-DistilQwen2.5-7B-R1 | `model-ep3155xg4i9t84u462` | 7B | Larger alternative if RAM budget allows |

### PAI Model Gallery Fine-tuning Steps

1. **Navigate to PAI Model Gallery** in the PAI console
2. **Select PAI-DistilQwen2.5-3B-R1** (Model ID: `model-x2m0l08m0q38nidd6s`)
3. **Click "Fine-tune"** to launch the built-in SFT/LoRA UI
4. **Upload training data** (processed Shona + English agriculture datasets)
5. **Configure LoRA parameters**:
   - Rank: 64
   - Alpha: 128
   - Target modules: all linear layers
   - Dropout: 0.05
6. **Configure training**:
   - Epochs: 3
   - Batch size: 4
   - Gradient accumulation: 8
   - Learning rate: 2e-4
   - Max sequence length: 2048
7. **Launch training** and monitor progress
8. **Export fine-tuned model** to OSS
9. **Download and quantize** to GGUF format locally

### Advantages of PAI Model Gallery

- **No code required** — UI-driven fine-tuning workflow
- **Pre-configured environment** — no dependency management
- **Integrated with PAI-DSW** — can switch to notebooks for custom work
- **Cost-effective** — pay only for compute time used
- **Built-in monitoring** — training metrics and logs in the console

---

## 📊 Alibaba Cloud Services We'll Use

### 1. PAI-DSW (Data Science Workshop) — Development IDE

| Feature | Detail |
|---|---|
| **What** | Cloud-based Jupyter/VS Code IDE with GPU access |
| **Use for** | Data exploration, notebook development, testing fine-tuning scripts |
| **Pricing** | From **$0.108/hour** (pay-as-you-go) |
| **GPU option** | NVIDIA T4 (16 GB) — sufficient for QLoRA on 3B model |
| **Key benefit** | Pre-built PyTorch images, SSH access, OSS/NAS mounting |

### 2. PAI-DLC (Deep Learning Containers) — Training

| Feature | Detail |
|---|---|
| **What** | Managed distributed training service |
| **Use for** | Running QLoRA fine-tuning jobs at scale |
| **Pricing** | Pay-as-you-go per GPU-hour |
| **GPU option** | NVIDIA A10 (24 GB) — ideal for 3B QLoRA |
| **Key benefit** | Auto-scaling, preemptible instances (cheaper), container-based |

### 3. ECS GPU Instances — Alternative Training

| Instance | GPU | GPU Mem | vCPUs | RAM | Est. Cost (PAYG) |
|---|---|---|---|---|---|
| **ecs.gn6i-c4g1.xlarge** | NVIDIA T4 × 1 | 16 GB | 4 | 15 GB | ~$0.50/hr |
| **ecs.gn7i-c8g1.2xlarge** | NVIDIA A10 × 1 | 24 GB | 8 | 30 GB | ~$1.20/hr |
| **ecs.gn7i-c16g1.4xlarge** | NVIDIA A10 × 1 | 24 GB | 16 | 60 GB | ~$1.80/hr |
| **ecs.vgn6i-m4-vws.xlarge** | NVIDIA T4 × 1/4 | 4 GB | 4 | 23 GB | ~$0.30/hr |

> **Recommendation**: `ecs.gn6i-c4g1.xlarge` (T4, 16 GB) is the **sweet spot** — enough VRAM for QLoRA on Qwen2.5-3B at the lowest cost.

### 4. OSS (Object Storage Service) — Data & Model Storage

| Feature | Detail |
|---|---|
| **What** | S3-compatible object storage |
| **Use for** | Storing datasets, model checkpoints, GGUF files |
| **Pricing** | ~$0.021/GB/month (standard), ~$0.01/GB/month (infrequent) |
| **Est. need** | ~20 GB (datasets + models) = ~$0.42/month |

### 5. ModelScope — Model Hub

| Feature | Detail |
|---|---|
| **What** | Alibaba's model hub (like HuggingFace) |
| **Use for** | Downloading Qwen2.5-3B base model (faster from China servers) |
| **Pricing** | Free |
| **Key benefit** | Native Qwen hosting, faster downloads from Alibaba Cloud |

---

## 💰 Cost Estimate

### Option A: PAI-DSW Only (Simplest)

| Item | Hours | Rate | Cost |
|---|---|---|---|
| PAI-DSW (T4 GPU) — Development | 20 hrs | $0.108/hr | $2.16 |
| PAI-DSW (T4 GPU) — Fine-tuning | 8 hrs | $0.108/hr | $0.86 |
| PAI-DSW (T4 GPU) — Testing/Benchmarking | 10 hrs | $0.108/hr | $1.08 |
| OSS Storage (20 GB, 2 months) | — | $0.021/GB/mo | $0.84 |
| **Total** | | | **~$5** |

### Option B: ECS GPU Instance (More Control)

| Item | Hours | Rate | Cost |
|---|---|---|---|
| ecs.gn6i-c4g1.xlarge (T4) — Setup & Dev | 20 hrs | ~$0.50/hr | $10.00 |
| ecs.gn6i-c4g1.xlarge (T4) — Fine-tuning | 8 hrs | ~$0.50/hr | $4.00 |
| ecs.gn6i-c4g1.xlarge (T4) — Testing | 10 hrs | ~$0.50/hr | $5.00 |
| ESSD Disk (100 GB, 2 months) | — | ~$0.0004/GB/hr | ~$5.76 |
| OSS Storage (20 GB, 2 months) | — | $0.021/GB/mo | $0.84 |
| **Total** | | | **~$26** |

### Option C: Hybrid (Best Value)

| Item | Hours | Rate | Cost |
|---|---|---|---|
| PAI-DSW (T4) — Development & Testing | 30 hrs | $0.108/hr | $3.24 |
| ECS gn6i (T4) — Fine-tuning only | 8 hrs | ~$0.50/hr | $4.00 |
| OSS Storage (20 GB, 2 months) | — | $0.021/GB/mo | $0.84 |
| **Total** | | | **~$8** |

> 💡 **Preemptible/Spot instances** can reduce ECS costs by **60-90%** — bringing Option B down to ~$10-15.

---

## 📋 Step-by-Step Execution Plan

### Phase 1: Environment Setup (Day 1-2)

```
1. Create Alibaba Cloud account
2. Activate PAI service
3. Create OSS bucket for project storage
4. Set up PAI-DSW instance with PyTorch image
5. Install dependencies:
   - unsloth
   - transformers
   - peft
   - bitsandbytes
   - llama-cpp-python
   - datasets
```

### Phase 2: Data Preparation (Day 3-5)

```
1. Download datasets to PAI-DSW:
   - cybux/ruzivo-shona-rag (644K records)
   - saillab/alpaca_shona_taco (62K)
   - sairos/Zimbabwe_agriculture_dataset
   - Ruramai/zimbabwe_history_heritage
   - michsethowusu/Code-170k-shona (subset)

2. Clean and format datasets:
   - Convert to instruction-following format
   - Mix Shona + English (70/30 ratio)
   - Create train/val/test splits
   - Upload processed data to OSS

3. Build RAG corpus:
   - Index ruzivo-shona-rag for retrieval
   - Test retrieval quality
```

### Phase 3: Fine-tuning (Day 6-10)

```
1. Load Qwen2.5-3B-Instruct in 4-bit (QLoRA)
2. Configure LoRA:
   - rank: 64
   - alpha: 128
   - target_modules: all linear layers
   - dropout: 0.05
3. Training config:
   - epochs: 3
   - batch_size: 4 (gradient_accumulation: 8)
   - learning_rate: 2e-4
   - max_seq_length: 2048
   - warmup_ratio: 0.03
4. Train on PAI-DLC or ECS GPU instance
5. Monitor with TensorBoard
6. Save LoRA adapters to OSS
```

### Phase 4: Quantization & Export (Day 11-12)

```
1. Merge LoRA adapters into base model
2. Export to GGUF format:
   - Q4_K_M (primary — best quality/size ratio)
   - Q5_K_M (backup — slightly larger, better quality)
   - Q3_K_M (fallback — smallest, if RAM is tight)
3. Upload GGUF files to OSS
4. Download to local laptop
```

### Phase 5: Local Inference & RAG (Day 13-18)

```
1. Set up llama.cpp locally
2. Load Q4_K_M model
3. Build RAG pipeline:
   - ChromaDB for vector storage
   - Sentence-transformers for embeddings (small model)
   - Retrieval + context injection
4. Benchmark:
   - Tokens/second on target hardware
   - RAM usage (must stay < 7 GB)
   - Quality evaluation on domain tasks
5. Build application UI (Streamlit/Gradio)
```

### Phase 6: Benchmarking & Packaging (Day 19-25)

```
1. Run ADTC benchmarks on target hardware spec
2. Document memory usage at every stage
3. Write REPORT.md
4. Record 2-minute demo video
5. Package repo for Gate 1 submission
6. Submit on DevPost before Aug 25
```

---

## 🔧 Technical Setup Commands

### PAI-DSW Instance Setup

```bash
# After launching PAI-DSW with PyTorch image:

# Install fine-tuning dependencies
pip install unsloth[colab-new] \
    "xformers==0.0.28.post3" \
    "trl<0.9.0" \
    peft accelerate bitsandbytes \
    datasets \
    llama-cpp-python \
    chromadb sentence-transformers \
    streamlit gradio

# Verify GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Download base model from ModelScope (faster on Alibaba Cloud)
pip install modelscope
python -c "
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen2.5-3B-Instruct')
print(f'Model downloaded to: {model_dir}')
"
```

### ECS GPU Instance Setup

```bash
# Launch Ubuntu 22.04 on ecs.gn6i-c4g1.xlarge
# SSH into instance

# Install NVIDIA drivers + CUDA
sudo apt update && sudo apt install -y nvidia-driver-535
sudo reboot

# Install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create environment
conda create -n zim-my python=3.10 -y
conda activate zim-my

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install unsloth transformers peft accelerate bitsandbytes datasets
pip install llama-cpp-python chromadb sentence-transformers streamlit

# Install Alibaba Cloud CLI
pip install aliyun-cli ossutil
```

### OSS Data Transfer

```bash
# Configure OSS
ossutil config

# Upload processed datasets
ossutil cp -r data/processed/ oss://zim-my-data/datasets/

# Upload model checkpoints
ossutil cp -r models/ oss://zim-my-data/models/

# Download GGUF to local laptop
ossutil cp oss://zim-my-data/models/zim-my-q4_k_m.gguf ./models/gguf/
```

---

## 🗺️ Alibaba Cloud Region Selection

| Region | Latency to Zimbabwe | GPU Availability | Notes |
|---|---|---|---|
| **South Africa (Johannesburg)** | ⭐ Lowest | Limited | Best for final deployment demo |
| **UAE (Dubai)** | Medium | Good | Good balance of latency + availability |
| **Singapore** | Medium-High | Excellent | Best GPU availability + pricing |
| **Germany (Frankfurt)** | Medium | Good | Good for European judges |

> **Recommendation**: Use **Singapore** for training (best GPU availability/pricing), then demo from **South Africa** if available.

---

## 🔄 Workflow: Cloud ↔ Local

```
Alibaba Cloud                          Local MacBook M4 Pro
─────────────                          ────────────────────
PAI-DSW / ECS GPU                      
  │                                    
  ├── Download Qwen2.5-3B              
  ├── Download datasets                
  ├── Fine-tune (QLoRA)                
  ├── Merge adapters                   
  ├── Export GGUF (Q4_K_M)            
  │                                    
  └── Upload to OSS ──────────────────▶ Download GGUF
                                        │
                                        ├── Load in llama.cpp
                                        ├── Build RAG pipeline
                                        ├── Benchmark (< 7 GB RAM)
                                        ├── Build UI
                                        └── Record demo video
```

---

## ⚠️ Key Considerations

### Memory Budget Validation

Before submission, validate on actual ADTC target hardware:

```python
import psutil
import subprocess

# Monitor RAM during inference
process = psutil.Process()
mem_mb = process.memory_info().rss / 1024 / 1024
print(f"Process RAM: {mem_mb:.1f} MB")

total_used = psutil.virtual_memory().used / 1024 / 1024 / 1024
print(f"Total system RAM used: {total_used:.2f} GB")
assert total_used < 7.0, f"EXCEEDED 7 GB LIMIT: {total_used:.2f} GB"
```

### Cost Optimization Tips

1. **Use preemptible instances** — 60-90% cheaper, fine for training (can resume from checkpoints)
2. **Stop PAI-DSW when not in use** — billed by the minute
3. **Use scheduled shutdown** — auto-stop after idle period
4. **Free trial** — PAI offers free trial hours for new accounts
5. **OSS infrequent access** — cheaper for model checkpoints you don't access daily

### Risk Mitigation

| Risk | Mitigation |
|---|---|
| GPU instance unavailable | Try multiple regions; use PAI-DSW as fallback |
| Training OOM on T4 (16 GB) | Reduce batch size, use gradient checkpointing, lower LoRA rank |
| Model too large after merge | Use Q3_K_M quantization instead of Q4_K_M |
| RAM exceeds 7 GB at inference | Reduce context window, use smaller RAG index, Q3_K_M |
| Network costs (data transfer) | Keep data in same region; use internal endpoints |

---

## 📅 Timeline (58 Days to Gate 1)

| Week | Phase | Cloud Usage | Est. Cost |
|---|---|---|---|
| **Week 1** (Jun 28–Jul 4) | Setup + Data Prep | PAI-DSW (10 hrs) | ~$1 |
| **Week 2** (Jul 5–11) | Fine-tuning | ECS T4 (8 hrs) + PAI-DSW (5 hrs) | ~$5 |
| **Week 3** (Jul 12–18) | Quantize + Local Setup | PAI-DSW (5 hrs) | ~$0.50 |
| **Week 4** (Jul 19–25) | RAG + Inference | Local only | $0 |
| **Week 5** (Jul 26–Aug 1) | UI + Benchmarking | Local only | $0 |
| **Week 6** (Aug 2–8) | Polish + REPORT.md | Local only | $0 |
| **Week 7** (Aug 9–15) | Demo Video + Testing | PAI-DSW (5 hrs) for re-training if needed | ~$0.50 |
| **Week 8** (Aug 16–25) | Final Submission | Local only | $0 |
| | | **Total Estimated** | **~$7–15** |

---

## 🚀 Quick Start Checklist

- [ ] Create Alibaba Cloud account at [alibabacloud.com](https://www.alibabacloud.com)
- [ ] Activate PAI service in console
- [ ] Create OSS bucket: `zim-my-adtc-2026`
- [ ] Launch PAI-DSW instance (T4 GPU, PyTorch image)
- [ ] Clone this repo into DSW
- [ ] Run `scripts/setup_alibaba_cloud.sh` (TODO: create)
- [ ] Download Qwen2.5-3B from ModelScope
- [ ] Download Zimbabwean datasets
- [ ] Start fine-tuning notebook

---

## 📚 References

- [PAI Documentation](https://www.alibabacloud.com/help/en/pai/)
- [PAI-DSW Overview](https://www.alibabacloud.com/help/en/pai/user-guide/dsw-overview)
- [PAI-DLC Training](https://www.alibabacloud.com/help/machine-learning-platform-for-ai/latest/container-training-dlc)
- [GPU ECS Instances](https://www.alibabacloud.com/help/en/ecs/user-guide/gpu-accelerated-compute-optimized-and-vgpu-accelerated-instance-families)
- [OSS Documentation](https://www.alibabacloud.com/help/en/oss/)
- [ModelScope](https://modelscope.cn/)
- [Unsloth Documentation](https://docs.unsloth.ai/)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [ADTC 2026 Challenge](https://africadeeptech.org/challenge-2026)
- [DevPost Submission](https://adtc-2026.devpost.com/)
