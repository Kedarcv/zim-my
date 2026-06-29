# Clair v3 — Deployment Guide


### Available Quantizations

| File | Size | Quality | Use Case |
|------|------|---------|----------|
| `clair-v3-Q4_K_M.gguf` | 1.80 GB | Good | **Default** — best balance |
| `clair-v3-Q5_K_M.gguf` | 2.07 GB | Better | When you have extra RAM |
| `clair-v3-Q3_K_M.gguf` | 1.48 GB | Acceptable | Low-memory environments |
| `clair-v3-float16.gguf` | 6.17 GB | Best | Maximum quality (needs 8GB+ VRAM) |

---

## Option 1: Ollama (Recommended)

### Prerequisites

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh
```

### Deploy

```bash
# On PAI-DSW:
cd /mnt/workspace/zim-my
git pull origin main
bash deploy/deploy_ollama.sh
```

### Manual Setup

```bash
# Copy GGUF to a working directory
mkdir -p ~/clair-deploy
cp /mnt/workspace/zim-my/models/clair-gguf-v3/clair-v3-Q4_K_M.gguf ~/clair-deploy/
cp deploy/Modelfile ~/clair-deploy/
cd ~/clair-deploy

# Create the model
ollama create clair -f Modelfile

# Run it
ollama run clair
```

### API Usage

```bash
# Chat completion
curl http://localhost:11434/api/chat -d '{
  "model": "clair",
  "messages": [
    {"role": "user", "content": "Hello! Who are you?"}
  ],
  "stream": false
}'

# Generate completion
curl http://localhost:11434/api/generate -d '{
  "model": "clair",
  "prompt": "What is the capital of Zimbabwe?",
  "stream": false
}'
```

### Python API

```python
import requests

response = requests.post("http://localhost:11434/api/chat", json={
    "model": "clair",
    "messages": [
        {"role": "user", "content": "Hello! Who are you?"}
    ],
    "stream": False
})

print(response.json()["message"]["content"])
```

---

## Option 2: llama.cpp (Direct)

### Run with llama-server

```bash
# Start an OpenAI-compatible API server
/root/.unsloth/llama.cpp/build/bin/llama-server \
    --model /mnt/workspace/zim-my/models/clair-gguf-v3/clair-v3-Q4_K_M.gguf \
    --n-gpu-layers 999 \
    --ctx-size 4096 \
    --port 8080 \
    --host 0.0.0.0

# Then use OpenAI-compatible API
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [
            {"role": "system", "content": "You are Clair, a helpful AI assistant."},
            {"role": "user", "content": "Hello!"}
        ],
        "max_tokens": 256,
        "temperature": 0.7
    }'
```

### Run with llama-cli

```bash
/root/.unsloth/llama.cpp/build/bin/llama-cli \
    --model /mnt/workspace/zim-my/models/clair-gguf-v3/clair-v3-Q4_K_M.gguf \
    --n-gpu-layers 999 \
    --ctx-size 4096 \
    --conversation \
    --prompt "You are Clair, a helpful AI assistant."
```

---

## Option 3: llama-cpp-python

```python
from llama_cpp import Llama

llm = Llama(
    model_path="/mnt/workspace/zim-my/models/clair-gguf-v3/clair-v3-Q4_K_M.gguf",
    n_gpu_layers=-1,       # All layers on GPU
    n_ctx=4096,            # Context window
    verbose=False,
)

response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are Clair, a helpful AI assistant created by Michael Mlungisi Nkomo from Zimbabwe."},
        {"role": "user", "content": "Hello! Who are you?"}
    ],
    max_tokens=256,
    temperature=0.7,
    stop=["\n\n", "User:", "Human:", "<|im_end|>"],
)

print(response["choices"][0]["message"]["content"])
```

---

## Prompting Best Practices

### ✅ Do

- **Keep system prompts concise**: `"You are Clair, a helpful AI assistant created by Michael Mlungisi Nkomo from Zimbabwe."`
- **Use `max_tokens`**: Limit responses to prevent repetition (100-512 tokens)
- **Use `stop` tokens**: `["\n\n", "User:", "Human:", "<|im_end|>"]`
- **Use lower temperature for facts**: `temperature=0.3` for factual questions
- **Use higher temperature for creativity**: `temperature=0.7-0.9` for open-ended chat

### ❌ Don't

- **Don't use overly long system prompts** — causes prompt leakage
- **Don't omit `max_tokens`** — model may generate repetitive text
- **Don't omit `stop` tokens** — model may generate multiple conversation turns

---

## Troubleshooting

### Model outputs garbage text
- Ensure you're using the **v3** GGUF files (merged with PEFT, not Unsloth)
- Path: `/mnt/workspace/zim-my/models/clair-gguf-v3/`

### Model is slow
- Use `n_gpu_layers=-1` to offload all layers to GPU
- Use Q4_K_M instead of Q5_K_M or float16
- Reduce `n_ctx` if you don't need long context

### OOM (Out of Memory)
- Use Q3_K_M (1.48 GB) instead of Q4_K_M (1.80 GB)
- Reduce `n_ctx` to 2048
- Free GPU memory: `torch.cuda.empty_cache()`

### Repetitive responses
- Add `repeat_penalty=1.1` (Ollama) or `repeat_penalty=1.1` (llama-cpp-python)
- Lower `max_tokens`
- Add more `stop` tokens

---

## File Locations (PAI-DSW)

```
/mnt/workspace/zim-my/
├── models/
│   ├── merged/clair-v3/          # Merged HuggingFace model
│   ├── clair-gguf-v3/            # GGUF files
│   │   ├── clair-v3-float16.gguf # 6.17 GB
│   │   ├── clair-v3-Q4_K_M.gguf # 1.80 GB (default)
│   │   ├── clair-v3-Q5_K_M.gguf # 2.07 GB
│   │   └── clair-v3-Q3_K_M.gguf # 1.48 GB
│   └── clair-lora-v2/            # LoRA adapters
├── deploy/
│   ├── Modelfile                 # Ollama Q4_K_M
│   ├── Modelfile.q5              # Ollama Q5_K_M
│   ├── deploy_ollama.sh          # Auto-deploy script
│   └── DEPLOYMENT_GUIDE.md       # This file
└── notebooks/
    ├── rem_merge_with_peft.py    # Merge script
    └── test_clair_v3_better.py   # Test script
```
