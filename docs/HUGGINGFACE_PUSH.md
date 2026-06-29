# Clair v5 — Hugging Face Model Card

This README will be uploaded to Hugging Face as the model card.

## Quick Push Instructions

Run this on your PAI-DSW instance:

```bash
# 1. Install huggingface_hub if not already installed
pip install huggingface_hub

# 2. Set your HF token
export HF_TOKEN="your_huggingface_token_here"

# 3. Run the push script
python scripts/push_to_huggingface.py
```

## What Gets Uploaded

1. **Merged Model** (safetensors format)
   - config.json, tokenizer files, model weights
   - ~6 GB total

2. **GGUF Quantized Models**
   - clair-v5-float16.gguf (5.75 GB)
   - clair-v5-Q4_K_M.gguf (~2.0 GB)
   - clair-v5-Q5_K_M.gguf (~2.5 GB)
   - clair-v5-Q3_K_M.gguf (~1.5 GB)

3. **Model Card** (this README)

## After Upload

Your model will be available at: `https://huggingface.co/r245142r/Clair-3B`

### Running Benchmarks

Hugging Face provides automatic inference API and benchmarks. After upload:

1. **Test the model** directly on the HF website
2. **View automatic benchmarks** (perplexity, etc.)
3. **Get download stats** and usage metrics

### Using with Ollama

```bash
# Pull from Hugging Face (if you set up HF integration)
ollama pull r245142r/Clair-3B

# Or use the local GGUF file
ollama create r245142r/Clair-3B -f deploy/Modelfile
ollama push r245142r/Clair-3B
```

### Using with llama.cpp

```bash
# Download Q4_K_M (recommended for CPU)
wget https://huggingface.co/r245142r/Clair-3B/resolve/main/gguf/clair-v5-Q4_K_M.gguf

# Run inference
./llama-cli -m clair-v5-Q4_K_M.gguf \
  -p "Who are you?" \
  -n 256 \
  --temp 0.7
```

## Model Card Content

The model card includes:
- ✅ Model details (base model, parameters, architecture)
- ✅ Training configuration (LoRA, epochs, dataset)
- ✅ Usage examples (Transformers, Ollama, llama.cpp)
- ✅ Hardware requirements and benchmarks
- ✅ Identity information
- ✅ License (Apache 2.0)

## Troubleshooting

**Error: "Repository not found"**
- Make sure you've created the repo on Hugging Face first
- Or set `exist_ok=True` in the script (already done)

**Error: "Authentication failed"**
- Check your HF_TOKEN is correct
- Token needs "write" permissions

**Error: "File not found"**
- Make sure you've run `merge_clair_v5.py` and `quantize_clair_v5.py` first
- Check the paths in the script match your PAI-DSW setup

## Next Steps

After successful upload:

1. ✅ Test the model on Hugging Face website
2. ✅ Share the link: `https://huggingface.co/r245142r/Clair-3B`
3. ✅ Add to your ADTC 2026 submission
4. ✅ Update REPORT.md with the HF link
5. ✅ Create demo video showing HF page + local inference
