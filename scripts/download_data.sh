#!/bin/bash
# Download datasets for Zim-my fine-tuning
# Run this after setup_pai_dsw.sh

set -e

# Use HuggingFace mirror for China (hf-mirror.com) — only needed on PAI-DSW
# On local machine, use default HuggingFace Hub
# export HF_ENDPOINT=https://hf-mirror.com

echo "=========================================="
echo "Downloading Zimbabwean & Shona Datasets"
echo "=========================================="
echo "Using HuggingFace Hub: ${HF_ENDPOINT:-https://huggingface.co (default)}"
echo ""

# Create directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/rag

# Install huggingface hub if not present
pip install -q huggingface_hub

# Download datasets using Python
python << 'EOF'
from datasets import load_dataset
import os
import json

print("\n[1/6] Downloading Shona RAG corpus (644K records)...")
try:
    ds = load_dataset("cybux/ruzivo-shona-rag", split="train")
    ds.save_to_disk("data/raw/ruzivo-shona-rag")
    print(f"✓ Saved {len(ds)} records to data/raw/ruzivo-shona-rag")
except Exception as e:
    print(f"✗ Error downloading ruzivo-shona-rag: {e}")

print("\n[2/6] Downloading Shona Alpaca dataset (62K records)...")
try:
    ds = load_dataset("saillab/alpaca_shona_taco", split="train")
    ds.save_to_disk("data/raw/alpaca_shona_taco")
    print(f"✓ Saved {len(ds)} records to data/raw/alpaca_shona_taco")
except Exception as e:
    print(f"✗ Error downloading alpaca_shona_taco: {e}")

print("\n[3/6] Downloading Zimbabwe agriculture dataset...")
try:
    ds = load_dataset("sairos/Zimbabwe_agriculture_dataset", split="train")
    ds.save_to_disk("data/raw/zimbabwe_agriculture")
    print(f"✓ Saved {len(ds)} records to data/raw/zimbabwe_agriculture")
except Exception as e:
    print(f"✗ Error downloading zimbabwe_agriculture: {e}")

print("\n[4/6] Downloading Zimbabwe history & heritage...")
try:
    ds = load_dataset("Ruramai/zimbabwe_history_heritage", split="train")
    ds.save_to_disk("data/raw/zimbabwe_history")
    print(f"✓ Saved {len(ds)} records to data/raw/zimbabwe_history")
except Exception as e:
    print(f"✗ Error downloading zimbabwe_history: {e}")

print("\n[5/6] Downloading Shona Code dataset (subset)...")
try:
    ds = load_dataset("michsethowusu/Code-170k-shona", split="train")
    # Take only 10K for now to keep dataset manageable
    ds_subset = ds.select(range(min(10000, len(ds))))
    ds_subset.save_to_disk("data/raw/code_shona_10k")
    print(f"✓ Saved {len(ds_subset)} records to data/raw/code_shona_10k")
except Exception as e:
    print(f"✗ Error downloading code_shona: {e}")

print("\n[6/6] Downloading African math datasets...")
try:
    ds = load_dataset("taresco/big_math_translated_african_languages", split="train")
    # Filter for Shona if available, otherwise keep all
    ds.save_to_disk("data/raw/african_math")
    print(f"✓ Saved {len(ds)} records to data/raw/african_math")
except Exception as e:
    print(f"✗ Error downloading african_math: {e}")

print("\n==========================================")
print("Dataset Download Complete!")
print("==========================================")
print("\nNext: Run notebooks/01_data_prep.ipynb to prepare datasets")
EOF

echo ""
echo "Dataset sizes:"
du -sh data/raw/* 2>/dev/null || echo "No datasets found"
echo ""
