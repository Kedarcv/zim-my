#!/usr/bin/env python3
"""Diagnose the corrupted GGUF file"""

import os
import sys
from pathlib import Path

gguf_dir = Path("/mnt/workspace/zim-my/models/clair-gguf-v2")

print("="*60)
print("GGUF File Diagnostic")
print("="*60)
print()

# Check if files exist
files = list(gguf_dir.glob("*.gguf"))
if not files:
    print("❌ No GGUF files found!")
    sys.exit(1)

print(f"Found {len(files)} GGUF file(s):")
for f in files:
    size_gb = f.stat().st_size / (1024**3)
    print(f"  - {f.name}: {size_gb:.2f} GB")
print()

# Check merged model
merged_dir = Path("/mnt/workspace/zim-my/models/merged/clair-v2")
print("Checking merged model directory:")
if merged_dir.exists():
    safetensors = list(merged_dir.glob("*.safetensors"))
    print(f"  ✓ Found {len(safetensors)} safetensors file(s)")
    
    config_file = merged_dir / "config.json"
    if config_file.exists():
        import json
        with open(config_file) as f:
            config = json.load(f)
        print(f"  ✓ config.json exists")
        print(f"    - vocab_size: {config.get('vocab_size', 'N/A')}")
        print(f"    - bos_token_id: {config.get('bos_token_id', 'N/A')}")
        print(f"    - eos_token_id: {config.get('eos_token_id', 'N/A')}")
        print(f"    - pad_token_id: {config.get('pad_token_id', 'N/A')}")
    else:
        print(f"  ❌ config.json missing!")
    
    tokenizer_file = merged_dir / "tokenizer.json"
    if tokenizer_file.exists():
        print(f"  ✓ tokenizer.json exists")
    else:
        print(f"  ❌ tokenizer.json missing!")
else:
    print(f"  ❌ Merged model directory not found: {merged_dir}")

print()
print("="*60)
print("Testing merged model (safetensors) directly")
print("="*60)
print()

# Try loading the merged model directly
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    print(f"Loading merged model from: {merged_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(str(merged_dir), trust_remote_code=True)
    print(f"✓ Tokenizer loaded (vocab_size: {tokenizer.vocab_size})")
    
    model = AutoModelForCausalLM.from_pretrained(
        str(merged_dir),
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"✓ Model loaded")
    print(f"  - Embedding size: {model.get_input_embeddings().weight.shape}")
    print()
    
    # Test generation
    test_prompt = "What is your name?"
    print(f"Test prompt: {test_prompt}")
    print("-"*60)
    
    messages = [
        {"role": "user", "content": test_prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")
    print()
    
    # Check if output is coherent
    if len(response.strip()) > 20 and not all(c in ' \t\n+|[]' for c in response):
        print("✓ Merged model generates coherent text!")
        print("  → Problem is in GGUF conversion")
    else:
        print("❌ Merged model also generates garbage!")
        print("  → Problem is in LoRA merge process")
        
except Exception as e:
    print(f"❌ Failed to load merged model: {e}")
    import traceback
    traceback.print_exc()
