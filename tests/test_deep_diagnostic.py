#!/usr/bin/env python3
"""Deep diagnostic: check tokenizer_config.json and test raw generation"""

import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/mnt/workspace/zim-my/models/merged/clair-v2"
tokenizer_config = os.path.join(model_path, "tokenizer_config.json")

print("="*60)
print("DIAGNOSTIC 1: Check tokenizer_config.json on disk")
print("="*60)

with open(tokenizer_config, 'r') as f:
    config = json.load(f)

chat_template = config.get('chat_template', 'NOT FOUND')
if 'Clair' in chat_template:
    print("✓ Chat template on disk contains 'Clair'")
elif 'Qwen' in chat_template:
    print("✗ Chat template on disk still has 'Qwen' (template fix didn't save)")
else:
    print("? Chat template doesn't contain 'Clair' or 'Qwen'")

print(f"\nFirst 300 chars of chat_template:\n{chat_template[:300]}")
print()

print("="*60)
print("DIAGNOSTIC 2: Check model weights integrity")
print("="*60)

from safetensors import safe_open
import glob

safetensor_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
print(f"Found {len(safetensor_files)} safetensor files:")
for f in safetensor_files:
    size_gb = os.path.getsize(f) / 1024**3
    print(f"  {os.path.basename(f)}: {size_gb:.2f} GB")

# Check a few weight tensors for NaN/Inf
print("\nChecking weight integrity...")
for sf_file in safetensor_files:
    with safe_open(sf_file, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        # Check first and last tensor
        for key in [keys[0], keys[-1]]:
            tensor = f.get_tensor(key)
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()
            mean_val = tensor.float().mean().item()
            std_val = tensor.float().std().item()
            status = "✗ CORRUPT" if (has_nan or has_inf) else "✓ OK"
            print(f"  {status} {key}: mean={mean_val:.6f}, std={std_val:.6f}, nan={has_nan}, inf={has_inf}")
print()

print("="*60)
print("DIAGNOSTIC 3: Test base model (before merge)")
print("="*60)

base_model_path = "/mnt/workspace/models/Qwen/Qwen2.5-3B-Instruct"
print(f"Loading base model: {base_model_path}")

base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Say hello in one sentence."}
]
text = base_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"Prompt:\n{text}")

inputs = base_tokenizer([text], return_tensors="pt").to(base_model.device)
outputs = base_model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.7,
    do_sample=True,
)
response = base_tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nBase model response:\n{response}")
print()

# Clean up base model
del base_model, base_tokenizer
import gc
gc.collect()
torch.cuda.empty_cache()

print("="*60)
print("DIAGNOSTIC 4: Test merged model with simple generation")
print("="*60)

print(f"Loading merged model: {model_path}")
merged_tokenizer = AutoTokenizer.from_pretrained(model_path)
merged_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Check tokenizer vocab size
print(f"Tokenizer vocab size: {merged_tokenizer.vocab_size}")
print(f"Model vocab size: {merged_model.get_input_embeddings().weight.shape[0]}")

# Simple completion test (no chat template)
simple_prompt = "Hello, my name is"
inputs = merged_tokenizer(simple_prompt, return_tensors="pt").to(merged_model.device)
print(f"\nSimple prompt: '{simple_prompt}'")
print(f"Input token IDs: {inputs['input_ids'][0][:10].tolist()}")

outputs = merged_model.generate(
    **inputs,
    max_new_tokens=30,
    temperature=0.7,
    do_sample=True,
)
response = merged_tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Merged model response: '{response}'")
print()

# Check if output tokens are valid
output_tokens = outputs[0][inputs['input_ids'].shape[1]:]
print(f"Generated token IDs: {output_tokens[:20].tolist()}")
print(f"Generated tokens decoded: '{merged_tokenizer.decode(output_tokens)}'")

print()
print("="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)
print()
print("Summary:")
print("- If base model works but merged doesn't → merge corrupted the weights")
print("- If both fail → base model or environment issue")
print("- If tokenizer_config still has 'Qwen' → template fix didn't persist")
