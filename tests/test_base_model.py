#!/usr/bin/env python3
"""Test if the base Qwen2.5-3B model works correctly"""

import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_model_path = "/mnt/workspace/models/Qwen/Qwen2.5-3B-Instruct"

print("="*60)
print("Testing BASE model (before LoRA merge)")
print("="*60)
print(f"Loading: {base_model_path}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
print(f"✓ Tokenizer loaded (vocab_size: {tokenizer.vocab_size})")

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
print(f"✓ Model loaded")
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
    print("✓ Base model generates coherent text")
else:
    print("❌ Base model also generates garbage!")
    print("   This suggests a hardware or environment issue")
