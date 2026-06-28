#!/usr/bin/env python3
"""Test the merged model directly to verify it works before GGUF conversion"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/mnt/workspace/zim-my/models/merged/clair-v2"

print(f"Loading merged model: {model_path}")
print("="*60)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("✓ Model loaded successfully")
print()

# Test 1: Basic identity
print("Test 1: What is your name?")
print("-"*60)

messages = [
    {"role": "system", "content": "You are Clair, a helpful AI assistant created by Michael Mlungisi Nkomo from Zimbabwe."},
    {"role": "user", "content": "What is your name?"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# Extract just the assistant's response
response = response.split("<|im_start|>assistant")[-1].strip()
print(f"Response: {response}")
print()

# Test 2: Creator
print("Test 2: Who created you?")
print("-"*60)

messages = [
    {"role": "system", "content": "You are Clair, a helpful AI assistant created by Michael Mlungisi Nkomo from Zimbabwe."},
    {"role": "user", "content": "Who created you?"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
response = response.split("<|im_start|>assistant")[-1].strip()
print(f"Response: {response}")
print()

# Test 3: Without system prompt (test embedded template)
print("Test 3: Without system prompt (testing embedded template)")
print("-"*60)

messages = [
    {"role": "user", "content": "What is your name and who made you?"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"Generated prompt:\n{text[:200]}...")
print()

inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
response = response.split("<|im_start|>assistant")[-1].strip()
print(f"Response: {response}")
print()

print("="*60)
print("✓ Test complete!")
print()
print("If the merged model works correctly, the issue is with GGUF conversion.")
print("If it also produces garbage, the merge process corrupted the model.")
