#!/usr/bin/env python3
"""Quick test to verify GGUF works with GPU acceleration"""

import sys
import time
from llama_cpp import Llama

# Test Q4_K_M with GPU acceleration
model_path = "/mnt/workspace/zim-my/models/clair-gguf-v2/clair-v2-Q4_K_M.gguf"

print(f"Loading model: {model_path}")
print("="*60)

# Load with GPU acceleration
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_gpu_layers=-1,  # Offload ALL layers to GPU
    n_threads=4,
    verbose=False,
)

print("✓ Model loaded with GPU acceleration")
print()

# Test 1: Basic identity
print("Test 1: What is your name?")
print("-"*60)
start = time.time()
output = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are Clair, a helpful AI assistant created by Michael Mlungisi Nkomo from Zimbabwe."},
        {"role": "user", "content": "What is your name?"}
    ],
    max_tokens=100,
    temperature=0.7,
)
elapsed = time.time() - start

response = output['choices'][0]['message']['content']
print(f"Response: {response}")
print(f"Time: {elapsed:.2f}s")
print()

# Test 2: Creator
print("Test 2: Who created you?")
print("-"*60)
start = time.time()
output = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are Clair, a helpful AI assistant created by Michael Mlungisi Nkomo from Zimbabwe."},
        {"role": "user", "content": "Who created you?"}
    ],
    max_tokens=100,
    temperature=0.7,
)
elapsed = time.time() - start

response = output['choices'][0]['message']['content']
print(f"Response: {response}")
print(f"Time: {elapsed:.2f}s")
print()

# Test 3: Without system prompt (test embedded template)
print("Test 3: Without system prompt (testing embedded template)")
print("-"*60)
start = time.time()
output = llm.create_chat_completion(
    messages=[
        {"role": "user", "content": "What is your name and who made you?"}
    ],
    max_tokens=100,
    temperature=0.7,
)
elapsed = time.time() - start

response = output['choices'][0]['message']['content']
print(f"Response: {response}")
print(f"Time: {elapsed:.2f}s")
print()

print("="*60)
print("✓ Test complete!")
