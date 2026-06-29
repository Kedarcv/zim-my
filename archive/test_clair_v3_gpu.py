#!/usr/bin/env python3
"""
Test Clair v3 GGUF model directly with llama-cpp-python + GPU acceleration.

Tests identity, personality, factual knowledge, and conversation quality.
"""

import sys
import time
from llama_cpp import Llama

# Clair v3 GGUF path
model_path = "/mnt/workspace/zim-my/models/clair-gguf-v3/clair-v3-Q4_K_M.gguf"

print("=" * 60)
print("Clair v3 — Direct GGUF Test (GPU Accelerated)")
print("=" * 60)
print(f"Model: {model_path}")
print()

# Load with GPU acceleration
print("Loading model...")
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_gpu_layers=-1,  # Offload ALL layers to GPU
    n_threads=4,
    verbose=False,
)
print("✓ Model loaded with GPU acceleration\n")

# ── Test 1: Identity ──────────────────────────────────────
print("=" * 60)
print("Test 1: What is your name?")
print("=" * 60)
start = time.time()
output = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are Clair, a helpful AI assistant created by Michael Mlungisi Nkomo from Zimbabwe."},
        {"role": "user", "content": "What is your name?"}
    ],
    max_tokens=100,
    temperature=0.7,
    stop=["\n\n", "User:", "Human:"],
)
elapsed = time.time() - start
response = output["choices"][0]["message"]["content"]
print(f"Response: {response}")
print(f"Time: {elapsed:.2f}s\n")

# ── Test 2: Creator ───────────────────────────────────────
print("=" * 60)
print("Test 2: Who created you?")
print("=" * 60)
start = time.time()
output = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are Clair, a helpful AI assistant created by Michael Mlungisi Nkomo from Zimbabwe."},
        {"role": "user", "content": "Who created you?"}
    ],
    max_tokens=100,
    temperature=0.7,
    stop=["\n\n", "User:", "Human:"],
)
elapsed = time.time() - start
response = output["choices"][0]["message"]["content"]
print(f"Response: {response}")
print(f"Time: {elapsed:.2f}s\n")

# ── Test 3: Factual knowledge ─────────────────────────────
print("=" * 60)
print("Test 3: What is the capital of Zimbabwe?")
print("=" * 60)
start = time.time()
output = llm.create_chat_completion(
    messages=[
        {"role": "user", "content": "What is the capital of Zimbabwe? Answer in one sentence."}
    ],
    max_tokens=50,
    temperature=0.3,
    stop=["\n", "."],
)
elapsed = time.time() - start
response = output["choices"][0]["message"]["content"]
print(f"Response: {response}")
print(f"Time: {elapsed:.2f}s\n")

# ── Test 4: Without system prompt ─────────────────────────
print("=" * 60)
print("Test 4: No system prompt — testing embedded personality")
print("=" * 60)
start = time.time()
output = llm.create_chat_completion(
    messages=[
        {"role": "user", "content": "What is your name and who made you?"}
    ],
    max_tokens=100,
    temperature=0.7,
    stop=["\n\n", "User:", "Human:"],
)
elapsed = time.time() - start
response = output["choices"][0]["message"]["content"]
print(f"Response: {response}")
print(f"Time: {elapsed:.2f}s\n")

# ── Test 5: Multi-turn conversation ───────────────────────
print("=" * 60)
print("Test 5: Multi-turn conversation")
print("=" * 60)
start = time.time()
output = llm.create_chat_completion(
    messages=[
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hello! I'm Clair. How can I help you today?"},
        {"role": "user", "content": "Where are you from?"}
    ],
    max_tokens=100,
    temperature=0.7,
    stop=["\n\n", "User:", "Human:"],
)
elapsed = time.time() - start
response = output["choices"][0]["message"]["content"]
print(f"Response: {response}")
print(f"Time: {elapsed:.2f}s\n")

# ── Test 6: Helpful coding response ───────────────────────
print("=" * 60)
print("Test 6: Coding help — write a Python function")
print("=" * 60)
start = time.time()
output = llm.create_chat_completion(
    messages=[
        {"role": "user", "content": "Write a Python function that checks if a number is prime."}
    ],
    max_tokens=256,
    temperature=0.5,
    stop=["\n\n\n", "User:", "Human:"],
)
elapsed = time.time() - start
response = output["choices"][0]["message"]["content"]
print(f"Response:\n{response}")
print(f"Time: {elapsed:.2f}s\n")

# ── Test 7: Personality / friendliness ────────────────────
print("=" * 60)
print("Test 7: How are you today?")
print("=" * 60)
start = time.time()
output = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are Clair, a helpful AI assistant created by Michael Mlungisi Nkomo from Zimbabwe."},
        {"role": "user", "content": "How are you today?"}
    ],
    max_tokens=100,
    temperature=0.7,
    stop=["\n\n", "User:", "Human:"],
)
elapsed = time.time() - start
response = output["choices"][0]["message"]["content"]
print(f"Response: {response}")
print(f"Time: {elapsed:.2f}s\n")

# ── Summary ───────────────────────────────────────────────
print("=" * 60)
print("✓ All tests complete!")
print("=" * 60)
print()
print("Check the responses above for:")
print("  1. Does it say its name is Clair?")
print("  2. Does it mention Michael Mlungisi Nkomo / Zimbabwe?")
print("  3. Does it answer factual questions correctly?")
print("  4. Does it work without a system prompt?")
print("  5. Does it maintain conversation context?")
print("  6. Does it generate useful code?")
print("  7. Is it friendly and personable?")
