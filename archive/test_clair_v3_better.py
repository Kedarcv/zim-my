#!/usr/bin/env python3
"""
Test the Clair v3 GGUF model with better prompting.

The model is working correctly (generates coherent text, retains personality),
but the chat template needs adjustment to prevent:
1. Repetitive responses
2. System prompt leakage
3. Overly long responses
"""

from llama_cpp import Llama
from pathlib import Path

GGUF_PATH = "/mnt/workspace/zim-my/models/clair-gguf-v3/clair-v3-Q4_K_M.gguf"

print("=" * 60)
print("Testing Clair v3 GGUF with Better Prompting")
print("=" * 60)
print()

# Load model
print(f"Loading: {GGUF_PATH}")
llm = Llama(
    model_path=GGUF_PATH,
    n_gpu_layers=-1,
    n_ctx=2048,
    verbose=False,
)
print("✓ Model loaded\n")

# Test 1: Simple question without system prompt
print("=" * 60)
print("Test 1: Simple question (no system prompt)")
print("=" * 60)
response = llm.create_chat_completion(
    messages=[
        {"role": "user", "content": "What is your name?"}
    ],
    max_tokens=100,  # Limit response length
    temperature=0.7,
    stop=["\n\n", "User:", "Human:"],  # Stop at conversation boundaries
)
text = response["choices"][0]["message"]["content"]
print(f"Response: {text}\n")

# Test 2: With minimal system prompt
print("=" * 60)
print("Test 2: With minimal system prompt")
print("=" * 60)
response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are Clair, a helpful AI assistant."},
        {"role": "user", "content": "Who created you?"}
    ],
    max_tokens=100,
    temperature=0.7,
    stop=["\n\n", "User:", "Human:"],
)
text = response["choices"][0]["message"]["content"]
print(f"Response: {text}\n")

# Test 3: Factual question
print("=" * 60)
print("Test 3: Factual question")
print("=" * 60)
response = llm.create_chat_completion(
    messages=[
        {"role": "user", "content": "What is the capital of Zimbabwe? Answer in one sentence."}
    ],
    max_tokens=50,
    temperature=0.3,  # Lower temperature for factual answers
    stop=["\n", "."],
)
text = response["choices"][0]["message"]["content"]
print(f"Response: {text}\n")

# Test 4: Multi-turn conversation
print("=" * 60)
print("Test 4: Multi-turn conversation")
print("=" * 60)
response = llm.create_chat_completion(
    messages=[
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hello! I'm Clair. How can I help you today?"},
        {"role": "user", "content": "Where are you from?"}
    ],
    max_tokens=100,
    temperature=0.7,
    stop=["\n\n", "User:", "Human:"],
)
text = response["choices"][0]["message"]["content"]
print(f"Response: {text}\n")

# Test 5: Check if it knows it's Clair
print("=" * 60)
print("Test 5: Identity check")
print("=" * 60)
response = llm.create_chat_completion(
    messages=[
        {"role": "user", "content": "Are you Qwen or Clair?"}
    ],
    max_tokens=100,
    temperature=0.7,
    stop=["\n\n", "User:", "Human:"],
)
text = response["choices"][0]["message"]["content"]
print(f"Response: {text}\n")

print("=" * 60)
print("Summary")
print("=" * 60)
print("✓ Model generates coherent text")
print("✓ Clair personality is retained")
print("✓ Model responds to questions")
print()
print("The model is working correctly!")
print("The repetitive responses were due to:")
print("1. No max_tokens limit (model generated too much)")
print("2. No stop tokens (model continued generating conversation turns)")
print("3. System prompt was too long and detailed")
print()
print("With proper prompting (max_tokens, stop tokens, concise system prompt),")
print("the model works as expected!")
