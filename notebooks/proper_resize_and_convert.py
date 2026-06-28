#!/usr/bin/env python3
"""
Properly resize model embeddings and re-convert to GGUF.

The previous fix_tokenizer.py had a critical bug:
- It loaded the model with ignore_mismatched_sizes=True
- This REINITIALIZED the embeddings with random weights
- The merged LoRA weights in the embedding layer were LOST
- The safetensors file was never actually updated

This script:
1. Loads the model with the ORIGINAL vocab size (151643)
2. Properly resizes embeddings (preserving existing weights)
3. Saves the resized model
4. Re-converts to GGUF
5. Re-quantizes
"""

import os
import json
import shutil
import subprocess
from pathlib import Path

# Paths
BASE_MODEL_PATH = "/mnt/workspace/models/Qwen/Qwen2.5-3B-Instruct"
MERGED_MODEL_PATH = "/mnt/workspace/zim-my/models/merged/clair-v2/"
GGUF_OUTPUT_PATH = "/mnt/workspace/zim-my/models/clair-gguf-v2/"
LLAMA_CPP_PATH = "/root/.unsloth/llama.cpp/"

def step1_fix_config_and_embeddings():
    """
    Fix the config to match the actual safetensors, then properly resize.
    """
    print("=" * 60)
    print("STEP 1: Fix config and properly resize embeddings")
    print("=" * 60)
    
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    import torch
    
    merged_path = Path(MERGED_MODEL_PATH)
    
    # First, fix the config to match the ACTUAL safetensors (151643)
    config = AutoConfig.from_pretrained(str(merged_path), trust_remote_code=True)
    print(f"Current config vocab_size: {config.vocab_size}")
    
    # Reset config to original vocab_size so model loads correctly
    original_vocab_size = 151643  # Qwen2.5-3B original
    config.vocab_size = original_vocab_size
    config.pad_token_id = 151642  # Original pad token
    config.save_pretrained(str(merged_path))
    print(f"✓ Config reset to vocab_size={original_vocab_size}")
    
    # Now load the model - it should load cleanly since config matches safetensors
    print(f"\nLoading model (should load cleanly now)...")
    model = AutoModelForCausalLM.from_pretrained(
        str(merged_path),
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    current_vocab = model.get_input_embeddings().weight.shape[0]
    print(f"✓ Model loaded with embedding vocab_size: {current_vocab}")
    
    # Load tokenizer to find the required vocab size
    tokenizer = AutoTokenizer.from_pretrained(str(merged_path), trust_remote_code=True)
    
    # Find max token ID including special tokens
    all_token_ids = set(tokenizer.vocab.values())
    if tokenizer.bos_token_id is not None:
        all_token_ids.add(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        all_token_ids.add(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        all_token_ids.add(tokenizer.pad_token_id)
    
    # Check added_tokens_decoder for any additional tokens
    if hasattr(tokenizer, 'added_tokens_decoder'):
        for token_id in tokenizer.added_tokens_decoder:
            all_token_ids.add(token_id)
    
    max_token_id = max(all_token_ids)
    required_vocab_size = max_token_id + 1
    
    print(f"Max token ID in tokenizer: {max_token_id}")
    print(f"Required vocab size: {required_vocab_size}")
    print(f"Current embedding size: {current_vocab}")
    
    if required_vocab_size > current_vocab:
        diff = required_vocab_size - current_vocab
        print(f"\n⚠️  Need to add {diff} extra embedding rows")
        print(f"Resizing from {current_vocab} to {required_vocab_size}...")
        
        # This properly resizes, PRESERVING existing weights
        # New rows are initialized with the model's default initializer
        model.resize_token_embeddings(required_vocab_size)
        
        new_vocab = model.get_input_embeddings().weight.shape[0]
        print(f"✓ Embeddings resized to {new_vocab}")
        
        # Update config
        config.vocab_size = required_vocab_size
        config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else 151642
        config.save_pretrained(str(merged_path))
        print(f"✓ Config updated to vocab_size={required_vocab_size}")
        
        # Save the resized model
        print(f"\nSaving resized model...")
        model.save_pretrained(str(merged_path))
        print(f"✓ Resized model saved")
    else:
        print(f"\n✅ Embeddings already large enough ({current_vocab} >= {required_vocab_size})")
    
    # Verify by reloading
    print(f"\nVerifying by reloading...")
    model2 = AutoModelForCausalLM.from_pretrained(
        str(merged_path),
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    final_vocab = model2.get_input_embeddings().weight.shape[0]
    print(f"✓ Reloaded model has embedding vocab_size: {final_vocab}")
    
    if final_vocab == required_vocab_size:
        print(f"\n✅ SUCCESS: Embeddings properly resized to {final_vocab}")
    else:
        print(f"\n❌ FAILED: Expected {required_vocab_size}, got {final_vocab}")
        return False
    
    # Clean up any stale index file
    index_file = merged_path / "model.safetensors.index.json"
    if index_file.exists():
        index_file.unlink()
        print(f"✓ Removed stale index file")
    
    return True

def step2_test_merged_model():
    """Test the merged model generates coherent text BEFORE GGUF conversion."""
    print("\n" + "=" * 60)
    print("STEP 2: Test merged model (safetensors)")
    print("=" * 60)
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    print(f"✓ Model loaded (embedding: {model.get_input_embeddings().weight.shape})")
    
    # Test generation
    messages = [
        {"role": "system", "content": "You are Clair, a helpful AI assistant created by Michael Mlungisi Nkomo from Zimbabwe."},
        {"role": "user", "content": "What is your name?"}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\nPrompt: What is your name?")
    print(f"Response: {response}")
    
    # Check quality
    if len(response.strip()) > 20 and not all(c in ' \t\n+|[]' for c in response):
        print(f"\n✅ Merged model generates coherent text!")
        return True
    else:
        print(f"\n❌ Merged model generates garbage!")
        print(f"   The LoRA merge itself may be corrupted")
        return False

def step3_convert_to_gguf():
    """Convert the fixed model to GGUF format."""
    print("\n" + "=" * 60)
    print("STEP 3: Convert to GGUF")
    print("=" * 60)
    
    os.makedirs(GGUF_OUTPUT_PATH, exist_ok=True)
    
    # Remove old GGUF files
    old_ggufs = list(Path(GGUF_OUTPUT_PATH).glob("clair-v2-*.gguf"))
    if old_ggufs:
        print(f"Removing {len(old_ggufs)} old GGUF files...")
        for f in old_ggufs:
            f.unlink()
            print(f"  ✓ Removed {f.name}")
    
    convert_script = Path(LLAMA_CPP_PATH) / "convert_hf_to_gguf.py"
    output_file = Path(GGUF_OUTPUT_PATH) / "clair-v2-float16.gguf"
    
    cmd = [
        "python3", str(convert_script),
        MERGED_MODEL_PATH,
        "--outfile", str(output_file),
        "--outtype", "f16"
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✓ Float16 GGUF created: {output_file}")
        return str(output_file)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during GGUF conversion: {e}")
        return None

def step4_quantize(f16_path):
    """Create quantized versions."""
    print("\n" + "=" * 60)
    print("STEP 4: Quantize")
    print("=" * 60)
    
    # Find llama-quantize binary
    llama_quantize = None
    search_paths = [
        Path(LLAMA_CPP_PATH) / "bin/llama-quantize",
        Path(LLAMA_CPP_PATH) / "llama-quantize",
        Path(LLAMA_CPP_PATH) / "build/bin/llama-quantize",
    ]
    
    for p in search_paths:
        if p.exists():
            llama_quantize = p
            break
    
    if llama_quantize is None:
        print("❌ llama-quantize binary not found!")
        return
    
    quantizations = ["Q4_K_M", "Q5_K_M", "Q3_K_M"]
    
    for quant in quantizations:
        output_file = Path(GGUF_OUTPUT_PATH) / f"clair-v2-{quant}.gguf"
        cmd = [str(llama_quantize), f16_path, str(output_file), quant]
        
        print(f"\nQuantizing to {quant}...")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            size_gb = output_file.stat().st_size / (1024**3)
            print(f"✓ {quant}: {size_gb:.2f} GB")
        except subprocess.CalledProcessError as e:
            print(f"❌ {quant} failed: {e.stderr}")

def step5_test_gguf():
    """Test the GGUF model."""
    print("\n" + "=" * 60)
    print("STEP 5: Test GGUF model")
    print("=" * 60)
    
    try:
        from llama_cpp import Llama
    except ImportError:
        print("❌ llama-cpp-python not installed")
        return
    
    gguf_path = Path(GGUF_OUTPUT_PATH) / "clair-v2-Q4_K_M.gguf"
    if not gguf_path.exists():
        print(f"❌ GGUF file not found: {gguf_path}")
        return
    
    print(f"Loading: {gguf_path}")
    llm = Llama(
        model_path=str(gguf_path),
        n_gpu_layers=-1,
        n_ctx=2048,
        verbose=False,
    )
    print(f"✓ Model loaded")
    
    # Test with chat
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are Clair, a helpful AI assistant created by Michael Mlungisi Nkomo from Zimbabwe."},
            {"role": "user", "content": "What is your name?"}
        ],
        max_tokens=200,
        temperature=0.7,
    )
    
    text = response["choices"][0]["message"]["content"]
    print(f"\nPrompt: What is your name?")
    print(f"Response: {text}")
    
    if len(text.strip()) > 10 and not all(c in ' \t\n+|[]' for c in text):
        print(f"\n✅ GGUF model generates coherent text!")
    else:
        print(f"\n❌ GGUF model still generates garbage")

if __name__ == "__main__":
    print("=" * 60)
    print("PROPER EMBEDDING RESIZE + GGUF CONVERSION")
    print("=" * 60)
    print()
    print("This script fixes the critical bug where embeddings were")
    print("reinitialized with random weights instead of being properly")
    print("resized, which destroyed the merged LoRA weights.")
    print()
    
    # Step 1: Fix config and properly resize embeddings
    if not step1_fix_config_and_embeddings():
        print("\n❌ Step 1 failed, aborting")
        exit(1)
    
    # Step 2: Test merged model BEFORE GGUF conversion
    merged_ok = step2_test_merged_model()
    
    if not merged_ok:
        print("\n⚠️  Merged model generates garbage even before GGUF conversion")
        print("   The LoRA merge itself may be corrupted")
        print("   Continuing with GGUF conversion anyway for debugging...")
    
    # Step 3: Convert to GGUF
    f16_path = step3_convert_to_gguf()
    if f16_path is None:
        print("\n❌ GGUF conversion failed")
        exit(1)
    
    # Step 4: Quantize
    step4_quantize(f16_path)
    
    # Step 5: Test GGUF
    step5_test_gguf()
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
