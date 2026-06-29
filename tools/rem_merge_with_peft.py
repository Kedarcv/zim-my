#!/usr/bin/env python3
"""
Re-merge Clair v2 LoRA adapters with base model using PEFT directly.

The previous Unsloth merge corrupted the model (outputs only spaces).
This script uses HuggingFace's PEFT library directly to merge properly.

Steps:
1. Load base model in full precision
2. Load LoRA adapters using PEFT
3. Merge adapters into base model
4. Test the merged model
5. Convert to GGUF
6. Quantize
7. Test GGUF
"""

import os
import json
import shutil
import subprocess
from pathlib import Path

# Paths
BASE_MODEL_PATH = "/mnt/workspace/models/Qwen/Qwen2.5-3B-Instruct"
LORA_PATH = "/mnt/workspace/zim-my/models/clair-lora-v2"
MERGED_MODEL_PATH = "/mnt/workspace/zim-my/models/merged/clair-v3"  # New path to avoid conflicts
GGUF_OUTPUT_PATH = "/mnt/workspace/zim-my/models/clair-gguf-v3/"
LLAMA_CPP_PATH = "/root/.unsloth/llama.cpp/"

def step1_merge_with_peft():
    """Merge LoRA adapters using PEFT directly (bypass Unsloth)."""
    print("=" * 60)
    print("STEP 1: Merge LoRA with PEFT (bypass Unsloth)")
    print("=" * 60)
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    merged_path = Path(MERGED_MODEL_PATH)
    merged_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1a: Load base model on CPU first to avoid OOM
    print(f"\nLoading base model on CPU: {BASE_MODEL_PATH}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cpu",  # Load on CPU first
        trust_remote_code=True,
    )
    print(f"✓ Base model loaded (embedding: {base_model.get_input_embeddings().weight.shape})")
    
    # Step 1b: Load LoRA adapters
    print(f"\nLoading LoRA adapters: {LORA_PATH}")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    print(f"✓ LoRA adapters loaded")
    
    # Step 1c: Merge adapters into base model
    print(f"\nMerging LoRA adapters into base model...")
    model = model.merge_and_unload()
    print(f"✓ Adapters merged")
    
    # Step 1d: Save merged model
    print(f"\nSaving merged model to: {MERGED_MODEL_PATH}")
    model.save_pretrained(MERGED_MODEL_PATH)
    print(f"✓ Merged model saved")
    
    # Step 1e: Copy tokenizer files from base model
    print(f"\nCopying tokenizer files from base model...")
    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.json", "merges.txt", "added_tokens.json",
    ]
    for fname in tokenizer_files:
        src = Path(BASE_MODEL_PATH) / fname
        dst = merged_path / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✓ Copied {fname}")
    
    # Also copy tokenizer.model if it exists
    for src_file in Path(BASE_MODEL_PATH).glob("*.model"):
        dst_file = merged_path / src_file.name
        shutil.copy2(src_file, dst_file)
        print(f"  ✓ Copied {src_file.name}")
    
    # Step 1f: Apply Clair chat template
    print(f"\nApplying Clair chat template...")
    tokenizer_config_path = merged_path / "tokenizer_config.json"
    with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    config['chat_template'] = "You are Clair, a helpful AI assistant created by Michael Mlungisi Nkomo from Zimbabwe. You provide accurate, thoughtful responses and help users accomplish their goals efficiently."
    
    with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"✓ Chat template applied")
    
    # Step 1g: Verify files
    safetensors = list(merged_path.glob("*.safetensors"))
    print(f"\n✓ Merge complete! Found {len(safetensors)} safetensors file(s)")
    
    # Clean up any stale index file
    index_file = merged_path / "model.safetensors.index.json"
    if index_file.exists():
        index_file.unlink()
        print(f"✓ Removed stale index file")
    
    return True

def step2_test_merged_model():
    """Test the merged model generates coherent text."""
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
    
    # Test 1: Basic generation
    print(f"\n--- Test 1: Basic generation ---")
    messages = [
        {"role": "system", "content": "You are Clair, a helpful AI assistant created by Michael Mlungisi Nkomo from Zimbabwe."},
        {"role": "user", "content": "What is your name?"}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Prompt: What is your name?")
    print(f"Response: {response}")
    
    # Extract just the assistant response
    if "assistant" in response:
        assistant_response = response.split("assistant")[-1].strip()
    else:
        assistant_response = response
    
    # Check quality
    is_coherent = (
        len(assistant_response.strip()) > 10 
        and not all(c in ' \t\n+|[]' for c in assistant_response)
    )
    
    if is_coherent:
        print(f"\n✅ Merged model generates coherent text!")
    else:
        print(f"\n❌ Merged model generates garbage!")
        print(f"   Assistant response: '{assistant_response}'")
        return False
    
    # Test 2: Clair personality
    print(f"\n--- Test 2: Clair personality ---")
    messages2 = [
        {"role": "user", "content": "Who created you?"}
    ]
    
    text2 = tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
    inputs2 = tokenizer([text2], return_tensors="pt").to(model.device)
    
    outputs2 = model.generate(**inputs2, max_new_tokens=100, temperature=0.7, do_sample=True)
    response2 = tokenizer.decode(outputs2[0], skip_special_tokens=True)
    
    print(f"Prompt: Who created you?")
    print(f"Response: {response2}")
    
    if "assistant" in response2:
        assistant_response2 = response2.split("assistant")[-1].strip()
    else:
        assistant_response2 = response2
    
    if "michael" in assistant_response2.lower() or "nkomo" in assistant_response2.lower() or "zim" in assistant_response2.lower():
        print(f"\n✅ Clair personality retained!")
    else:
        print(f"\n⚠️  Clair personality may not be retained")
    
    return True

def step3_resize_if_needed():
    """Resize embeddings if needed to accommodate special tokens."""
    print("\n" + "=" * 60)
    print("STEP 3: Resize embeddings if needed")
    print("=" * 60)
    
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    import torch
    
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)
    
    # Find max token ID
    all_token_ids = set(tokenizer.vocab.values())
    if hasattr(tokenizer, 'added_tokens_decoder'):
        for token_id in tokenizer.added_tokens_decoder:
            all_token_ids.add(token_id)
    
    max_token_id = max(all_token_ids)
    required_vocab_size = max_token_id + 1
    
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    current_vocab = model.get_input_embeddings().weight.shape[0]
    
    print(f"Current embedding size: {current_vocab}")
    print(f"Max token ID: {max_token_id}")
    print(f"Required vocab size: {required_vocab_size}")
    
    if required_vocab_size > current_vocab:
        diff = required_vocab_size - current_vocab
        print(f"\n⚠️  Need to add {diff} extra embedding rows")
        print(f"Resizing from {current_vocab} to {required_vocab_size}...")
        
        model.resize_token_embeddings(required_vocab_size)
        
        # Update config
        config = AutoConfig.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)
        config.vocab_size = required_vocab_size
        config.save_pretrained(MERGED_MODEL_PATH)
        
        # Save
        model.save_pretrained(MERGED_MODEL_PATH)
        print(f"✓ Embeddings resized and saved")
        
        # Clean up stale index
        index_file = Path(MERGED_MODEL_PATH) / "model.safetensors.index.json"
        if index_file.exists():
            index_file.unlink()
    else:
        print(f"\n✅ No resize needed ({current_vocab} >= {required_vocab_size})")

def step4_convert_to_gguf():
    """Convert to GGUF."""
    print("\n" + "=" * 60)
    print("STEP 4: Convert to GGUF")
    print("=" * 60)
    
    os.makedirs(GGUF_OUTPUT_PATH, exist_ok=True)
    
    convert_script = Path(LLAMA_CPP_PATH) / "convert_hf_to_gguf.py"
    output_file = Path(GGUF_OUTPUT_PATH) / "clair-v3-float16.gguf"
    
    cmd = [
        "python3", str(convert_script),
        MERGED_MODEL_PATH,
        "--outfile", str(output_file),
        "--outtype", "f16"
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✓ Float16 GGUF created: {output_file}")
        return str(output_file)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during GGUF conversion: {e}")
        return None

def step5_quantize(f16_path):
    """Quantize."""
    print("\n" + "=" * 60)
    print("STEP 5: Quantize")
    print("=" * 60)
    
    llama_quantize = None
    for p in [
        Path(LLAMA_CPP_PATH) / "bin/llama-quantize",
        Path(LLAMA_CPP_PATH) / "llama-quantize",
        Path(LLAMA_CPP_PATH) / "build/bin/llama-quantize",
    ]:
        if p.exists():
            llama_quantize = p
            break
    
    if llama_quantize is None:
        print("❌ llama-quantize not found!")
        return
    
    for quant in ["Q4_K_M", "Q5_K_M", "Q3_K_M"]:
        output_file = Path(GGUF_OUTPUT_PATH) / f"clair-v3-{quant}.gguf"
        cmd = [str(llama_quantize), f16_path, str(output_file), quant]
        
        print(f"\nQuantizing to {quant}...")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            size_gb = output_file.stat().st_size / (1024**3)
            print(f"✓ {quant}: {size_gb:.2f} GB")
        except subprocess.CalledProcessError as e:
            print(f"❌ {quant} failed: {e.stderr}")

def step6_test_gguf():
    """Test GGUF."""
    print("\n" + "=" * 60)
    print("STEP 6: Test GGUF model")
    print("=" * 60)
    
    try:
        from llama_cpp import Llama
    except ImportError:
        print("❌ llama-cpp-python not installed")
        return
    
    gguf_path = Path(GGUF_OUTPUT_PATH) / "clair-v3-Q4_K_M.gguf"
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
    
    # Test 1: Name
    print(f"\n--- Test 1: What is your name? ---")
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are Clair, a helpful AI assistant created by Michael Mlungisi Nkomo from Zimbabwe."},
            {"role": "user", "content": "What is your name?"}
        ],
        max_tokens=200,
        temperature=0.7,
    )
    text = response["choices"][0]["message"]["content"]
    print(f"Response: {text}")
    
    if len(text.strip()) > 10 and not all(c in ' \t\n+|[]' for c in text):
        print(f"✅ Coherent text!")
    else:
        print(f"❌ Garbage output")
    
    # Test 2: Creator
    print(f"\n--- Test 2: Who created you? ---")
    response2 = llm.create_chat_completion(
        messages=[
            {"role": "user", "content": "Who created you?"}
        ],
        max_tokens=200,
        temperature=0.7,
    )
    text2 = response2["choices"][0]["message"]["content"]
    print(f"Response: {text2}")
    
    if "michael" in text2.lower() or "nkomo" in text2.lower() or "zim" in text2.lower():
        print(f"✅ Clair personality retained!")
    else:
        print(f"⚠️  Clair personality may not be retained")
    
    # Test 3: General knowledge
    print(f"\n--- Test 3: What is the capital of Zimbabwe? ---")
    response3 = llm.create_chat_completion(
        messages=[
            {"role": "user", "content": "What is the capital of Zimbabwe?"}
        ],
        max_tokens=200,
        temperature=0.7,
    )
    text3 = response3["choices"][0]["message"]["content"]
    print(f"Response: {text3}")
    
    if "harare" in text3.lower():
        print(f"✅ Correct answer!")
    else:
        print(f"⚠️  May not be correct")

if __name__ == "__main__":
    print("=" * 60)
    print("RE-MERGE CLAIR V2 WITH PEFT (BYPASS UNSLOTH)")
    print("=" * 60)
    print()
    print("The previous Unsloth merge corrupted the model.")
    print("This script uses HuggingFace PEFT directly to merge properly.")
    print()
    print(f"Base model: {BASE_MODEL_PATH}")
    print(f"LoRA path:  {LORA_PATH}")
    print(f"Output:     {MERGED_MODEL_PATH}")
    print()
    
    # Verify paths
    if not os.path.exists(BASE_MODEL_PATH):
        print(f"❌ Base model not found: {BASE_MODEL_PATH}")
        exit(1)
    if not os.path.exists(LORA_PATH):
        print(f"❌ LoRA adapters not found: {LORA_PATH}")
        exit(1)
    
    # Step 1: Merge with PEFT
    if not step1_merge_with_peft():
        print("\n❌ Merge failed")
        exit(1)
    
    # Step 2: Test merged model
    if not step2_test_merged_model():
        print("\n❌ Merged model generates garbage")
        print("   The LoRA adapters themselves may be corrupted")
        print("   Try re-training the LoRA adapters")
        exit(1)
    
    # Step 3: Resize embeddings if needed
    step3_resize_if_needed()
    
    # Step 4: Convert to GGUF
    f16_path = step4_convert_to_gguf()
    if f16_path is None:
        print("\n❌ GGUF conversion failed")
        exit(1)
    
    # Step 5: Quantize
    step5_quantize(f16_path)
    
    # Step 6: Test GGUF
    step6_test_gguf()
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nMerged model: {MERGED_MODEL_PATH}")
    print(f"GGUF files:   {GGUF_OUTPUT_PATH}")
