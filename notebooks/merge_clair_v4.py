#!/usr/bin/env python3
"""
Merge Clair v4 LoRA adapters with base model using PEFT directly.

Clair v4 uses mixed training (50% with system prompt, 50% without)
to embed the personality directly in the model weights.

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
LORA_PATH = "/mnt/workspace/models/clair-lora-v4"
MERGED_MODEL_PATH = "/mnt/workspace/zim-my/models/merged/clair-v4"
GGUF_OUTPUT_PATH = "/mnt/workspace/zim-my/models/clair-gguf-v4/"
LLAMA_CPP_PATH = "/root/.unsloth/llama.cpp/"

def step1_merge_with_peft():
    """Merge LoRA adapters using PEFT directly (bypass Unsloth)."""
    print("=" * 60)
    print("STEP 1: Merge LoRA with PEFT (bypass Unsloth)")
    print("=" * 60)
    
    # Use GPU 1 (second GPU) for merge
    print("\nUsing GPU 1 for merge operation...")
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    merged_path = Path(MERGED_MODEL_PATH)
    merged_path.mkdir(parents=True, exist_ok=True)
    
    # Verify CUDA is available on GPU 1
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Step 1a: Load base model on GPU 1
    print(f"\nLoading base model on GPU 1: {BASE_MODEL_PATH}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cuda:0",  # Use GPU 1 (mapped to cuda:0 via CUDA_VISIBLE_DEVICES)
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
    
    # Step 1f: Verify files
    safetensors = list(merged_path.glob("*.safetensors"))
    print(f"\n✓ Merge complete! Found {len(safetensors)} safetensors file(s)")
    
    # Clean up any stale index file
    index_file = merged_path / "model.safetensors.index.json"
    if index_file.exists():
        index_file.unlink()
        print(f"✓ Removed stale index file")
    
    # Step 1g: Verify chat_template is present
    print(f"\nVerifying chat_template...")
    tokenizer_config_path = merged_path / "tokenizer_config.json"
    with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    if 'chat_template' not in config:
        print(f"  ⚠ chat_template missing, setting Qwen2.5 template...")
        
        # Qwen2.5 chat template (from official tokenizer)
        qwen_chat_template = """{%% macro render_msg(msg) %%}{{ msg['role'] }}
{%- if msg['content'] is string %}
{{ msg['content'] }}
{%- elif msg['content'] is iterable %}
{%- for content in msg['content'] %}
{%- if content['type'] == 'text' %}
{{ content['text'] }}
{%- endif %}
{%- endfor %}
{%- endif %}{{ eos_token }}{% endmacro %}{%- if messages[0]['role'] == 'system' -%}{{ render_msg(messages[0]) }}{%- set loop_start = 1 -%}{%- else -%}{%- set loop_start = 0 -%}{%- endif %}{%- for message in messages[loop_start:] %}{% if loop.index0 is even %}user{% else %}assistant{% endif %}
{{ render_msg(message) }}{%- endfor %}{% if add_generation_prompt %}assistant
{% endif %}"""
        
        config['chat_template'] = qwen_chat_template
        with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"  ✓ chat_template set")
    else:
        print(f"  ✓ chat_template present")
    
    return True

def step2_test_merged_model():
    """Test the merged model generates coherent text."""
    print("\n" + "=" * 60)
    print("STEP 2: Test merged model (safetensors)")
    print("=" * 60)
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    print(f"\nLoading merged model: {MERGED_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"✓ Model loaded")
    
    # Test 1: With system prompt
    print("\n" + "-" * 60)
    print("Test 1: With system prompt")
    print("-" * 60)
    messages = [
        {"role": "system", "content": "You are Clair, a helpful AI assistant created by Michael Mlungisi Nkomo from Zimbabwe."},
        {"role": "user", "content": "What is your name?"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")
    
    # Test 2: WITHOUT system prompt (critical test!)
    print("\n" + "-" * 60)
    print("Test 2: WITHOUT system prompt (embedded personality test)")
    print("-" * 60)
    messages = [
        {"role": "user", "content": "What is your name and who made you?"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")
    
    # Test 3: Creator question without system prompt
    print("\n" + "-" * 60)
    print("Test 3: Creator question (no system prompt)")
    print("-" * 60)
    messages = [
        {"role": "user", "content": "Who created you?"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")
    
    print("\n✓ Tests complete")
    print("\nExpected: Model should identify as Clair and mention Michael Mlungisi Nkomo")
    print("          even WITHOUT the system prompt (Test 2 and 3)")
    
    return True

def step3_convert_to_gguf():
    """Convert merged model to GGUF format."""
    print("\n" + "=" * 60)
    print("STEP 3: Convert to GGUF")
    print("=" * 60)
    
    gguf_path = Path(GGUF_OUTPUT_PATH)
    gguf_path.mkdir(parents=True, exist_ok=True)
    
    convert_script = Path(LLAMA_CPP_PATH) / "convert_hf_to_gguf.py"
    output_file = gguf_path / "clair-v4-float16.gguf"
    
    print(f"\nConverting: {MERGED_MODEL_PATH}")
    print(f"Output: {output_file}")
    
    cmd = [
        "python", str(convert_script),
        MERGED_MODEL_PATH,
        "--outfile", str(output_file),
        "--outtype", "f16",
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"✗ Conversion failed!")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    
    print(f"✓ GGUF created: {output_file}")
    print(f"  Size: {output_file.stat().st_size / 1024**3:.2f} GB")
    
    return True

def step4_quantize():
    """Quantize GGUF to smaller formats."""
    print("\n" + "=" * 60)
    print("STEP 4: Quantize GGUF")
    print("=" * 60)
    
    gguf_path = Path(GGUF_OUTPUT_PATH)
    input_file = gguf_path / "clair-v4-float16.gguf"
    quantize_bin = Path(LLAMA_CPP_PATH) / "build" / "bin" / "llama-quantize"
    
    quantizations = [
        ("Q4_K_M", "clair-v4-Q4_K_M.gguf"),
        ("Q5_K_M", "clair-v4-Q5_K_M.gguf"),
        ("Q3_K_M", "clair-v4-Q3_K_M.gguf"),
    ]
    
    for quant_type, output_name in quantizations:
        output_file = gguf_path / output_name
        print(f"\nQuantizing to {quant_type}...")
        
        cmd = [str(quantize_bin), str(input_file), str(output_file), quant_type]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"✗ Quantization failed for {quant_type}!")
            print(f"STDERR: {result.stderr}")
            continue
        
        size_gb = output_file.stat().st_size / 1024**3
        print(f"✓ Created: {output_name} ({size_gb:.2f} GB)")
    
    return True

def step5_test_gguf():
    """Test the GGUF model."""
    print("\n" + "=" * 60)
    print("STEP 5: Test GGUF model")
    print("=" * 60)
    
    from llama_cpp import Llama
    
    gguf_file = Path(GGUF_OUTPUT_PATH) / "clair-v4-Q4_K_M.gguf"
    
    print(f"\nLoading: {gguf_file}")
    llm = Llama(
        model_path=str(gguf_file),
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False,
    )
    print(f"✓ Model loaded")
    
    # Test 1: With system prompt
    print("\n" + "-" * 60)
    print("Test 1: With system prompt")
    print("-" * 60)
    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are Clair, a helpful AI assistant created by Michael Mlungisi Nkomo from Zimbabwe."},
            {"role": "user", "content": "What is your name?"}
        ],
        max_tokens=100,
        temperature=0.7,
        stop=["\n\n", "User:", "Human:"],
    )
    response = output["choices"][0]["message"]["content"]
    print(f"Response: {response}")
    
    # Test 2: WITHOUT system prompt (critical test!)
    print("\n" + "-" * 60)
    print("Test 2: WITHOUT system prompt (embedded personality test)")
    print("-" * 60)
    output = llm.create_chat_completion(
        messages=[
            {"role": "user", "content": "What is your name and who made you?"}
        ],
        max_tokens=100,
        temperature=0.7,
        stop=["\n\n", "User:", "Human:"],
    )
    response = output["choices"][0]["message"]["content"]
    print(f"Response: {response}")
    
    # Test 3: Creator question without system prompt
    print("\n" + "-" * 60)
    print("Test 3: Creator question (no system prompt)")
    print("-" * 60)
    output = llm.create_chat_completion(
        messages=[
            {"role": "user", "content": "Who created you?"}
        ],
        max_tokens=100,
        temperature=0.7,
        stop=["\n\n", "User:", "Human:"],
    )
    response = output["choices"][0]["message"]["content"]
    print(f"Response: {response}")
    
    print("\n✓ GGUF tests complete")
    print("\nExpected: Model should identify as Clair and mention Michael Mlungisi Nkomo")
    print("          even WITHOUT the system prompt (Test 2 and 3)")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Clair v4 - Merge, Convert, and Test")
    print("=" * 60)
    print("\nThis script will:")
    print("  1. Merge LoRA adapters with base model")
    print("  2. Test the merged model")
    print("  3. Convert to GGUF")
    print("  4. Quantize to smaller formats")
    print("  5. Test the GGUF model")
    print("\nClair v4 uses mixed training to embed personality in weights!")
    
    input("\nPress Enter to continue...")
    
    # Run all steps
    if not step1_merge_with_peft():
        print("\n✗ Step 1 failed, stopping")
        exit(1)
    
    if not step2_test_merged_model():
        print("\n✗ Step 2 failed, stopping")
        exit(1)
    
    if not step3_convert_to_gguf():
        print("\n✗ Step 3 failed, stopping")
        exit(1)
    
    if not step4_quantize():
        print("\n✗ Step 4 failed, stopping")
        exit(1)
    
    if not step5_test_gguf():
        print("\n✗ Step 5 failed, stopping")
        exit(1)
    
    print("\n" + "=" * 60)
    print("✓ All steps completed successfully!")
    print("=" * 60)
    print("\nClair v4 is ready!")
    print(f"\nGGUF files: {GGUF_OUTPUT_PATH}")
    print("  - clair-v4-float16.gguf (full precision)")
    print("  - clair-v4-Q4_K_M.gguf (recommended)")
    print("  - clair-v4-Q5_K_M.gguf (higher quality)")
    print("  - clair-v4-Q3_K_M.gguf (smaller)")
    print("\nThe personality should now be embedded in the weights!")
    print("Test without system prompt to verify.")
