#!/usr/bin/env python3
"""
Merge Clair v5 LoRA adapters with base model using PEFT directly.

Clair v5 uses improved training to fix behavioral issues:
- Only mentions identity when explicitly asked
- Proper greeting/goodbye handling
- Says "I don't understand" when confused
- Normal conversations without constant identity mentions

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
LORA_PATH = "/mnt/workspace/models/clair-lora-v5"
MERGED_MODEL_PATH = "/mnt/workspace/zim-my/models/merged/clair-v5"
GGUF_OUTPUT_PATH = "/mnt/workspace/zim-my/models/clair-gguf-v5/"
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
    
    # Step 1a: Load base model
    print(f"\nLoading base model on GPU 1: {BASE_MODEL_PATH}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"✓ Base model loaded (embedding: {base_model.get_input_embeddings().weight.shape})")
    
    # Step 1b: Load LoRA adapters
    print(f"\nLoading LoRA adapters: {LORA_PATH}")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    print(f"✓ LoRA adapters loaded")
    
    # Step 1c: Merge
    print(f"\nMerging LoRA adapters into base model...")
    model = model.merge_and_unload()
    print(f"✓ Adapters merged")
    
    # Step 1d: Save merged model
    print(f"\nSaving merged model to: {MERGED_MODEL_PATH}")
    model.save_pretrained(MERGED_MODEL_PATH)
    print(f"✓ Merged model saved")
    
    # Step 1e: Copy tokenizer files
    print(f"\nCopying tokenizer files from base model...")
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]
    
    for filename in tokenizer_files:
        src = Path(BASE_MODEL_PATH) / filename
        dst = merged_path / filename
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✓ Copied {filename}")
    
    # Step 1f: Verify safetensors
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
        qwen_chat_template = """{% macro render_msg(msg) %}{{ msg['role'] }}
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
    
    # Load tokenizer and ensure chat_template is available
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
    
    # If chat_template is missing, set it directly on the tokenizer
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        print("  ⚠ Tokenizer missing chat_template, setting manually...")
        qwen_chat_template = """{% macro render_msg(msg) %}{{ msg['role'] }}
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
        tokenizer.chat_template = qwen_chat_template
        print("  ✓ chat_template set on tokenizer")
    
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"✓ Model loaded")
    
    # Test 1: Greeting (should NOT mention identity)
    print("\n" + "-" * 60)
    print("Test 1: Greeting (should NOT mention identity)")
    print("-" * 60)
    messages = [
        {"role": "user", "content": "Hi"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")
    
    # Test 2: Identity question (SHOULD mention identity)
    print("\n" + "-" * 60)
    print("Test 2: Identity question (SHOULD mention identity)")
    print("-" * 60)
    messages = [
        {"role": "user", "content": "Who are you?"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")
    
    # Test 3: Goodbye (should handle properly)
    print("\n" + "-" * 60)
    print("Test 3: Goodbye (should handle properly)")
    print("-" * 60)
    messages = [
        {"role": "user", "content": "Bye"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")
    
    # Test 4: Normal question (should NOT mention identity)
    print("\n" + "-" * 60)
    print("Test 4: Normal question (should NOT mention identity)")
    print("-" * 60)
    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")
    
    print("\n✓ Tests complete")
    print("\nExpected behavior:")
    print("  - Test 1: Simple greeting, no identity mention")
    print("  - Test 2: Should identify as Clair, created by Michael Mlungisi Nkomo")
    print("  - Test 3: Proper goodbye response")
    print("  - Test 4: Direct answer, no identity mention")
    
    return True

def step3_convert_to_gguf():
    """Convert merged model to GGUF format."""
    print("\n" + "=" * 60)
    print("STEP 3: Convert to GGUF")
    print("=" * 60)
    
    gguf_path = Path(GGUF_OUTPUT_PATH)
    gguf_path.mkdir(parents=True, exist_ok=True)
    
    output_file = gguf_path / "clair-v5-float16.gguf"
    
    print(f"\nConverting: {MERGED_MODEL_PATH}")
    print(f"Output: {output_file}")
    
    # Use llama.cpp convert script
    convert_script = Path(LLAMA_CPP_PATH) / "convert_hf_to_gguf.py"
    
    if not convert_script.exists():
        print(f"✗ Convert script not found: {convert_script}")
        return False
    
    cmd = [
        "python", str(convert_script),
        MERGED_MODEL_PATH,
        "--outfile", str(output_file),
        "--outtype", "f16"
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"✗ Conversion failed!")
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
    input_file = gguf_path / "clair-v5-float16.gguf"
    
    # Try multiple locations for llama-quantize
    possible_paths = [
        Path(LLAMA_CPP_PATH) / "build" / "bin" / "llama-quantize",
        Path("/usr/local/bin/llama-quantize"),
        Path("/usr/bin/llama-quantize"),
    ]
    
    quantize_bin = None
    for path in possible_paths:
        if path.exists():
            quantize_bin = path
            break
    
    # If not found, try to build llama.cpp
    if quantize_bin is None:
        print("⚠ llama-quantize not found, attempting to build llama.cpp...")
        llama_cpp_dir = Path(LLAMA_CPP_PATH)
        
        if llama_cpp_dir.exists():
            build_dir = llama_cpp_dir / "build"
            build_dir.mkdir(exist_ok=True)
            
            print("Running cmake...")
            result = subprocess.run(
                ["cmake", ".."],
                cwd=build_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("Building llama-quantize...")
                result = subprocess.run(
                    ["make", "llama-quantize", "-j"],
                    cwd=build_dir,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    quantize_bin = build_dir / "bin" / "llama-quantize"
                    print(f"✓ Built successfully")
                else:
                    print(f"✗ Build failed: {result.stderr[:500]}")
            else:
                print(f"✗ CMake failed: {result.stderr[:500]}")
    
    # If still not found, skip quantization
    if quantize_bin is None or not quantize_bin.exists():
        print("\n⚠ llama-quantize not available")
        print("The F16 GGUF model is ready for use:")
        print(f"  {input_file}")
        print("\nTo create quantized versions later, run:")
        print(f"  cd {LLAMA_CPP_PATH}/build && make llama-quantize")
        print(f"  {LLAMA_CPP_PATH}/build/bin/llama-quantize {input_file} output.gguf Q4_K_M")
        print("\nAlternatively, you can use the F16 version directly with Ollama or llama.cpp.")
        print("Skipping quantization...")
        return True
    
    quantizations = [
        ("Q4_K_M", "clair-v5-Q4_K_M.gguf"),
        ("Q5_K_M", "clair-v5-Q5_K_M.gguf"),
        ("Q3_K_M", "clair-v5-Q3_K_M.gguf"),
    ]
    
    created_count = 0
    for quant_type, output_name in quantizations:
        output_file = gguf_path / output_name
        print(f"\nQuantizing to {quant_type}...")
        
        cmd = [str(quantize_bin), str(input_file), str(output_file), quant_type]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"✗ Quantization failed for {quant_type}!")
            print(f"STDERR: {result.stderr[:200]}")
            continue
        
        size_gb = output_file.stat().st_size / 1024**3
        print(f"✓ Created: {output_name} ({size_gb:.2f} GB)")
        created_count += 1
    
    if created_count == 0:
        print("\n⚠ All quantizations failed")
        print("The F16 GGUF model is available for use")
    else:
        print(f"\n✓ Successfully created {created_count} quantized version(s)")
    
    return True

def step5_test_gguf():
    """Test the GGUF model."""
    print("\n" + "=" * 60)
    print("STEP 5: Test GGUF model")
    print("=" * 60)
    
    from llama_cpp import Llama
    
    gguf_path = Path(GGUF_OUTPUT_PATH)
    
    # Try quantized versions first, then F16
    model_files = [
        gguf_path / "clair-v5-Q4_K_M.gguf",
        gguf_path / "clair-v5-Q5_K_M.gguf",
        gguf_path / "clair-v5-float16.gguf",
    ]
    
    model_file = None
    for f in model_files:
        if f.exists():
            model_file = f
            break
    
    if model_file is None:
        print("✗ No GGUF model found!")
        return False
    
    print(f"\nLoading GGUF model: {model_file}")
    print(f"  Size: {model_file.stat().st_size / 1024**3:.2f} GB")
    
    llm = Llama(
        model_path=str(model_file),
        n_ctx=4096,
        n_gpu_layers=-1,  # Use all layers on GPU
        verbose=False,
    )
    print(f"✓ Model loaded")
    
    # Test 1: Greeting
    print("\n" + "-" * 60)
    print("Test 1: Greeting")
    print("-" * 60)
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=50,
        temperature=0.7,
    )
    print(f"Response: {response['choices'][0]['message']['content']}")
    
    # Test 2: Identity
    print("\n" + "-" * 60)
    print("Test 2: Identity question")
    print("-" * 60)
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": "Who are you?"}],
        max_tokens=100,
        temperature=0.7,
    )
    print(f"Response: {response['choices'][0]['message']['content']}")
    
    # Test 3: Goodbye
    print("\n" + "-" * 60)
    print("Test 3: Goodbye")
    print("-" * 60)
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": "Bye"}],
        max_tokens=50,
        temperature=0.7,
    )
    print(f"Response: {response['choices'][0]['message']['content']}")
    
    print("\n✓ GGUF tests complete")
    
    return True

def main():
    """Run all steps."""
    print("=" * 60)
    print("Clair v5 - Merge, Convert, and Test")
    print("=" * 60)
    print("\nThis script will:")
    print("  1. Merge LoRA adapters with base model")
    print("  2. Test the merged model")
    print("  3. Convert to GGUF")
    print("  4. Quantize to smaller formats")
    print("  5. Test the GGUF model")
    print("\nClair v5 fixes behavioral issues:")
    print("  - Only mentions identity when asked")
    print("  - Proper greeting/goodbye handling")
    print("  - Says 'I don't understand' when confused")
    print("  - No constant identity mentions")
    
    input("\nPress Enter to continue...")
    
    # Step 1: Merge
    if not step1_merge_with_peft():
        print("\n✗ Merge failed!")
        return False
    
    # Step 2: Test merged model
    if not step2_test_merged_model():
        print("\n✗ Testing failed!")
        return False
    
    # Step 3: Convert to GGUF
    if not step3_convert_to_gguf():
        print("\n✗ Conversion failed!")
        return False
    
    # Step 4: Quantize
    if not step4_quantize():
        print("\n✗ Quantization failed!")
        return False
    
    # Step 5: Test GGUF
    if not step5_test_gguf():
        print("\n✗ GGUF testing failed!")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All steps completed successfully!")
    print("=" * 60)
    print(f"\nMerged model: {MERGED_MODEL_PATH}")
    print(f"GGUF models: {GGUF_OUTPUT_PATH}")
    print("\nNext steps:")
    print("  1. Update Modelfile to point to clair-v5-float16.gguf")
    print("  2. Create Ollama model:")
    print("     ollama create r245142r/Clair-3B -f deploy/Modelfile")
    print("  3. Push to Ollama:")
    print("     ollama push r245142r/Clair-3B")
    
    return True

if __name__ == "__main__":
    main()
