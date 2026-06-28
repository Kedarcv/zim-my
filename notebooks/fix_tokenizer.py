#!/usr/bin/env python3
"""
Fix the corrupted tokenizer in the merged Clair model.

Root cause: Unsloth's merge process corrupted the tokenizer, causing vocab mismatch:
- Tokenizer vocab size: 151,643
- Model embedding vocab size: 151,936
- Difference: 293 tokens missing

This script:
1. Copies tokenizer files from base model to merged model
2. Re-applies the Clair chat template to tokenizer_config.json
3. Verifies vocab sizes match
4. Re-converts to GGUF format
"""

import os
import shutil
import json
import subprocess
from pathlib import Path

# Paths
BASE_MODEL_PATH = "/mnt/workspace/models/Qwen/Qwen2.5-3B-Instruct"
MERGED_MODEL_PATH = "/mnt/workspace/zim-my/models/merged/clair-v2/"
GGUF_OUTPUT_PATH = "/mnt/workspace/zim-my/models/clair-gguf-v2/"
LLAMA_CPP_PATH = "/root/.unsloth/llama.cpp/"

# Clair's identity for chat template
CLAIR_CHAT_TEMPLATE = """You are Clair, a helpful AI assistant created by Michael Mlungisi Nkomo from Zimbabwe. You provide accurate, thoughtful responses and help users accomplish their goals efficiently."""

def copy_tokenizer_files():
    """Copy tokenizer files from base model to merged model."""
    print("=" * 60)
    print("STEP 1: Copying tokenizer files from base model...")
    print("=" * 60)
    
    # For Qwen2.5, the key tokenizer files are:
    tokenizer_files = [
        "tokenizer.json",
        "vocab.json",  # Qwen2.5 uses vocab.json for BPE
    ]
    
    merged_path = Path(MERGED_MODEL_PATH)
    
    for filename in tokenizer_files:
        src = Path(BASE_MODEL_PATH) / filename
        dst = merged_path / filename
        
        if src.exists():
            print(f"✓ Copying {filename}...")
            shutil.copy2(src, dst)
        else:
            print(f"⚠ Warning: {filename} not found in base model")
    
    # Also copy any .model files if they exist
    for src_file in Path(BASE_MODEL_PATH).glob("*.model"):
        dst_file = merged_path / src_file.name
        print(f"✓ Copying {src_file.name}...")
        shutil.copy2(src_file, dst_file)
    
    print(f"\nTokenizer files copied to {MERGED_MODEL_PATH}\n")

def fix_chat_template():
    """Re-apply Clair's chat template to tokenizer_config.json."""
    print("=" * 60)
    print("STEP 2: Re-applying Clair chat template...")
    print("=" * 60)
    
    tokenizer_config_path = Path(MERGED_MODEL_PATH) / "tokenizer_config.json"
    
    if not tokenizer_config_path.exists():
        print(f"❌ Error: {tokenizer_config_path} not found!")
        return False
    
    # Load existing config
    with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Update chat template
    print(f"Current chat template preview: {config.get('chat_template', 'None')[:100]}...")
    config['chat_template'] = CLAIR_CHAT_TEMPLATE
    
    # Save updated config
    with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Chat template updated to Clair's identity")
    print(f"New template: {CLAIR_CHAT_TEMPLATE}\n")
    
    return True

def resize_tokenizer():
    """Resize model embeddings to match tokenizer's vocabulary size."""
    print("=" * 60)
    print("STEP 3: Resizing model embeddings to match tokenizer...")
    print("=" * 60)
    
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    import torch
    
    try:
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)
        tokenizer_vocab = tokenizer.vocab_size
        print(f"Tokenizer vocab size: {tokenizer_vocab:,}")
        print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
        
        # Update config BEFORE loading model
        config = AutoConfig.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)
        old_vocab = config.vocab_size
        old_padding_idx = config.pad_token_id if hasattr(config, 'pad_token_id') else None
        print(f"Current config vocab size: {old_vocab:,}")
        print(f"Current config padding_idx: {old_padding_idx}")
        
        # Fix padding_idx if it's out of bounds
        if old_padding_idx is not None and old_padding_idx >= tokenizer_vocab:
            print(f"⚠️  Padding_idx {old_padding_idx} is out of bounds for vocab size {tokenizer_vocab}")
            print(f"Setting padding_idx to {tokenizer_vocab - 1}")
            config.pad_token_id = tokenizer_vocab - 1
            config.save_pretrained(MERGED_MODEL_PATH)
            print(f"✓ Config saved with fixed padding_idx")
        
        if old_vocab != tokenizer_vocab:
            print(f"Updating config.vocab_size from {old_vocab:,} to {tokenizer_vocab:,}...")
            config.vocab_size = tokenizer_vocab
            config.save_pretrained(MERGED_MODEL_PATH)
            print(f"✓ Config updated")
        
        # Load model with ignore_mismatch=True to bypass embedding size mismatch
        print(f"Loading model with ignore_mismatch_embeddings=True...")
        model = AutoModelForCausalLM.from_pretrained(
            MERGED_MODEL_PATH,
            trust_remote_code=True,
            ignore_mismatched_sizes=True
        )
        model_vocab = model.get_input_embeddings().weight.shape[0]
        print(f"Model loaded with vocab size: {model_vocab:,}")
        
        if tokenizer_vocab == model_vocab:
            print(f"\n✅ Vocab sizes already match! ({tokenizer_vocab:,} tokens)")
            return True
        
        diff = model_vocab - tokenizer_vocab
        print(f"\n⚠️  Model has {diff:,} extra tokens in embeddings")
        print(f"Resizing model embeddings to {tokenizer_vocab:,} tokens...")
        
        # Resize model embeddings (truncates the extra tokens)
        model.resize_token_embeddings(tokenizer_vocab)
        
        # Save the resized model
        model.save_pretrained(MERGED_MODEL_PATH)
        print(f"✓ Model embeddings resized and saved")
        
        # Verify
        model2 = AutoModelForCausalLM.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)
        model_vocab2 = model2.get_input_embeddings().weight.shape[0]
        if model_vocab2 == tokenizer_vocab:
            print(f"✅ SUCCESS: Vocab sizes now match! ({tokenizer_vocab:,} tokens)")
            return True
        else:
            print(f"❌ FAILED: Vocab sizes still don't match")
            print(f"   Tokenizer: {tokenizer_vocab:,}, Model: {model_vocab2:,}")
            return False
            
    except Exception as e:
        print(f"❌ Error during embedding resize: {e}")
        import traceback
        traceback.print_exc()
        return False

def convert_to_gguf():
    """Convert the fixed model to GGUF format."""
    print("=" * 60)
    print("STEP 4: Converting to GGUF format...")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(GGUF_OUTPUT_PATH, exist_ok=True)
    
    # Check what model files exist
    merged_path = Path(MERGED_MODEL_PATH)
    safetensors_files = list(merged_path.glob("*.safetensors"))
    print(f"Found safetensors files: {[f.name for f in safetensors_files]}")
    
    # Check for index file and remove it if it references non-existent shards
    index_file = merged_path / "model.safetensors.index.json"
    if index_file.exists():
        print(f"Found index file: {index_file.name}")
        # Check if the referenced shards exist
        import json
        with open(index_file, 'r') as f:
            index = json.load(f)
        referenced_files = list(index.get('weight_map', {}).values())
        print(f"Index references: {referenced_files}")
        
        # Remove index file - we'll use the single safetensors file directly
        print(f"Removing index file (will use single safetensors file)")
        index_file.unlink()
    
    # Run conversion script
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
        print(f"\n✓ BF16 GGUF created: {output_file}\n")
        return str(output_file)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during GGUF conversion: {e}")
        return None

def quantize_gguf(bf16_path):
    """Create quantized versions of the GGUF model."""
    print("=" * 60)
    print("STEP 5: Creating quantized versions...")
    print("=" * 60)
    
    llama_quantize = Path(LLAMA_CPP_PATH) / "bin/llama-quantize"
    
    quantizations = [
        ("Q4_K_M", "1.80GB"),
        ("Q5_K_M", "2.07GB"),
        ("Q3_K_M", "1.48GB"),
    ]
    
    for quant, size in quantizations:
        input_file = bf16_path
        output_file = str(Path(GGUF_OUTPUT_PATH) / f"clair-v2-{quant}.gguf")
        
        cmd = [
            "python3", "-c",
            f"import subprocess; subprocess.run(['{llama_quantize}', '{input_file}', '{output_file}', '{quant}'])"
        ]
        
        print(f"\nCreating {quant} ({size})...")
        try:
            result = subprocess.run(
                ["{llama_quantize}", input_file, output_file, quant],
                check=True,
                capture_output=False
            )
            print(f"✓ {quant} created: {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error creating {quant}: {e}")

def main():
    print("\n" + "=" * 60)
    print("FIXING CLAIR MODEL TOKENIZER")
    print("=" * 60 + "\n")
    
    # Step 1: Copy tokenizer files
    copy_tokenizer_files()
    
    # Step 2: Fix chat template
    if not fix_chat_template():
        print("❌ Failed to fix chat template. Aborting.")
        return
    
    # Step 3: Resize tokenizer to match model vocab
    if not resize_tokenizer():
        print("❌ Failed to resize tokenizer. Aborting.")
        return
    
    # Step 4: Convert to GGUF
    bf16_path = convert_to_gguf()
    if not bf16_path:
        print("❌ Failed to convert to GGUF. Aborting.")
        return
    
    # Step 5: Create quantized versions
    quantize_gguf(bf16_path)
    
    print("\n" + "=" * 60)
    print("✅ ALL STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nGGUF files available at: {GGUF_OUTPUT_PATH}")
    print("\nYou can now benchmark the fixed model:")
    print(f"  python notebooks/test_gguf_gpu.py")
    print()

if __name__ == "__main__":
    main()
