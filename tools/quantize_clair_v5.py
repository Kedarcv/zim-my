#!/usr/bin/env python3
"""
Quantize Clair v5 GGUF to smaller formats for faster inference.

This script downloads llama.cpp and builds the quantize tool,
then creates Q4_K_M, Q5_K_M, and Q3_K_M versions.
"""

import subprocess
import os
from pathlib import Path

GGUF_PATH = "/mnt/workspace/zim-my/models/clair-gguf-v5/clair-v5-float16.gguf"
OUTPUT_DIR = "/mnt/workspace/zim-my/models/clair-gguf-v5/"
LLAMA_CPP_DIR = "/tmp/llama.cpp"

def clone_llama_cpp():
    """Clone llama.cpp repository."""
    print("=" * 60)
    print("Cloning llama.cpp...")
    print("=" * 60)
    
    if Path(LLAMA_CPP_DIR).exists():
        print("✓ llama.cpp already exists")
        return True
    
    cmd = ["git", "clone", "https://github.com/ggerganov/llama.cpp.git", LLAMA_CPP_DIR]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"✗ Failed to clone: {result.stderr}")
        return False
    
    print("✓ Cloned successfully")
    return True

def build_quantize():
    """Build llama-quantize tool."""
    print("\n" + "=" * 60)
    print("Building llama-quantize...")
    print("=" * 60)
    
    build_dir = Path(LLAMA_CPP_DIR) / "build"
    build_dir.mkdir(exist_ok=True)
    
    # Run cmake
    print("Running cmake...")
    result = subprocess.run(
        ["cmake", ".."],
        cwd=build_dir,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"✗ CMake failed: {result.stderr[:500]}")
        return False
    
    # Build llama-quantize
    print("Building...")
    result = subprocess.run(
        ["make", "llama-quantize", "-j"],
        cwd=build_dir,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"✗ Build failed: {result.stderr[:500]}")
        return False
    
    print("✓ Build complete")
    return True

def quantize_models():
    """Create quantized versions."""
    print("\n" + "=" * 60)
    print("Quantizing models...")
    print("=" * 60)
    
    quantize_bin = Path(LLAMA_CPP_DIR) / "build" / "bin" / "llama-quantize"
    output_dir = Path(OUTPUT_DIR)
    
    quantizations = [
        ("Q4_K_M", "clair-v5-Q4_K_M.gguf", "~2GB, fastest"),
        ("Q5_K_M", "clair-v5-Q5_K_M.gguf", "~2.5GB, balanced"),
        ("Q3_K_M", "clair-v5-Q3_K_M.gguf", "~1.5GB, smallest"),
    ]
    
    for quant_type, output_name, description in quantizations:
        output_file = output_dir / output_name
        print(f"\nQuantizing to {quant_type} ({description})...")
        
        cmd = [str(quantize_bin), GGUF_PATH, str(output_file), quant_type]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"✗ Failed: {result.stderr[:200]}")
            continue
        
        size_gb = output_file.stat().st_size / 1024**3
        print(f"✓ Created: {output_name} ({size_gb:.2f} GB)")
    
    print("\n" + "=" * 60)
    print("✓ Quantization complete!")
    print("=" * 60)
    print(f"\nQuantized models saved to: {OUTPUT_DIR}")
    print("\nRecommendations:")
    print("  - Q4_K_M: Best for most use cases (fast, good quality)")
    print("  - Q5_K_M: Better quality, slightly slower")
    print("  - Q3_K_M: Smallest size, lower quality")
    
    return True

def main():
    """Run all steps."""
    print("=" * 60)
    print("Clair v5 - Quantize GGUF Models")
    print("=" * 60)
    print("\nThis will create smaller, faster versions of your model:")
    print("  - Q4_K_M: ~2GB (recommended)")
    print("  - Q5_K_M: ~2.5GB")
    print("  - Q3_K_M: ~1.5GB")
    
    input("\nPress Enter to continue...")
    
    if not clone_llama_cpp():
        return False
    
    if not build_quantize():
        return False
    
    if not quantize_models():
        return False
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("=" * 60)
    print("1. Update deploy/Modelfile to use Q4_K_M:")
    print("   FROM ../models/clair-gguf-v5/clair-v5-Q4_K_M.gguf")
    print("\n2. Create Ollama model:")
    print("   ollama create r245142r/Clair-3B -f deploy/Modelfile")
    print("\n3. Push to Ollama:")
    print("   ollama push r245142r/Clair-3B")
    
    return True

if __name__ == "__main__":
    main()
