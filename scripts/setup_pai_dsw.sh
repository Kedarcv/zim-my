#!/bin/bash
# Setup script for PAI-DSW environment
# Run this after launching PAI-DSW with PyTorch image

set -e

echo "=========================================="
echo "Zim-my ADTC 2026 - Environment Setup"
echo "=========================================="

# Update system packages
echo "[1/8] Updating system packages..."
sudo apt-get update -qq

# Install system dependencies
echo "[2/8] Installing system dependencies..."
sudo apt-get install -y -qq \
    git \
    wget \
    curl \
    htop \
    tree \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev

# Upgrade pip
echo "[3/8] Upgrading pip..."
pip install --upgrade pip -q

# Install PyTorch (if not already installed)
echo "[4/8] Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" || {
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
}

# Install fine-tuning dependencies
echo "[5/8] Installing fine-tuning dependencies..."
pip install -q \
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
    "xformers==0.0.28.post3" \
    "trl<0.9.0" \
    peft \
    accelerate \
    bitsandbytes \
    datasets \
    transformers \
    sentencepiece \
    protobuf

# Install inference and RAG dependencies
echo "[6/8] Installing inference and RAG dependencies..."
pip install -q \
    llama-cpp-python \
    chromadb \
    sentence-transformers \
    langchain \
    langchain-community

# Install UI and utilities
echo "[7/8] Installing UI and utilities..."
pip install -q \
    streamlit \
    gradio \
    psutil \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    tqdm \
    rich \
    jupyter \
    ipywidgets

# Install Alibaba Cloud tools
echo "[8/8] Installing Alibaba Cloud tools..."
pip install -q \
    modelscope \
    oss2

# Verify installation
echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Verifying key packages..."
python << 'EOF'
import torch
import transformers
import peft
import bitsandbytes
import datasets
import chromadb
import streamlit

print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"✓ Transformers: {transformers.__version__}")
print(f"✓ PEFT: {peft.__version__}")
print(f"✓ Datasets: {datasets.__version__}")
print(f"✓ ChromaDB: {chromadb.__version__}")
print(f"✓ Streamlit: {streamlit.__version__}")
print("")
print("Environment ready for Zim-my development!")
EOF

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo "1. Run: bash scripts/download_data.sh"
echo "2. Open: notebooks/01_data_prep.ipynb"
echo "3. Follow the data preparation workflow"
echo ""
