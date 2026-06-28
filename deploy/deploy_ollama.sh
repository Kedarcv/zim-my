#!/bin/bash
# Deploy Clair v3 to Ollama
# Run this on PAI-DSW after the model has been built

set -e

GGUF_DIR="/mnt/workspace/zim-my/models/clair-gguf-v3"
DEPLOY_DIR="/mnt/workspace/zim-my/deploy"

echo "============================================================"
echo "  Clair v3 - Ollama Deployment"
echo "============================================================"
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Installing..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✓ Ollama installed"
fi

# Start Ollama server if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama server..."
    ollama serve &
    sleep 3
    echo "✓ Ollama server started"
fi

# Copy GGUF files to deploy directory
mkdir -p "$DEPLOY_DIR"
echo ""
echo "Copying GGUF files to deploy directory..."
cp "$GGUF_DIR/clair-v3-Q4_K_M.gguf" "$DEPLOY_DIR/"
cp "$GGUF_DIR/clair-v3-Q5_K_M.gguf" "$DEPLOY_DIR/"
cp "$GGUF_DIR/clair-v3-Q3_K_M.gguf" "$DEPLOY_DIR/"
echo "✓ GGUF files copied"

# Copy Modelfiles
cp /mnt/workspace/zim-my/deploy/Modelfile "$DEPLOY_DIR/" 2>/dev/null || true
cp /mnt/workspace/zim-my/deploy/Modelfile.q5 "$DEPLOY_DIR/" 2>/dev/null || true

cd "$DEPLOY_DIR"

# Create Q4_K_M model (default - best balance of size/quality)
echo ""
echo "Creating Ollama model: clair (Q4_K_M, 1.80 GB)..."
ollama create clair -f Modelfile
echo "✓ Model 'clair' created"

# Create Q5_K_M model (higher quality)
echo ""
echo "Creating Ollama model: clair-q5 (Q5_K_M, 2.07 GB)..."
ollama create clair-q5 -f Modelfile.q5
echo "✓ Model 'clair-q5' created"

# List models
echo ""
echo "============================================================"
echo "  Installed Models"
echo "============================================================"
ollama list

# Quick test
echo ""
echo "============================================================"
echo "  Quick Test"
echo "============================================================"
echo ""
echo "Testing: 'What is your name?'"
echo ""
ollama run clair "What is your name?" 2>/dev/null || echo "(Run 'ollama run clair' manually to test)"

echo ""
echo "============================================================"
echo "  Deployment Complete!"
echo "============================================================"
echo ""
echo "Usage:"
echo "  ollama run clair              # Interactive chat"
echo "  ollama run clair-q5           # Higher quality version"
echo ""
echo "API access:"
echo "  curl http://localhost:11434/api/chat -d '{"
echo "    \"model\": \"clair\","
echo "    \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]"
echo "  }'"
echo ""
echo "Open WebUI (if installed):"
echo "  http://localhost:3000"
echo ""
