#!/bin/bash
# MaxText Llama3 200M MoE Training on Kaggle
# 6 layers, MoE on layers 2/4/6, 8 experts, top-2 activation
# TinyStories dataset, 2 epochs, FP16, 2x T4 15GB

set -e  # Exit on error
set -o pipefail  # Output pip install progress

echo "=== MaxText Llama3 200M MoE Training on Kaggle ==="
echo ""

# ==============================================================================
# 1. Install Python Dependencies
# ==============================================================================

echo "Step 1: Installing Python dependencies..."

pip install --upgrade pip setuptools wheel

# JAX and JAX ecosystem
pip install jax==0.4.37 jaxlib==0.4.37 -q
pip install flax==0.9.0 optax==0.2.1

# Deep learning frameworks
pip install --upgrade tensorflow==2.16.1
pip install --upgrade transformers[accelerate]

# Data processing
pip install --upgrade datasets grain
pip install --download tqdm

# Monitoring
pip install --upgrade wandb

# Optional: Install PyTorch for better dataset streaming performance
pip install --upgrade torch torchvision

echo "Python dependencies installed successfully!"
echo ""

# ==============================================================================
# 2. Install MaxText
# ==============================================================================

echo "Step 2: Installing MaxText from source..."

# Create setup.py if not exists (for newer MaxText versions)
cd /kaggle/working/maxtext
if [ ! -f "setup.py" ]; then
    echo "Creating setup.py for PyPI-style installation..."
    cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="maxtext",
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"maxtext": "src/MaxText"},
    install_requires=[
        "jax>=0.4.37",
        "jaxlib>=0.4.37",
        "flax>=0.9.0",
        "optax>=0.2.1",
        "tensorflow>=2.16.1",
        "transformers[accelerate]>=4.0.0",
        "datasets>=2.0.0",
        "grain>=0.3.0",
        "wandb>=0.17.0",
        "tqdm>=4.0.0",
    ],
    python_requires=">=3.10",
)
EOF
fi

# Install MaxText in editable mode
pip install -e . -v

echo "MaxText installed successfully!"
cd /kaggle/working
echo ""

# ==============================================================================
# 3. Set Environment Variables
# ==============================================================================

echo "Step 3: Configuring environment variables..."

# Kaggle automatically sets these secrets
export HF_TOKEN="${HF_TOKEN:-}"
export WANW_API_KEY="${WANDB_API_KEY:-}"

# Verify secrets are set
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN is not set. TinyStories and Llama3 tokenizer access will fail!"
    echo "Please add HF_TOKEN to Kaggle Secrets."
    exit 1
fi

if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY is not set. Training will proceed without W&B logging."
    echo "Please add WANDB_API_KEY to Kaggle Secrets."
fi

echo "Environment configured!"
echo ""

# ==============================================================================
# 4. Start Training
# ==============================================================================

echo "Step 4: Starting MaxText training..."
echo ""

# Display training configuration
echo "Configuration:"
echo "  - Model: Llama3 200M MoE (6 layers, d_model=768, 8 experts, top-2)"
echo "  - Hardware: 2x T4 15GB"
echo "  - Dataset: TinyStories (HuggingFace, streaming)"
echo "  - Epochs: 2"
echo "  - Checkpoint period: 200 steps"
echo "  - Precision: FP16"
echo "  - W&B: ${WANDB_API_KEY:+enabled, disabled}"
echo "  - HF push: ${HF_TOKEN:+enabled}"
echo ""

# Set maxtext PYTHONPATH
export PYTHONPATH="/kaggle/working/maxtext/src:$PYTHONPATH"

# Run MaxText training
# Note: Config path is relative to working directory
python -m maxtext.train \
  --workdir=/kaggle/working \
  --config=/kaggle/working/configs/llama3_moe_200m_tinystories.yml \
  2>&1 | tee training.log

echo ""
echo "==============================================================================="
echo "Training completed!"
echo "Check training.log for detailed output."
echo "Model checkpoints will be saved to: /kaggle/working/llama3_200m_moe_tinystories"
