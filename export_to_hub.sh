#!/bin/bash
# Push trained model to HuggingFace Hub
# Exports MaxText checkpoint to HuggingFace format and pushes

set -e

echo "=== Export MaxText Checkpoint to HuggingFace Hub ==="
echo ""

# Check environment variables
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN is not set!"
    echo "Please set HF_TOKEN environment variable."
    exit 1
fi

# User can override these
REPO_NAME="${HF_REPO:-lyyh/llama3-200m-moe}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/kaggle/working/llama3_200m_moe_tinystories}"

echo "Configuration:"
echo "  - HuggingFace repo: $REPO_NAME"
echo "  - Checkpoint path: $CHECKPOINT_PATH"
echo ""

# ==============================================================================
# Install HuggingFace dependencies
# ==============================================================================

echo "Step 1: Installing HuggingFace dependencies..."

pip install --upgrade huggingface_hub transformers safetensors

echo "Dependencies installed!"
echo ""

# ==============================================================================
# Step 2: Convert MaxText checkpoint to HuggingFace format
# ==============================================================================

echo "Step 2: Converting MaxText checkpoint to HuggingFace format..."
echo ""

cd /kaggle/working/maxtext

# Run checkpoint conversion
# MaxText provides conversion utilities in src/MaxText/utils/ckpt_conversion/
python -c "
from maxtext.utils.ckpt_conversion.to_huggingface import maxtext_to_hf
import os

ckpt_path = '$CHECKPOINT_PATH'
hf_repo_name = '$REPO_NAME'

print(f'Converting checkpoint from {ckpt_path}...')
print(f'Target HF repo: {hf_repo_name}')

# This function converts MaxText Orbax checkpoint to HuggingFace format
maxtext_to_hf(
    maxtext_checkpoint_path=ckpt_path,
    hf_repo_name=hf_repo_name,
    output_dir='/kaggle/working/hf_export',
)

print('Conversion completed!')
print(f'HF files saved to: /kaggle/working/hf_export')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Checkpoint conversion failed!"
"
    exit 1
fi

echo ""
echo "Conversion successful!"
echo ""

# ==============================================================================
# Step 3: Upload to HuggingFace Hub
# ==============================================================================

echo "Step 3: Uploading to HuggingFace Hub..."
echo ""

pip install --upgrade huggingface_hub

# Push to Hub
cd /kaggle/working/hf_export

huggingface-cli login --token "$HF_TOKEN"
huggingface-cli upload "$REPO_NAME" . --repo-type model --private false

echo ""
echo "======================================================================"
echo "Export completed!"
echo "  - HF repo: $REPO_NAME"
echo "  - Local HF files: /k"
echo "working/hf_export"
echo "You can view your model at: https://huggingface.co/$REPO_NAME"
