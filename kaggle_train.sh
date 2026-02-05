#!/bin/bash
# MaxText Llama3 200M MoE Training on Kaggle
# 6 layers, MoE on layers 2/4/6, 8 experts, top-2 activation
# TinyStories dataset, 2 epochs, FP16, 2x T4 15GB


# 1. Install Python Dependencies
# ==============================================================================

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
python -m MaxText.train \
  --workdir=/kaggle/working \
  --config=/kaggle/working/configs/llama3_moe_200m_tinystories.yml \
  2>&1 | tee training.log

echo ""
echo "==============================================================================="
echo "Training completed!"
echo "Check training.log for detailed output."
echo "Model checkpoints will be saved to: /kaggle/working/llama3_200m_moe_tinystories"
