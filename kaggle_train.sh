#!/bin/bash
# MaxText Llama3 200M MoE Training on Kaggle
# 6 layers, MoE on layers 2/4/6, 8 experts, top-2 activation
# TinyStories dataset, 2 epochs, FP16, 2x T4 15GB


# 1. Install Python Dependencies
# ==============================================================================

# ==============================================================================
# 2. Configure GCS Authentication (Optional)
# ==============================================================================

echo "Step 2: Configuring GCS authentication..."

if [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "GCS credentials found. Setting up authentication..."
    mkdir -p /root/.config/gcloud
    echo "$GOOGLE_APPLICATION_CREDENTIALS" > /root/.config/gcloud/application_default_credentials.json
    export GOOGLE_APPLICATION_CREDENTIALS="/root/.config/gcloud/application_default_credentials.json"
    export GOOGLE_APPLICATION_CREDENTIALS_JSON="$GOOGLE_APPLICATION_CREDENTIALS"
    echo "GCS authentication configured!"
else
    echo "No GCS credentials found. Using local checkpoint storage."
    echo "To use GCS, add GOOGLE_APPLICATION_CREDENTIALS to Kaggle Secrets."
fi
echo ""

# ==============================================================================
# 3. Set Environment Variables
# ==============================================================================

echo "Step 3: Configuring environment variables..."

# Kaggle automatically sets these secrets
export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

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

# Configure JAX for multi-GPU training
# Kaggle 2x T4 setup
export JAX_ENABLE_X64=False
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_force_compilation_parallelism=1"

# Configure JAX for multi-GPU training
# Kaggle 2x T4 setup
export JAX_ENABLE_X64=False
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_force_compilation_parallelism=1"

# Configure checkpoint storage based on GCS credentials
export TF_FORCE_GPU_ALLOW_GROWTH=true

if [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    # Use GCS for checkpoint storage
    export GCS_BUCKET="gs://gwhwh153td-maxtext-outputs"
    export BASE_OUTPUT_DIRECTORY="gs://gwhwh153td-maxtext-outputs"
    echo "Using GCS checkpoint storage: gs://gwhwh153td-maxtext-outputs"
else
    # Use local storage
    export BASE_OUTPUT_DIRECTORY="/kaggle/working/checkpoints"
    echo "Using local checkpoint storage: /kaggle/working/checkpoints"
fi

# Create local checkpoint directory if needed
mkdir -p /kaggle/working/checkpoints

# Run MaxText training
python -m MaxText.train \
  --workdir=/kaggle/working \
  --config=/kaggle/working/configs/llama3_moe_200m_tinystories.yml \
  --skip_jax_distributed_system=True \
  --base_output_directory="$BASE_OUTPUT_DIRECTORY" \
  2>&1 | tee training.log

echo ""
echo "==============================================================================="
echo "Training completed!"
echo "Check training.log for detailed output."
echo "Model checkpoints will be saved to: /kaggle/working/llama3_200m_moe_tinystories"
