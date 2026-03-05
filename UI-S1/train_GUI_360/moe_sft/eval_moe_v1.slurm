#!/bin/bash
#SBATCH --job-name=eval_moe_v1
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/eval_moe_v1_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/eval_moe_v1_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Evaluate MoE v1 Baseline (Merged Model)
# ============================================
# MoE v1 router collapsed → expert_1 weight ≈ 1.0
# Merged expert_1 LoRA into base model for vLLM serving
# ============================================

set -e

# ============================================
# Configuration
# ============================================
PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
TRAIN_DIR="$PROJECT_DIR/train_GUI_360"
EVAL_DIR="$TRAIN_DIR/GUI-360-eval"
LOG_DIR="$TRAIN_DIR/logs"
RESULTS_DIR="$TRAIN_DIR/moe_sft/output/eval_moe_v1"

# Step 1: Merge MoE expert_1 into base model (if not already done)
MERGE_SCRIPT="$TRAIN_DIR/moe_sft/merge_moe_to_hf.py"
MERGED_MODEL="$TRAIN_DIR/moe_sft/output/moe_v1_merged"
MOE_CHECKPOINT="$TRAIN_DIR/moe_sft/output/moe_sft_v1/final"
BASE_MODEL="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"

MODEL_TYPE="qwen2.5_vl_7b"

# Dataset path
DATASET_ROOT="$PROJECT_DIR/datasets/GUI-360/test"

# vLLM server configuration
VLLM_PORT=19807
VLLM_HOST="localhost"

# Evaluation configuration
EVAL_THREADS=5

# ============================================
# Environment Setup
# ============================================
echo "=========================================="
echo "MoE v1 Baseline Evaluation"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

# Activate conda environment
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

# CUDA setup (needed by deepspeed import during save_pretrained)
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Create directories
mkdir -p $LOG_DIR
mkdir -p $RESULTS_DIR

# ============================================
# Step 1: Merge MoE → HF (if needed)
# ============================================
if [ ! -f "$MERGED_MODEL/model.safetensors.index.json" ] && [ ! -f "$MERGED_MODEL/model.safetensors" ]; then
    echo "=========================================="
    echo "Merging MoE expert_1 into base model..."
    echo "=========================================="

    python $MERGE_SCRIPT \
        --base_model $BASE_MODEL \
        --moe_checkpoint $MOE_CHECKPOINT \
        --output_dir $MERGED_MODEL \
        --expert_idx 1

    echo "Merge complete."
else
    echo "Merged model already exists at $MERGED_MODEL"
fi

# Verify merged model
MODEL_NAME=$MERGED_MODEL
if [ ! -d "$MODEL_NAME" ]; then
    echo "ERROR: Merged model not found at $MODEL_NAME"
    exit 1
fi
echo "Model: $MODEL_NAME"

# ============================================
# Step 2: Start vLLM Server
# ============================================
echo "=========================================="
echo "Starting vLLM Server (port $VLLM_PORT)"
echo "=========================================="

# Kill any existing vLLM processes on this port
pkill -f "vllm.entrypoints.openai.api_server.*${VLLM_PORT}" || true
sleep 2

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee $LOG_DIR/vllm_moe_v1_${SLURM_JOB_ID}.log &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# Wait for server to be ready
echo "Waiting for vLLM server to be ready..."
MAX_WAIT=600
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s http://${VLLM_HOST}:${VLLM_PORT}/health > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
    echo "Waiting... ($WAIT_COUNT/$MAX_WAIT seconds)"
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM server failed to start within $MAX_WAIT seconds"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# ============================================
# Step 3: Run Evaluations
# ============================================
cd $EVAL_DIR

API_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

run_eval() {
    local EVAL_TYPE=$1
    echo "=========================================="
    echo "Running $EVAL_TYPE evaluation"
    echo "=========================================="

    local OUTPUT_DIR="$RESULTS_DIR/${EVAL_TYPE}_${TIMESTAMP}"
    mkdir -p $OUTPUT_DIR

    python evaluation.py \
        --root_dir $DATASET_ROOT \
        --type $EVAL_TYPE \
        --model_type $MODEL_TYPE \
        --model_name $MODEL_NAME \
        --api_url $API_URL \
        --threads $EVAL_THREADS \
        --output_dir $OUTPUT_DIR

    echo "$EVAL_TYPE completed. Results: $OUTPUT_DIR"
}

run_eval "grounding"
run_eval "action_prediction"
run_eval "action_prediction_a11y"

# ============================================
# Cleanup
# ============================================
echo "=========================================="
echo "Stopping vLLM Server"
echo "=========================================="
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server.*${VLLM_PORT}" || true

# ============================================
# Summary
# ============================================
echo "=========================================="
echo "MoE v1 Evaluation Completed"
echo "=========================================="
echo "End time: $(date)"
echo "Results: $RESULTS_DIR"
echo ""
echo "Results:"
ls -la $RESULTS_DIR/ 2>/dev/null
echo ""
echo "Summaries:"
cat $RESULTS_DIR/*/evaluation_summary_*.json 2>/dev/null || echo "No summaries found"
echo "=========================================="
