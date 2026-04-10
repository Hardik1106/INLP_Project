#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/test_run_%j.txt
#SBATCH --job-name=cot_test
#SBATCH --nodelist=gnode067

# ===========================================================================
# CoT Mechanistic Faithfulness - Test Run (Small)
# ===========================================================================
# Quick verification that the pipeline works with minimal resources
# - 1 GPU, 4 CPU cores, 32GB memory
# - 50 samples (small dataset)
# - Single model (Pythia-70M)
# - Limited layers (layer 2 only)
# ===========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Setup and Activation
# ---------------------------------------------------------------------------

echo "=============================================="
echo "CoT Test Run - Starting at $(date)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Create logs directory
mkdir -p logs

# Activate conda environment (adjust if using venv or other environment manager)
# Uncomment and modify the conda path as needed:
source /home2/hardikchadha/miniconda3/etc/profile.d/conda.sh || { echo "ERROR: conda.sh not found at /home2/hardikchadha/miniconda3/etc/profile.d/conda.sh"; exit 1; }
conda activate inlp_project

# Or if using venv:
# source /path/to/venv/bin/activate

# Set working directory
cd /home2/hardikchadha/INLP_Project

# Verify Python and CUDA
echo "Python version: $(python --version)"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# ---------------------------------------------------------------------------
# Run Test Pipeline
# ---------------------------------------------------------------------------

echo "Starting test pipeline with small dataset..."
echo "Config: configs/default.yaml"
echo "Samples: 50 (test)"
echo "Model: from config (configs/default.yaml)"
echo ""

DEVICE="cuda"
if ! python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "WARNING: CUDA not available in this environment; falling back to CPU for test run."
    DEVICE="cpu"
fi

echo "Using device: ${DEVICE}"
echo ""

python scripts/run_pipeline.py \
    --config configs/default.yaml \
    --num-samples 50 \
    --device "${DEVICE}" \
    2>&1 | tee -a logs/test_run_${SLURM_JOB_ID}.log

EXIT_CODE=$?

# ---------------------------------------------------------------------------
# Cleanup and Summary
# ---------------------------------------------------------------------------

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Test Run COMPLETED SUCCESSFULLY at $(date)"
else
    echo "Test Run FAILED with exit code $EXIT_CODE at $(date)"
fi
echo "=============================================="

exit $EXIT_CODE
