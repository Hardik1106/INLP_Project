#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH -n 1
#SBATCH --cpus-per-task=9
#SBATCH --gres=gpu:1
#SBATCH --mem=112G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/full_run_%j.txt
#SBATCH --job-name=cot_full_run
#SBATCH --nodelist=gnode067

# ===========================================================================
# CoT Mechanistic Faithfulness - Full Run
# ===========================================================================
# Complete pipeline execution with full dataset and all models
# - 1 GPU, 9 CPU cores, 112GB memory
# - 200+ samples (full dataset)
# - Cross-model sweep (Pythia-70M, [add more models as needed])
# - Multiple layers and intervention types
# - Duration: up to 4 days
# ===========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Setup and Activation
# ---------------------------------------------------------------------------

echo "=============================================="
echo "CoT Full Run - Starting at $(date)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPU cores: $SLURM_CPUS_PER_TASK"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Memory: 112GB"
echo "Time limit: 4 days"
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

# Full run expects GPU. Fail early if CUDA is unavailable in this environment.
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: CUDA is not available."
    echo "This usually means a torch/CUDA-driver mismatch in the active environment."
    echo "Fix the environment first, then resubmit this full run."
    exit 1
fi

# ---------------------------------------------------------------------------
# Pre-flight Checks
# ---------------------------------------------------------------------------

echo "Checking required files..."
[ -f "configs/default.yaml" ] || { echo "ERROR: configs/default.yaml not found!"; exit 1; }
[ -f "scripts/run_pipeline.py" ] || { echo "ERROR: scripts/run_pipeline.py not found!"; exit 1; }
echo "All required files present."
echo ""

# ---------------------------------------------------------------------------
# Run Full Pipeline with Cross-Model Sweep
# ---------------------------------------------------------------------------

echo "Starting full pipeline sweep..."
echo "Config: configs/default.yaml"
echo "Samples: 200+ (full dataset)"
echo "Models: Pythia-70M (expand to Llama-2-7b, Gemma-2b as needed)"
echo ""

# Option 1: Run the full sweep script
bash scripts/run_experiments.sh configs/default.yaml

EXIT_CODE=$?

# ---------------------------------------------------------------------------
# Post-Processing and Cleanup
# ---------------------------------------------------------------------------

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Full Run COMPLETED SUCCESSFULLY at $(date)"
    python -c "import torch; torch.cuda.empty_cache()"
    echo "GPU cache cleared."
else
    echo "Full Run FAILED with exit code $EXIT_CODE at $(date)"
    echo "Check logs/full_run_${SLURM_JOB_ID}.txt for details"
fi
echo "=============================================="
echo ""
echo "Results location: outputs/"
ls -lh outputs/ 2>/dev/null | head -20

exit $EXIT_CODE
