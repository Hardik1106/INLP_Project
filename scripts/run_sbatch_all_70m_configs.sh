#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/full_run_all_70m_%j.txt
#SBATCH --job-name=cot_70m_all_cfg
#SBATCH --nodelist=gnode067

# ===========================================================================
# CoT Mechanistic Faithfulness - 70M Multi-Config Run
# ===========================================================================
# Runs all newly added 70M experiment configs in one SLURM job.
# - Reuses the same environment setup as run_sbatch.sh
# - Executes configs sequentially
# - Tracks per-config status and returns non-zero if any fail
# ===========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Setup and Activation
# ---------------------------------------------------------------------------

echo "=============================================="
echo "CoT 70M Multi-Config Run - Starting at $(date)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPU cores: $SLURM_CPUS_PER_TASK"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Memory: 64GB"
echo "Time limit: 4 days"
echo ""

# Create logs directory
mkdir -p logs

# Activate conda environment
source /home2/hardikchadha/miniconda3/etc/profile.d/conda.sh || { echo "ERROR: conda.sh not found at /home2/hardikchadha/miniconda3/etc/profile.d/conda.sh"; exit 1; }
conda activate inlp_project

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
[ -f "scripts/run_pipeline.py" ] || { echo "ERROR: scripts/run_pipeline.py not found!"; exit 1; }

echo "All required files present."
echo ""

# ---------------------------------------------------------------------------
# Config List (all newly created 70M configs)
# ---------------------------------------------------------------------------

CONFIGS=(
  "configs/70m_deep_layers.yaml"
  "configs/70m_triviaqa_contrast.yaml"
  "configs/70m_cross_layer_mapping.yaml"
  "configs/70m_causal_necessity_final.yaml"
  "configs/70m_pretrained_reference.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  [ -f "$cfg" ] || { echo "ERROR: Missing config file: $cfg"; exit 1; }
done

STATUS_FILE="logs/all_70m_config_status_${SLURM_JOB_ID}.txt"
: > "$STATUS_FILE"

# ---------------------------------------------------------------------------
# Run all configs
# ---------------------------------------------------------------------------

echo "Starting multi-config pipeline runs..."
echo "Total configs: ${#CONFIGS[@]}"
echo "Status file: $STATUS_FILE"
echo ""

OVERALL_EXIT=0
INDEX=0

for cfg in "${CONFIGS[@]}"; do
    INDEX=$((INDEX + 1))

    echo "======================================================"
    echo "[$INDEX/${#CONFIGS[@]}] Running config: $cfg"
    echo "Started at: $(date)"
    echo "======================================================"

    set +e
    python scripts/run_pipeline.py --config "$cfg"
    CFG_EXIT=$?
    set -e

    if [ $CFG_EXIT -eq 0 ]; then
        echo "SUCCESS: $cfg" | tee -a "$STATUS_FILE"
        echo "[$INDEX/${#CONFIGS[@]}] COMPLETED: $cfg"
    else
        echo "FAILED (exit $CFG_EXIT): $cfg" | tee -a "$STATUS_FILE"
        echo "[$INDEX/${#CONFIGS[@]}] FAILED: $cfg (exit $CFG_EXIT)"
        OVERALL_EXIT=1
    fi

    echo "Finished at: $(date)"
    echo ""

    # Clear GPU memory between runs
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
done

# ---------------------------------------------------------------------------
# Post-Processing and Cleanup
# ---------------------------------------------------------------------------

echo ""
echo "=============================================="
if [ $OVERALL_EXIT -eq 0 ]; then
    echo "All 70M config runs COMPLETED SUCCESSFULLY at $(date)"
else
    echo "One or more 70M config runs FAILED at $(date)"
fi
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
echo "GPU cache cleared."
echo "=============================================="

echo ""
echo "Per-config status:"
cat "$STATUS_FILE"

echo ""
echo "Results locations:"
ls -lh outputs/ 2>/dev/null | head -50

exit $OVERALL_EXIT
