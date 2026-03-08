#!/usr/bin/env bash
# ===========================================================================
# Cross-Model Sweep Script
# ===========================================================================
# Runs the full CoT mechanistic faithfulness pipeline across multiple
# model architectures and layer configurations.
#
# Usage:
#   bash scripts/run_experiments.sh [--config configs/default.yaml] [--dry-run]
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG="${1:-configs/default.yaml}"
DRY_RUN="${2:-false}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="outputs/sweep_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "CoT Mechanistic Faithfulness - Cross-Model Sweep"
echo "=============================================="
echo "Config:    $CONFIG"
echo "Timestamp: $TIMESTAMP"
echo "Log dir:   $LOG_DIR"
echo ""

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
# Format: "model_name|tl_name|sae_release|layers"
# Layers are comma-separated

MODELS=(
    "EleutherAI/pythia-70m|pythia-70m|pythia-70m-res-jb|1,2,3"
    "EleutherAI/pythia-2.8b|pythia-2.8b|pythia-2.8b-res-jb|2,12,14,16"
)

# Uncomment additional models as compute allows:
# MODELS+=(
#     "meta-llama/Llama-2-7b-hf|llama-2-7b|llama-2-7b-res-jb|2,12,16,24"
#     "google/gemma-2b|gemma-2b|gemma-2b-res-jb|2,8,12,16"
# )

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

NUM_SAMPLES=200
K_VALUES="10,50,100"
DEVICE="cuda"

# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

TOTAL_MODELS=${#MODELS[@]}
CURRENT=0

for MODEL_SPEC in "${MODELS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    # Parse model spec
    IFS='|' read -r MODEL_NAME TL_NAME SAE_RELEASE LAYERS <<< "$MODEL_SPEC"
    
    echo ""
    echo "======================================================"
    echo "[$CURRENT/$TOTAL_MODELS] Running: $MODEL_NAME"
    echo "======================================================"
    echo "  TL Name:     $TL_NAME"
    echo "  SAE Release: $SAE_RELEASE"
    echo "  Layers:      $LAYERS"
    echo ""
    
    # Create model-specific output directory
    MODEL_DIR="$LOG_DIR/${TL_NAME}"
    mkdir -p "$MODEL_DIR"
    
    if [ "$DRY_RUN" = "--dry-run" ] || [ "$DRY_RUN" = "true" ]; then
        echo "  [DRY RUN] Would run pipeline for $MODEL_NAME"
        continue
    fi
    
    # Run the pipeline
    python scripts/run_pipeline.py \
        --config "$CONFIG" \
        --model "$TL_NAME" \
        --num-samples "$NUM_SAMPLES" \
        --device "$DEVICE" \
        2>&1 | tee "$MODEL_DIR/run.log"
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "  [SUCCESS] $MODEL_NAME completed"
        echo "$MODEL_NAME: SUCCESS" >> "$LOG_DIR/sweep_status.txt"
    else
        echo "  [FAILED] $MODEL_NAME exited with code $EXIT_CODE"
        echo "$MODEL_NAME: FAILED (exit $EXIT_CODE)" >> "$LOG_DIR/sweep_status.txt"
    fi
    
    # Clear GPU memory between models
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    
done

echo ""
echo "======================================================"
echo "Sweep complete. Results in: $LOG_DIR"
echo "======================================================"

# Print summary
if [ -f "$LOG_DIR/sweep_status.txt" ]; then
    echo ""
    echo "--- Sweep Status ---"
    cat "$LOG_DIR/sweep_status.txt"
fi
