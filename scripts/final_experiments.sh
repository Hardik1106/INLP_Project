#!/bin/bash

# ========================================
# WAIT FOR PROCESS TO COMPLETE
# ========================================
PID=4070473  # ← Your running process ID

echo "Waiting for process $PID to complete..."
echo "Start time: $(date)"

# Check if process exists and wait for it to finish
while ps -p $PID > /dev/null; do
    echo "Process $PID still running... $(date)"
    sleep 1000
done

echo "Process $PID completed at $(date)"
echo ""

# ========================================
# THEN RUN THE PIPELINE COMMANDS
# ========================================

# Configuration
CONDA_ENV="cot2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../inference_logs"
echo $LOG_DIR

# Array of config files to run
CONFIG_FILES=(
    "configs/70m_cot.yaml"
    # "configs/2_8b_cot.yaml"
    "configs/70m_nocot.yaml"
    # "configs/2_8b_nocot.yaml"
)

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Activate conda environment
source /ssd_scratch/miniconda3/etc/profile.d/conda.sh
conda activate cot2

echo "Starting pipeline runs with config files..."
echo "Conda environment: $CONDA_ENV"
echo "Logs directory: $LOG_DIR"
echo ""

# Run each config file
for i in "${!CONFIG_FILES[@]}"; do
    config="${CONFIG_FILES[$i]}"
    run_num=$((i + 1))
    total=${#CONFIG_FILES[@]}
    
    # Extract config name for log file
    config_name=$(basename "$config" .yaml)
    
    echo "=========================================="
    echo "Run $run_num of $total - $(date)"
    echo "Config: $config"
    echo "=========================================="
    
    cd "$SCRIPT_DIR" || exit 1
    python -u run_pipeline.py --config "$config"  > "$LOG_DIR/${config_name}_run.log" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✓ Run $run_num ($config_name) completed successfully"
    else
        echo "✗ Run $run_num ($config_name) failed - check log at $LOG_DIR/${config_name}_run.log"
    fi
    echo ""
done

echo "All config runs completed at $(date)"