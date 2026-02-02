#!/usr/bin/env bash
set -euo pipefail

# Absolute paths
SCRIPT="$(pwd)/src/features/generate_3di_train.py"
ENV_NAME="cafa6"
SCREEN_NAME="generate_3di_train_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$(pwd)/logs"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${SCREEN_NAME}.log"

echo "Starting screen session $SCREEN_NAME running $SCRIPT"

# Start detached screen that initializes conda safely, activates env, runs the script
screen -dmS "$SCREEN_NAME" bash -lc "
source ~/.bashrc
conda activate $ENV_NAME || source activate $ENV_NAME
python $SCRIPT 2>&1 | tee -a $LOG_FILE
"