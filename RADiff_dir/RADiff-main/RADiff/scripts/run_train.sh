#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

echo "Running train.py in background..."
log_file="./logs/train_$(date +%Y%m%d_%H%M%S).log"
nohup python train.py > $log_file 2>&1 &

echo "Log: $log_file"
echo "PID: $!"
