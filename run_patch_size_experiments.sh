#!/bin/bash
# Run patch size experiments sequentially
# 32x32 is already running (woven-pond-29)
# This script will run 64x64 and 256x256 after 32x32 completes

set -e

echo "Waiting for 32x32 training to complete..."
# Wait for 32x32 training to finish by checking if the process is still running
while pgrep -f "experiment_groundtruth_finegrained_p32" > /dev/null; do
    sleep 60
    echo "$(date): 32x32 training still running..."
done
echo "32x32 training complete!"

# Run 64x64 training
echo "Starting 64x64 patch training..."
uv run dapidl train \
  -d ./experiment_groundtruth_finegrained_p64/dataset \
  --epochs 200 \
  --batch-size 64 \
  --backbone resnet50 \
  --backend dali-lmdb \
  -o ./experiment_groundtruth_finegrained_p64/training \
  --wandb

echo "64x64 training complete!"

# Run 256x256 training (need smaller batch size for larger patches)
echo "Starting 256x256 patch training..."
uv run dapidl train \
  -d ./experiment_groundtruth_finegrained_p256/dataset \
  --epochs 200 \
  --batch-size 32 \
  --backbone resnet50 \
  --backend dali-lmdb \
  -o ./experiment_groundtruth_finegrained_p256/training \
  --wandb

echo "256x256 training complete!"
echo "All patch size experiments finished!"
