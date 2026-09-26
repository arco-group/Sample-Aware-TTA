#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$REPO_ROOT/model"
: "${DATA_ROOT:?Set DATA_ROOT to your dataset root}"
: "${TASK_CHECKPOINT_PATH:?Set TASK_CHECKPOINT_PATH to the pretrained task checkpoint}"
: "${AE_CHECKPOINT_DIR:?Set AE_CHECKPOINT_DIR to the AE checkpoint directory}"

MODEL_NAME="${MODEL_NAME:-sample_aware_tta}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/$MODEL_NAME}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$OUTPUT_ROOT/checkpoints}"
RESULTS_DIR="${RESULTS_DIR:-$OUTPUT_ROOT/results}"
MODEL_TYPE="${MODEL_TYPE:-cycle_gan}"
DATASET_MODE="${DATASET_MODE:-paired_slice}"
MODALITIES_STRING="${MODALITIES:-source target}"
INPUT_NC="${INPUT_NC:-1}"
OUTPUT_NC="${OUTPUT_NC:-1}"
NETG="${NETG:-resnet_9blocks}"
NORM="${NORM:-instance}"
GPU_IDS="${GPU_IDS:-0}"
AE_EPOCH="${AE_EPOCH:-49}"

read -r -a STRATEGIES <<< "${TTA_STRATEGIES:-rndm_10 rndm_50 grid forward backward}"
read -r -a THRESHOLDS <<< "${TTA_THRESHOLDS:-0.0064}"
read -r -a MODALITIES <<< "$MODALITIES_STRING"
cd "$MODEL_DIR"

for strategy in "${STRATEGIES[@]}"; do
  for threshold in "${THRESHOLDS[@]}"; do
    echo "Running TTA: strategy=$strategy, threshold=$threshold"
    TTA_STRATEGY="$strategy" TTA_THRESHOLD="$threshold" python "$MODEL_DIR/run_tta.py" \
      --norm "$NORM" --dataset_mode "$DATASET_MODE" --modalities "${MODALITIES[@]}" \
      --dataroot "$DATA_ROOT" --input_nc "$INPUT_NC" --output_nc "$OUTPUT_NC" \
      --phase tta --model "$MODEL_TYPE" --netG "$NETG" --batch_size 1 --serial_batches \
      --display_freq 8000 --update_html_freq 1000 --name "$MODEL_NAME" \
      --checkpoints_dir "$CHECKPOINTS_DIR" --gpu_ids "$GPU_IDS" --results_dir "$RESULTS_DIR" \
      --task_checkpoint_path "$TASK_CHECKPOINT_PATH" --ae_checkpoint_dir "$AE_CHECKPOINT_DIR" \
      --ae_epoch "$AE_EPOCH" --tta_strategy "$strategy" --tta_threshold "$threshold" \
      --alr "${ALR:-0.0001}"
  done
done
