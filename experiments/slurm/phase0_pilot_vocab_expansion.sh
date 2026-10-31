#!/bin/bash
#SBATCH --job-name=morphotok-p0-vocab_expansion
#SBATCH --output=/home/it238/projects/morphotok/experiments/logs/p0_vocab_expansion_%j.out
#SBATCH --error=/home/it238/projects/morphotok/experiments/logs/p0_vocab_expansion_%j.out
#SBATCH --time=16:00:00
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

set -euo pipefail
export HF_HUB_OFFLINE=1

VENV=/home/it238/projects/mapudungun/mapudungun-mt/.venv
SCRIPT=/home/it238/projects/morphotok/experiments/scripts/finetune_nllb_pilot.py
DATA_DIR=/home/it238/projects/morphotok/experiments/data_pipeline/mapudungun/pilot_data
OUT=/home/it238/projects/morphotok/experiments/models/phase0_pilot/nllb-600M-es-arn-vocab_expansion

"$VENV/bin/python" "$SCRIPT" \
  --src es --tgt arn \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUT" \
  --tokenizer-approach vocab_expansion \
  --gpu-hours-budget 15
