#!/bin/bash
#SBATCH --job-name=morphotok-p0-diagnose-ve
#SBATCH --output=/home/it238/projects/morphotok/experiments/logs/p0_diagnose_ve_%j.out
#SBATCH --error=/home/it238/projects/morphotok/experiments/logs/p0_diagnose_ve_%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

set -euo pipefail
export HF_HUB_OFFLINE=1

VENV=/home/it238/projects/mapudungun/mapudungun-mt/.venv
SCRIPT=/home/it238/projects/morphotok/experiments/scripts/phase0_diagnose_vocab_expansion.py
DATA_DIR=/home/it238/projects/morphotok/experiments/data_pipeline/mapudungun/pilot_data
MODEL_DIR=/home/it238/projects/morphotok/experiments/models/phase0_pilot/nllb-600M-es-arn-vocab_expansion/best
NEW_TOKENS=/home/it238/projects/morphotok/experiments/data_pipeline/mapudungun/pilot_data/blocks/vocab_expansion/new_tokens.json

"$VENV/bin/python" "$SCRIPT" \
  --data-dir "$DATA_DIR" \
  --model-dir "$MODEL_DIR" \
  --new-tokens-json "$NEW_TOKENS" \
  --batch-size 32
