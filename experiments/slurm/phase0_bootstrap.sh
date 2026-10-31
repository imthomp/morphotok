#!/bin/bash
#SBATCH --job-name=morphotok-p0-bootstrap
#SBATCH --output=/home/it238/projects/morphotok/experiments/logs/p0_bootstrap_%j.out
#SBATCH --error=/home/it238/projects/morphotok/experiments/logs/p0_bootstrap_%j.out
#SBATCH --time=03:00:00
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

set -euo pipefail
export HF_HUB_OFFLINE=1

VENV=/home/it238/projects/mapudungun/mapudungun-mt/.venv
SCRIPT=/home/it238/projects/morphotok/experiments/scripts/phase0_predict_and_bootstrap.py
DATA_DIR=/home/it238/projects/morphotok/experiments/data_pipeline/mapudungun/pilot_data
MODELS_DIR=/home/it238/projects/morphotok/experiments/models/phase0_pilot

"$VENV/bin/python" "$SCRIPT" \
  --data-dir "$DATA_DIR" \
  --models-dir "$MODELS_DIR" \
  --batch-size 32
