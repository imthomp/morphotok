#!/bin/bash
#SBATCH --job-name=morphotok-p0_seed43_morfessor_vc
#SBATCH --output=/home/it238/projects/morphotok/experiments/logs/p0_seed43_morfessor_vc_%j.out
#SBATCH --error=/home/it238/projects/morphotok/experiments/logs/p0_seed43_morfessor_vc_%j.out
#SBATCH --time=16:00:00
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

set -euo pipefail
export HF_HUB_OFFLINE=1

VENV=/home/it238/projects/mapudungun/mapudungun-mt/.venv
DATA_DIR=/home/it238/projects/morphotok/experiments/data_pipeline/mapudungun/pilot_data

"$VENV/bin/python" /home/it238/projects/morphotok/experiments/scripts/finetune_nllb_pilot.py \
  --src es --tgt arn \
  --data-dir "$DATA_DIR" \
  --output-dir /home/it238/projects/morphotok/experiments/models/phase0_pilot/nllb-600M-es-arn-morfessor_vc_seed43 \
  --tokenizer-approach morfessor_vc --seed 43
