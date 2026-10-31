#!/bin/bash
#SBATCH --job-name=morphotok-p0_arn_es_standard_len256
#SBATCH --output=/home/it238/projects/morphotok/experiments/logs/p0_arn_es_standard_e25_len256_%j.out
#SBATCH --error=/home/it238/projects/morphotok/experiments/logs/p0_arn_es_standard_e25_len256_%j.out
#SBATCH --time=20:00:00
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --exclude=cs-1-2
set -euo pipefail
export HF_HUB_OFFLINE=1
/home/it238/projects/mapudungun/mapudungun-mt/.venv/bin/python /home/it238/projects/morphotok/experiments/scripts/finetune_nllb_pilot.py \
  --src arn --tgt es \
  --data-dir /home/it238/projects/morphotok/experiments/data_pipeline/mapudungun/pilot_data \
  --output-dir /home/it238/projects/morphotok/experiments/models/phase0_pilot/nllb-600M-arn-es-standard_e25_len256 \
  --tokenizer-approach standard --epochs 25 --max-length 256
