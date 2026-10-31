#!/bin/bash
#SBATCH --job-name=morphotok-p0_arn_es_standard_bs128
#SBATCH --output=/home/it238/projects/morphotok/experiments/logs/p0_arn_es_standard_bs128_%j.out
#SBATCH --error=/home/it238/projects/morphotok/experiments/logs/p0_arn_es_standard_bs128_%j.out
#SBATCH --time=16:00:00
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --exclude=cs-1-2

set -euo pipefail
export HF_HUB_OFFLINE=1

/home/it238/projects/mapudungun/mapudungun-mt/.venv/bin/python /home/it238/projects/morphotok/experiments/scripts/finetune_nllb_pilot.py \
  --src arn --tgt es \
  --data-dir /home/it238/projects/morphotok/experiments/data_pipeline/mapudungun/pilot_data \
  --output-dir /home/it238/projects/morphotok/experiments/models/phase0_pilot/nllb-600M-arn-es-standard_bs128 \
  --tokenizer-approach standard \
  --batch-size 32 --grad-accum 4
