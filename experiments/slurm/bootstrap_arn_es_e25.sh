#!/bin/bash
#SBATCH --job-name=morphotok-boot-arn-es-e25
#SBATCH --output=/home/it238/projects/morphotok/experiments/logs/boot_arn_es_e25_%j.out
#SBATCH --error=/home/it238/projects/morphotok/experiments/logs/boot_arn_es_e25_%j.out
#SBATCH --time=03:00:00
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --exclude=cs-1-2
set -euo pipefail
export HF_HUB_OFFLINE=1
/home/it238/projects/mapudungun/mapudungun-mt/.venv/bin/python /home/it238/projects/morphotok/experiments/scripts/phase0_bootstrap_general.py \
  --src arn --tgt es --suffix _e25 \
  --out-json /home/it238/projects/morphotok/experiments/results/phase0_bootstrap_arn_es_e25.json
