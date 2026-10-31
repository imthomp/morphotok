#!/bin/bash
#SBATCH --job-name=morphotok-boot_ikt_s43_en_ikt
#SBATCH --output=/home/it238/projects/morphotok/experiments/logs/boot_ikt_s43_en_ikt_%j.out
#SBATCH --time=05:00:00
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --exclude=cs-1-2
set -euo pipefail
export HF_HUB_OFFLINE=1
/home/it238/projects/mapudungun/mapudungun-mt/.venv/bin/python /home/it238/projects/morphotok/experiments/scripts/phase0_bootstrap_general.py \
  --src en --tgt ikt --suffix _e25_len256_s43 --conditions standard morfessor_vc --max-length 256 \
  --data-dir /home/it238/projects/morphotok/experiments/data_pipeline/inuktitut/pilot_data --models-dir /home/it238/projects/morphotok/experiments/models/inuktitut_pilot --out-json /home/it238/projects/morphotok/experiments/results/ikt_bootstrap_en_ikt_e25_len256_s43.json
