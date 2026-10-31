#!/bin/bash
#SBATCH --job-name=morphotok-ikt_ikt_en_morfessor_vc
#SBATCH --output=/home/it238/projects/morphotok/experiments/logs/ikt_ikt_en_morfessor_vc_%j.out
#SBATCH --error=/home/it238/projects/morphotok/experiments/logs/ikt_ikt_en_morfessor_vc_%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --exclude=cs-1-2
set -euo pipefail
export HF_HUB_OFFLINE=1
/home/it238/projects/mapudungun/mapudungun-mt/.venv/bin/python /home/it238/projects/morphotok/experiments/scripts/finetune_nllb_pilot.py \
  --src ikt --tgt en \
  --data-dir /home/it238/projects/morphotok/experiments/data_pipeline/inuktitut/pilot_data \
  --output-dir /home/it238/projects/morphotok/experiments/models/inuktitut_pilot/nllb-600M-ikt-en-morfessor_vc_e25_len256 \
  --tokenizer-approach morfessor_vc --epochs 25 --max-length 256
