#!/bin/bash
#SBATCH --job-name=morphotok-ikt_syl_en_ikt_syl_bpe
#SBATCH --output=/home/it238/projects/morphotok/experiments/logs/ikt_syl_en_ikt_syl_bpe_%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --exclude=cs-1-2
set -euo pipefail
export HF_HUB_OFFLINE=1
/home/it238/projects/mapudungun/mapudungun-mt/.venv/bin/python /home/it238/projects/morphotok/experiments/scripts/finetune_nllb_pilot.py --src en --tgt ikt \
  --data-dir /home/it238/projects/morphotok/experiments/data_pipeline/inuktitut/pilot_data_syl \
  --output-dir /home/it238/projects/morphotok/experiments/models/inuktitut_syl/nllb-600M-en-ikt-syl_bpe_e25_len256 \
  --tokenizer-approach standard --epochs 25 --max-length 256 --added-tokens-file /home/it238/projects/morphotok/experiments/data_pipeline/inuktitut/pilot_data_syl/inventories/syl_bpe.json
