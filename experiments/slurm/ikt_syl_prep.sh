#!/bin/bash
#SBATCH --job-name=morphotok-ikt-syl-prep
#SBATCH --output=/home/it238/projects/morphotok/experiments/logs/ikt_syl_prep_%j.out
#SBATCH --time=08:00:00
#SBATCH --partition=m9
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
set -euo pipefail
export HF_HUB_OFFLINE=1
/home/it238/projects/mapudungun/mapudungun-mt/.venv/bin/python /home/it238/projects/morphotok/experiments/scripts/prep_inuktitut_syllabics.py --corpus /home/it238/projects/morphotok/experiments/data_pipeline/inuktitut/cleaned \
  --rows /home/it238/projects/morphotok/experiments/data_pipeline/inuktitut/pilot_train_rows.txt --out /home/it238/projects/morphotok/experiments/data_pipeline/inuktitut/pilot_data_syl --k 8000
