#!/bin/bash
#SBATCH --job-name=morphotok-degen-ikt
#SBATCH --output=/home/it238/projects/morphotok/experiments/logs/degen_ikt_%j.out
#SBATCH --error=/home/it238/projects/morphotok/experiments/logs/degen_ikt_%j.out
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail
cd /home/it238/projects/morphotok
.venv/bin/python experiments/scripts/degenerate_check_single.py \
  --lang ikt \
  --corpus experiments/data_pipeline/inuktitut/cleaned/train.ikt \
  --max-lines 200000
