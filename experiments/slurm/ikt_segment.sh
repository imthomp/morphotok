#!/bin/bash
#SBATCH --job-name=morphotok-ikt-segment
#SBATCH --output=/home/it238/projects/morphotok/experiments/logs/ikt_segment_%j.out
#SBATCH --time=06:00:00
#SBATCH --partition=m9
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
set -euo pipefail
cd /home/it238/projects/mapudungun/mapudungun-mt
ROOT=/home/it238/projects/morphotok/experiments/data_pipeline/inuktitut/pilot_data
.venv/bin/python scripts/tokenization/segment_morfessor.py --data-root $ROOT --model-out $ROOT/morfessor_model.bin --approaches blocks
.venv/bin/python scripts/tokenization/segment_morfessor_vc.py --data-root $ROOT --approaches blocks
