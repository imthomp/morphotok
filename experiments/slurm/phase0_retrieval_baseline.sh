#!/bin/bash
#SBATCH --job-name=morphotok-p0-retrieval
#SBATCH --output=/home/it238/projects/morphotok/experiments/logs/p0_retrieval_%j.out
#SBATCH --error=/home/it238/projects/morphotok/experiments/logs/p0_retrieval_%j.out
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
set -euo pipefail
cd /home/it238/projects/morphotok
.venv/bin/python experiments/scripts/phase0_retrieval_baseline.py
