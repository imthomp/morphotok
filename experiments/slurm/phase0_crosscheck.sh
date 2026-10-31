#!/bin/bash
#SBATCH --job-name=morphotok-p0-crosscheck
#SBATCH --output=/home/it238/projects/morphotok/experiments/logs/p0_crosscheck_%j.out
#SBATCH --error=/home/it238/projects/morphotok/experiments/logs/p0_crosscheck_%j.out
#SBATCH --time=03:00:00
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --exclude=cs-1-2
set -euo pipefail
export HF_HUB_OFFLINE=1
/home/it238/projects/mapudungun/mapudungun-mt/.venv/bin/python /home/it238/projects/morphotok/experiments/scripts/phase0_crosscheck_paper.py
