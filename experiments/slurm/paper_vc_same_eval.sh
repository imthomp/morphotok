#!/bin/bash
#SBATCH --job-name=morphotok-paper-vc-eval
#SBATCH --output=/home/it238/projects/morphotok/experiments/logs/paper_vc_eval_%j.out
#SBATCH --error=/home/it238/projects/morphotok/experiments/logs/paper_vc_eval_%j.out
#SBATCH --time=06:00:00
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --exclude=cs-1-2
set -euo pipefail
export HF_HUB_OFFLINE=1
/home/it238/projects/mapudungun/mapudungun-mt/.venv/bin/python /home/it238/projects/morphotok/experiments/scripts/paper_vc_same_eval.py
