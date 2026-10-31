#!/bin/bash
#SBATCH --job-name=morphotok-probe-bos
#SBATCH --output=/home/it238/projects/morphotok/experiments/logs/probe_bos_%j.out
#SBATCH --error=/home/it238/projects/morphotok/experiments/logs/probe_bos_%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --exclude=cs-1-2
set -euo pipefail
export HF_HUB_OFFLINE=1
/home/it238/projects/mapudungun/mapudungun-mt/.venv/bin/python /home/it238/projects/morphotok/experiments/scripts/probe_forced_bos.py --n 300
