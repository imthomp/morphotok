#!/bin/bash
#SBATCH --job-name=morphotok-fertility
#SBATCH --output=/home/it238/projects/morphotok/experiments/logs/fertility_%j.out
#SBATCH --error=/home/it238/projects/morphotok/experiments/logs/fertility_%j.out
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
export HF_HUB_OFFLINE=1

VENV=/home/it238/projects/morphotok/.venv
SCRIPT=/home/it238/projects/morphotok/experiments/scripts/fertility_analysis.py
NLLB_TOK=/home/it238/.cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/snapshots/f8d333a098d19b4fd9a8b18f94170487ad3f821d/tokenizer.json
DATA=/home/it238/projects/morphotok/experiments/data/morphynet_cache

declare -A WORDLISTS=(
  [fin]="$DATA/wordforms_fin.txt"
  [hun]="$DATA/wordforms_hun.txt"
  [rus]="$DATA/wordforms_rus.txt"
  [ces]="$DATA/wordforms_ces.txt"
)

for lang in fin hun rus ces; do
  echo "=== $lang ==="
  "$VENV/bin/python" "$SCRIPT" \
    --nllb-tokenizer "$NLLB_TOK" \
    --lang "$lang" \
    --wordlist "${WORDLISTS[$lang]}" \
    --max-words 100000
done
