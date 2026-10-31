#!/bin/bash
#SBATCH --job-name=morphotok-vbbpe
#SBATCH --output=/home/it238/projects/morphotok/experiments/logs/vbbpe_%j.out
#SBATCH --error=/home/it238/projects/morphotok/experiments/logs/vbbpe_%j.out
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

VENV=/home/it238/projects/morphotok/.venv
SCRIPT=/home/it238/projects/morphotok/experiments/scripts/train_vocab_biased_bpe.py
NLLB_TOK=/home/it238/.cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/snapshots/f8d333a098d19b4fd9a8b18f94170487ad3f821d/tokenizer.json
DATA=/home/it238/projects/morphotok/experiments/data/morphynet_cache

cd /home/it238/projects/morphotok

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
    --max-words 100000 \
    --num-merges 8000
done
