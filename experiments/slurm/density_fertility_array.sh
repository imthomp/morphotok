#!/bin/bash
#SBATCH --job-name=morphotok-dens-fert
#SBATCH --output=/home/it238/projects/morphotok/experiments/logs/dens_fert_%A_%a.out
#SBATCH --error=/home/it238/projects/morphotok/experiments/logs/dens_fert_%A_%a.out
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --array=0-9

set -euo pipefail
export HF_HUB_OFFLINE=1
cd /home/it238/projects/morphotok

LANGS=(tur swh isl vie yor cmn arn nah quy ikt)
LANG=${LANGS[$SLURM_ARRAY_TASK_ID]}
NLLB_TOK=/home/it238/.cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/snapshots/f8d333a098d19b4fd9a8b18f94170487ad3f821d/tokenizer.json

case $LANG in
  tur|swh|isl|vie|yor) WL=experiments/data/flores_plus_cache/wordforms_${LANG}.txt ;;
  *) WL=experiments/data/wordlists/wordforms_${LANG}.txt ;;
esac

mkdir -p experiments/results/per_lang
echo "=== $LANG: $WL ==="
.venv/bin/python experiments/scripts/unsupervised_vocabulary_density.py \
  --nllb-tokenizer "$NLLB_TOK" --lang "$LANG" --wordlist "$WL" --max-words 100000 \
  --out-json experiments/results/per_lang/density_${LANG}.json
.venv/bin/python experiments/scripts/fertility_analysis.py \
  --nllb-tokenizer "$NLLB_TOK" --lang "$LANG" --wordlist "$WL" --max-words 100000 \
  --out-json experiments/results/per_lang/fertility_${LANG}.json
