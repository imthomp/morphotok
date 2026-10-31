#!/bin/bash
#SBATCH --job-name=morphotok-ikt-rom-density
#SBATCH --output=/home/it238/projects/morphotok/experiments/logs/ikt_rom_density_%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=m9
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
set -euo pipefail
cd /home/it238/projects/morphotok
TOK=/home/it238/.cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/snapshots/f8d333a098d19b4fd9a8b18f94170487ad3f821d/tokenizer.json
.venv/bin/python experiments/scripts/unsupervised_vocabulary_density.py --nllb-tokenizer $TOK --lang ikt_rom --wordlist experiments/data/wordlists/wordforms_ikt_rom.txt --max-words 100000 --out-json experiments/results/per_lang/density_ikt_rom.json
.venv/bin/python experiments/scripts/fertility_analysis.py --nllb-tokenizer $TOK --lang ikt_rom --wordlist experiments/data/wordlists/wordforms_ikt_rom.txt --max-words 100000 --out-json experiments/results/per_lang/fertility_ikt_rom.json
