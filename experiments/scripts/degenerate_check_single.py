#!/usr/bin/env python3
"""Degenerate-segmentation spot-check for one language: train Morfessor on a
real corpus's word types, sample 40 words, count near-character-level splits
(avg piece length <= 1.5 chars, word length > 3). Merges result into
results/degenerate_segmentation_check.json. See notes/plan.md, Known Design
Issues item 7.
"""
import argparse, json, random
from pathlib import Path
import morfessor

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_JSON = SCRIPT_DIR / ".." / "results" / "degenerate_segmentation_check.json"

ap = argparse.ArgumentParser()
ap.add_argument("--lang", required=True)
ap.add_argument("--corpus", required=True, help="one sentence per line")
ap.add_argument("--max-lines", type=int, default=200000)
ap.add_argument("--seed", type=int, default=5)
args = ap.parse_args()

random.seed(args.seed)
words, n_lines = set(), 0
for line in open(args.corpus, encoding="utf-8"):
    words.update(line.split())
    n_lines += 1
    if n_lines >= args.max_lines:
        break
words = list(words)
print(f"training on {len(words)} unique words from {n_lines} lines", flush=True)

model = morfessor.BaselineModel()
model.load_data([(1, w) for w in words])
model.train_batch()

n = 40
sample = random.sample(words, n)
near_char, examples = 0, []
for w in sample:
    segs = model.viterbi_segment(w)[0]
    if sum(len(s) for s in segs) / len(segs) <= 1.5 and len(w) > 3:
        near_char += 1
    examples.append((w, segs))
print(f"{args.lang}: {near_char}/{n} near-character-level (n_words={len(words)})", flush=True)
for w, segs in examples[:8]:
    print("   ", w, "->", segs, flush=True)

existing = json.loads(OUT_JSON.read_text(encoding="utf-8")) if OUT_JSON.exists() else {}
existing[args.lang] = {"near_char_level": near_char, "n_sampled": n, "n_words": len(words)}
OUT_JSON.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("done")
