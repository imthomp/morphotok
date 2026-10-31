#!/usr/bin/env python3
"""Train Vocab-biased BPE for one language and save a few sanity-check
segmentations + timing to results/vocab_biased_bpe_smoke.json.

This is a smoke test / sanity check of the trainer at real corpus scale
(experiments/scripts/fertility_analysis.py and the density scripts are the
actual research-facing outputs), not itself a paper-facing number.
"""

import argparse
import json
import time
from pathlib import Path

from morphotok.methods import VocabBiasedBPE

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_JSON = SCRIPT_DIR / ".." / "results" / "vocab_biased_bpe_smoke.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nllb-tokenizer", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--wordlist", required=True)
    ap.add_argument("--max-words", type=int, default=100000)
    ap.add_argument("--num-merges", type=int, default=8000)
    ap.add_argument("--lam", type=float, default=1.0)
    args = ap.parse_args()

    vocab = set(json.loads(Path(args.nllb_tokenizer).read_text(encoding="utf-8"))["model"]["vocab"].keys())
    words = [w for w in Path(args.wordlist).read_text(encoding="utf-8").splitlines() if w]
    if args.max_words and len(words) > args.max_words:
        step = len(words) // args.max_words
        words = words[::step][: args.max_words]

    t0 = time.time()
    model = VocabBiasedBPE(vocab, lam=args.lam)
    model.fit(words, num_merges=args.num_merges)
    elapsed = time.time() - t0

    import random
    random.seed(1)
    sample_idx = random.sample(range(len(words)), min(10, len(words)))
    examples = [{"word": words[i], "pieces": model.segment(words[i])} for i in sample_idx]
    n_split = sum(1 for w in words if len(model.segment(w)) > 1)

    out = {
        args.lang: {
            "generated_by": "scripts/train_vocab_biased_bpe.py",
            "n_words": len(words),
            "num_merges": args.num_merges,
            "lam": args.lam,
            "elapsed_seconds": round(elapsed, 2),
            "n_words_split_gt1_piece": n_split,
            "examples": examples,
        }
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if OUT_JSON.exists():
        try:
            existing = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    existing.update(out)
    OUT_JSON.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{args.lang}: {len(words)} words, {args.num_merges} merges, {elapsed:.1f}s, {n_split} words split >1 piece")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
