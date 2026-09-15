#!/usr/bin/env python3
"""
Fertility: average number of NLLB subword tokens per whitespace-delimited
word, for a given tokenization condition. Extends the fertility analysis
already implemented for Mapudungun (Thompson et al., 2026) to other
languages as their word lists become available.

Two conditions computed here:
  - "standard_bpe": NLLB's own SentencePiece tokenizer applied directly to
    each word (no pre-segmentation).
  - "morfessor": Morfessor's own segmentation (pieces per word), independent
    of whether those pieces are in NLLB's vocab -- this is pre-segmentation
    fertility, comparable to the Mapudungun paper's Figure ("Fertility
    Comparison") which excludes Standard BPE for the same reason (its
    segmentation is applied internally, not comparable pre-segmentation
    fertility).
"""

import argparse
import json
from pathlib import Path

import morfessor
from tokenizers import Tokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_JSON = SCRIPT_DIR / ".." / "results" / "fertility_analysis.json"


def train_morfessor(words):
    model = morfessor.BaselineModel()
    model.load_data([(1, w) for w in words])
    model.train_batch()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nllb-tokenizer", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--wordlist", required=True)
    ap.add_argument("--max-words", type=int, default=50000)
    args = ap.parse_args()

    words = [w for w in Path(args.wordlist).read_text(encoding="utf-8").splitlines() if w]
    if args.max_words and len(words) > args.max_words:
        step = len(words) // args.max_words
        words = words[::step][: args.max_words]

    tok = Tokenizer.from_file(args.nllb_tokenizer)
    bpe_fertilities = [len(tok.encode(w).tokens) for w in words]
    bpe_fertility = sum(bpe_fertilities) / len(bpe_fertilities)

    model = train_morfessor(words)
    morf_fertilities = [len(model.viterbi_segment(w)[0]) for w in words]
    morf_fertility = sum(morf_fertilities) / len(morf_fertilities)

    result = {
        "fertility_analysis": {
            args.lang: {
                "generated_by": "scripts/fertility_analysis.py",
                "n_words": len(words),
                "standard_bpe_fertility": round(bpe_fertility, 3),
                "morfessor_fertility": round(morf_fertility, 3),
                "note": "standard_bpe = NLLB's own tokenizer applied directly per word; morfessor = pre-segmentation piece count, independent of NLLB vocab membership.",
            }
        }
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if OUT_JSON.exists():
        try:
            existing = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    existing.setdefault("fertility_analysis", {}).update(result["fertility_analysis"])
    OUT_JSON.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{args.lang}: standard_bpe={bpe_fertility:.3f}  morfessor={morf_fertility:.3f}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
