#!/usr/bin/env python3
"""
Unsupervised vocabulary density: train Morfessor on a plain word list (no
gold segmentation used), segment the vocabulary, and compute what fraction
of Morfessor's own predicted morphemes are already in NLLB's tokenizer
vocab. This is the estimator that has to scale to the full 116-language
NLLB population where gold segmentation doesn't exist -- this script
computes it; gold_vocabulary_density.py computes the ground truth it's
checked against, for the subset of languages where that exists.

Input word list: one word-form per line, real text (e.g. the unique
inflected forms from a MorphyNet file, or any monolingual corpus). Morfessor
sees only the words, never the gold segmentation, even when both exist for
the same language.
"""

import argparse
import json
from pathlib import Path

import morfessor

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_JSON = SCRIPT_DIR / ".." / "results" / "unsupervised_vocabulary_density.json"


def load_nllb_vocab(tokenizer_json_path):
    d = json.loads(Path(tokenizer_json_path).read_text(encoding="utf-8"))
    return set(d["model"]["vocab"].keys())


def train_morfessor(words):
    io = morfessor.MorfessorIO()
    model = morfessor.BaselineModel()
    # unweighted: each unique word form counts once (we already deduped)
    data = [(1, w) for w in words]
    model.load_data(data)
    model.train_batch()
    return model


def compute_density(model, words, vocab):
    from collections import Counter
    morphemes = Counter()
    for w in words:
        segs, _ = model.viterbi_segment(w)
        for m in segs:
            morphemes[m] += 1

    types_total = len(morphemes)
    types_in_vocab = sum(1 for m in morphemes if m in vocab or f"▁{m}" in vocab)
    tokens_total = sum(morphemes.values())
    tokens_in_vocab = sum(c for m, c in morphemes.items() if m in vocab or f"▁{m}" in vocab)
    return {
        "morpheme_types_total": types_total,
        "morpheme_types_in_nllb_vocab": types_in_vocab,
        "density_by_type": round(types_in_vocab / types_total, 4) if types_total else None,
        "morpheme_tokens_total": tokens_total,
        "morpheme_tokens_in_nllb_vocab": tokens_in_vocab,
        "density_by_token": round(tokens_in_vocab / tokens_total, 4) if tokens_total else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nllb-tokenizer", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--wordlist", required=True, help="one word-form per line, real text")
    ap.add_argument("--max-words", type=int, default=200000, help="cap for Morfessor training speed")
    args = ap.parse_args()

    vocab = load_nllb_vocab(args.nllb_tokenizer)
    words = Path(args.wordlist).read_text(encoding="utf-8").splitlines()
    words = [w for w in words if w]
    if args.max_words and len(words) > args.max_words:
        # deterministic subsample, not random, for reproducibility
        step = len(words) // args.max_words
        words = words[::step][: args.max_words]

    model = train_morfessor(words)
    density = compute_density(model, words, vocab)

    out = {
        "unsupervised_vocabulary_density": {
            args.lang: {
                "generated_by": "scripts/unsupervised_vocabulary_density.py",
                "wordlist_source": str(args.wordlist),
                "n_words_used": len(words),
                "nllb_vocab_size": len(vocab),
                **density,
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
    existing.setdefault("unsupervised_vocabulary_density", {}).update(out["unsupervised_vocabulary_density"])
    OUT_JSON.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{args.lang}: {len(words)} words, {density['morpheme_types_total']} unique Morfessor morphemes")
    print(f"  density_by_type={density['density_by_type']}  density_by_token={density['density_by_token']}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
