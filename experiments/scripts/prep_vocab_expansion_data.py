#!/usr/bin/env python3
"""Prepare Morfessor-VE (Vocab Expansion) pre-segmented Mapudungun data for
the Phase 0 pilot, matching the @@ boundary-marker convention used by the
original Mapudungun paper's segment_morfessor.py / finetune_nllb.py so the
same desegment()/predict_test.py machinery works unchanged.

Unlike Morfessor-VC, every OOV morpheme found is a genuine new token to add
to the tokenizer -- this script also writes the new-token list (with
average-embedding init source pieces) that the training script consumes to
actually expand the model's vocabulary before fine-tuning.

Output layout matches finetune_nllb.py's expected `tokenizer-approach` dirs:
  {data_root}/blocks/vocab_expansion/{split}/{src.txt,tgt.txt}
  {data_root}/blocks/vocab_expansion/new_tokens.json
"""

import argparse
import json
from pathlib import Path

from morphotok.methods import MorfessorVE


def render_word(pieces: list[str]) -> str:
    parts = []
    for i, p in enumerate(pieces):
        parts.append(p + "@@" if i < len(pieces) - 1 else p)
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--nllb-tokenizer", required=True)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    cleaned_base = data_root / "blocks" / "cleaned"
    out_base = data_root / "blocks" / "vocab_expansion"

    vocab = set(json.loads(Path(args.nllb_tokenizer).read_text(encoding="utf-8"))["model"]["vocab"].keys())

    train_words = sorted({
        w for line in (cleaned_base / "train" / "cleaned" / "src.txt").read_text(encoding="utf-8").splitlines()
        for w in line.split()
    })

    ve = MorfessorVE(vocab)
    ve.fit(train_words)

    # Derive new vocab tokens as the LITERAL @@-marked surface strings that
    # will appear in the pre-segmented text (not MorfessorVE.new_tokens()'s
    # bare-morpheme + separate "▁"-marker convention, which is designed for
    # the general-purpose toolkit API, not this project's @@-marker text
    # format) -- the tokenizer needs to match the exact string it will see.
    new_token_set: set[str] = set()
    for w in train_words:
        pieces = ve.segment(w)
        surface = [p + "@@" if i < len(pieces) - 1 else p for i, p in enumerate(pieces)]
        for i, tok in enumerate(surface):
            bare = tok[:-2] if tok.endswith("@@") else tok
            marked = f"▁{bare}" if i == 0 else bare
            if marked not in vocab and bare not in vocab:
                new_token_set.add(tok)
    new_tokens = sorted(new_token_set)
    print(f"Morfessor-VE: {len(train_words)} unique train words, {len(new_tokens)} new OOV @@-marked surface tokens")

    (out_base).mkdir(parents=True, exist_ok=True)
    (out_base / "new_tokens.json").write_text(json.dumps(new_tokens, ensure_ascii=False, indent=2), encoding="utf-8")

    for split in ("train", "dev", "test"):
        src_lines = (cleaned_base / split / "cleaned" / "src.txt").read_text(encoding="utf-8").splitlines()
        out_dir = out_base / split
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "src.txt", "w", encoding="utf-8") as f:
            for line in src_lines:
                words = [render_word(ve.segment(w)) for w in line.split()]
                f.write(" ".join(words) + "\n")
        # tgt (Spanish) unchanged
        (out_dir / "tgt.txt").write_text(
            (cleaned_base / split / "cleaned" / "tgt.txt").read_text(encoding="utf-8"), encoding="utf-8"
        )
        print(f"  {split}: wrote {out_dir}")


if __name__ == "__main__":
    main()
