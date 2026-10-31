#!/usr/bin/env python3
"""Build the syllabics Inuktitut pilot and its three added-token inventories.

Question: NLLB-200 can't read Canadian Aboriginal Syllabics (42.6% <unk>). When
the script is genuinely missing, does adding *morpheme* tokens help beyond
simply adding the missing characters, and beyond adding an equal number of
frequency-derived (BPE) pieces?

Conditions (all on the same 56,197 training pairs as the romanized pilot):
  standard   unmodified NLLB tokenizer (most syllabics -> <unk>)
  syl_char   + every training character NLLB maps to <unk>
  syl_bpe    + syl_char + top-K syllabics BPE pieces (in merge order)
  syl_morph  + syl_char + top-K syllabics Morfessor morphs (by token frequency)

syl_bpe and syl_morph add the same number of pieces and are encoded by the same
mechanism (HF added tokens, leftmost-longest match), so they differ only in
where the inventory came from. Caveat: leftmost-longest matching over the morph
inventory need not reproduce Morfessor's own segmentation; agreement is
reported in inventory_stats.json.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import morfessor
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import AutoTokenizer

SYL = re.compile(r"^[᐀-ᙿ]+$")


def write_split(out, split, ikt, en):
    d = out / "blocks" / "cleaned" / split / "cleaned"
    d.mkdir(parents=True, exist_ok=True)
    (d / "src.txt").write_text("\n".join(ikt) + "\n", encoding="utf-8")
    (d / "tgt.txt").write_text("\n".join(en) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="inuktitut/cleaned (syllabics)")
    ap.add_argument("--rows", required=True, help="train row indices of the romanized pilot, in pilot order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8000)
    ap.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    args = ap.parse_args()
    corpus, out = Path(args.corpus), Path(args.out)

    read = lambda f: (corpus / f).read_text(encoding="utf-8").splitlines()
    rows = [int(x) for x in Path(args.rows).read_text().split()]
    tr_ikt, tr_en = read("train.ikt"), read("train.en")
    train_ikt = [tr_ikt[i] for i in rows]
    write_split(out, "train", train_ikt, [tr_en[i] for i in rows])
    for s in ("dev", "test"):
        write_split(out, s, read(f"{s}.ikt"), read(f"{s}.en"))
    print(f"train {len(rows):,}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    vocab = tok.get_vocab()
    unk = tok.unk_token_id
    chars = sorted({c for l in train_ikt for c in l if not c.isspace()})
    unk_chars = [c for c in chars if tok(c, add_special_tokens=False)["input_ids"] in ([unk], [tok.convert_tokens_to_ids("▁"), unk])]
    print(f"{len(chars)} distinct chars, {len(unk_chars)} map to <unk>", flush=True)

    words = Counter(w for l in train_ikt for w in l.split())
    syl_words = {w: c for w, c in words.items() if SYL.match(w)}

    # Morfessor Baseline, trained as in the AmericasNLP paper's segment_morfessor.py
    # (token counts, identity count modifier, batch training).
    m = morfessor.BaselineModel()
    m.load_data([(c, w) for w, c in syl_words.items()], count_modifier=lambda x: x)
    m.train_batch()
    morph_freq, morfessor_seg = Counter(), {}
    for w, c in syl_words.items():
        seg = m.viterbi_segment(w)[0]
        morfessor_seg[w] = seg
        for p in seg:
            morph_freq[p] += c
    keep = lambda p: len(p) >= 2 and p not in vocab and ("▁" + p) not in vocab
    morphs = [p for p, _ in morph_freq.most_common() if keep(p)]

    bpe = Tokenizer(models.BPE(unk_token="[UNK]"))
    bpe.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    bpe.train_from_iterator((w for w, c in syl_words.items() for _ in range(c)),
                            trainers.BpeTrainer(vocab_size=4 * args.k, show_progress=False, special_tokens=["[UNK]"]))
    merges = json.loads(bpe.to_str())["model"]["merges"]
    seen, bpe_pieces = set(), []
    for mg in merges:
        a, b = mg if isinstance(mg, list) else mg.split(" ")
        p = a + b
        if p not in seen and keep(p):
            seen.add(p)
            bpe_pieces.append(p)

    k = min(args.k, len(morphs), len(bpe_pieces))
    inv = {"syl_char": unk_chars, "syl_bpe": unk_chars + bpe_pieces[:k], "syl_morph": unk_chars + morphs[:k]}
    inv_dir = out / "inventories"
    inv_dir.mkdir(exist_ok=True)
    for name, toks in inv.items():
        (inv_dir / f"{name}.json").write_text(json.dumps(toks, ensure_ascii=False), encoding="utf-8")

    # Fertility, truncation, and (for syl_morph) agreement with Morfessor's own segmentation.
    stats = {"k": k, "n_unk_chars": len(unk_chars), "n_morph_types": len(morph_freq), "n_bpe_candidates": len(bpe_pieces),
             "n_train_word_types_syllabic": len(syl_words)}
    from tokenizers import pre_tokenizers as pt
    sample = train_ikt[::10]
    for name in ["standard"] + list(inv):
        t = AutoTokenizer.from_pretrained(args.model)
        if name != "standard":
            t._tokenizer.pre_tokenizer = pt.Metaspace(replacement="▁", prepend_scheme="first", split=True)
            t.add_tokens(inv[name])
        lens = [len(t(l)["input_ids"]) for l in sample]
        n_unk = sum(t(l)["input_ids"].count(unk) for l in sample)
        nw = sum(len(l.split()) for l in sample)
        s = {"tokens_per_word": round(sum(lens) / nw, 3), "frac_over_256": round(sum(x > 256 for x in lens) / len(lens), 4),
             "unk_per_word": round(n_unk / nw, 4)}
        if name == "syl_morph":
            agree = tot = 0
            for w, c in list(syl_words.items())[:20000]:
                pieces = [p for p in t.tokenize(w) if p != "▁"]
                agree += c * (pieces == list(morfessor_seg[w]))
                tot += c
            s["token_weighted_agreement_with_morfessor"] = round(agree / tot, 4)
        stats[name] = s
        print(name, s, flush=True)
    (out / "inventory_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
