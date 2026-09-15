#!/usr/bin/env python3
"""
Gold-derived vocabulary density: for a language with real morpheme-boundary
gold data, what fraction of gold morphemes are already present in NLLB's
pretrained subword vocabulary?

Gold source: MorphyNet (Batsuren et al.), inflectional.v1.tsv files.
Confirmed 2026-09: UniMorph itself does NOT give morpheme boundaries (just
lemma, inflected form, feature tags), so it's the wrong resource for this
despite being the one originally assumed in thesis_proposal.md's "Scalable
density estimation" paragraph -- MorphyNet is the correct source, and only
covers a subset of our target languages.

MorphyNet's own schema is NOT consistent across languages, confirmed 2026-09
by inspection, not assumed:
  - Russian (and apparently most languages): 4 columns (lemma, form,
    features, segmentation), where segmentation is a single pipe-delimited
    string like "приравнять|л|вший" giving the full stem+affix breakdown.
  - Finnish: a different 5-column schema with a header row (base_word,
    trg_word, features, suffix, src_word), where "suffix" is a single
    suffix string (not pipe-delimited, not the full segmentation) and
    "src_word" is the stem separately. Rows with no overt suffix use "@"
    or "-" placeholders and vary between 4 and 5 fields.
An earlier version of this script assumed the Russian schema for both
languages, which silently "worked" on Finnish (no crash) but produced a
nonsense result (1 unique affix type across 3.5M words) by treating each
single suffix string as if it were the whole stem+affix segmentation.
Caught by sanity-checking the output rather than trusting it, not caught
automatically -- this class of bug (wrong-but-plausible-looking numbers from
inconsistent data schemas) doesn't throw an exception, so re-check any new
language's output against a few raw sample rows before trusting it.

Coverage as of 2026-09: MorphyNet has real segmentation data for Finnish and
Russian (both target languages). NOT available for Turkish, Icelandic, or
Quechua -- no direct substitute found yet; those need either a different
segmentation resource or the unsupervised (Morfessor-based) density estimate
only, which is exactly the fallback thesis_proposal.md already designed for
scaling beyond languages with gold data.

NLLB vocab membership is checked both with and without the SentencePiece
word-initial "▁" marker, since a morpheme's presence in-vocab can depend on
whether it appears word-initially or word-internally in the model's training
data.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_JSON = SCRIPT_DIR / ".." / "results" / "gold_vocabulary_density.json"


def load_nllb_vocab(tokenizer_json_path):
    d = json.loads(Path(tokenizer_json_path).read_text(encoding="utf-8"))
    return set(d["model"]["vocab"].keys())


def detect_schema(tsv_path):
    """Return 'pipe_segmentation' (4-col, col 4 = full pipe-delimited seg)
    or 'suffix_stem_split' (5-col, separate suffix + src_word/stem columns),
    based on the header row / first few data rows. Raises if neither matches,
    rather than silently guessing.
    """
    with open(tsv_path, encoding="utf-8") as f:
        first = f.readline().rstrip("\n").split("\t")
        if first == ["base_word", "trg_word", "features", "suffix", "src_word"]:
            return "suffix_stem_split", True  # has header row
        # otherwise assume first line is data in the 4-column pipe format
        if len(first) == 4:
            return "pipe_segmentation", False
    raise ValueError(f"Unrecognized MorphyNet schema in {tsv_path}; inspect raw rows before adding a new branch here.")


def load_morphynet_morphemes(tsv_path, max_lines=None):
    """Extract gold morphemes from a MorphyNet inflectional TSV, split into
    stem vs. affix pieces. This matters because stems are near-word-type
    cardinality (thousands of distinct lexical roots) while affixes are a
    small closed class -- pooling them into one "morpheme" count conflates
    lexical coverage with morphological (affix) coverage, which is what the
    vocabulary-density framework actually cares about.
    """
    schema, has_header = detect_schema(tsv_path)
    stems, affixes = Counter(), Counter()
    n_words = 0
    with open(tsv_path, encoding="utf-8") as f:
        if has_header:
            next(f)
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            parts = line.rstrip("\n").split("\t")

            if schema == "pipe_segmentation":
                if len(parts) < 4:
                    continue
                seg = parts[3]
                if seg == "-" or not seg:
                    continue
                pieces = [m for m in seg.split("|") if m]
                if not pieces:
                    continue
                n_words += 1
                stems[pieces[0]] += 1
                for m in pieces[1:]:
                    affixes[m] += 1

            elif schema == "suffix_stem_split":
                if len(parts) < 5:
                    continue  # unsegmented row (multi-word negated forms etc.), skip
                suffix, stem = parts[3], parts[4]
                if suffix in ("-", "@", "") or stem in ("-", "@", ""):
                    continue
                n_words += 1
                stems[stem] += 1
                affixes[suffix] += 1

    return stems, affixes, n_words


def compute_density(morphemes, vocab):
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
    ap.add_argument("--lang", required=True, help="ISO code label, e.g. fin, rus")
    ap.add_argument("--morphynet-tsv", required=True)
    ap.add_argument("--max-lines", type=int, default=None, help="cap for speed on huge files; None = full file")
    args = ap.parse_args()

    vocab = load_nllb_vocab(args.nllb_tokenizer)
    stems, affixes, n_words = load_morphynet_morphemes(args.morphynet_tsv, args.max_lines)
    all_morphemes = stems + affixes
    density_all = compute_density(all_morphemes, vocab)
    density_stems = compute_density(stems, vocab)
    density_affixes = compute_density(affixes, vocab)

    out = {
        "gold_vocabulary_density": {
            args.lang: {
                "generated_by": "scripts/gold_vocabulary_density.py",
                "gold_source": f"MorphyNet inflectional.v1.tsv ({args.morphynet_tsv})",
                "nllb_vocab_size": len(vocab),
                "n_words_with_segmentation": n_words,
                "max_lines_used": args.max_lines,
                "pooled_stem_and_affix": density_all,
                "stems_only": density_stems,
                "affixes_only": density_affixes,
                "note": "affixes_only is the density figure most relevant to the vocabulary-density hypothesis; pooled conflates lexical stem coverage with affix coverage.",
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
    existing.setdefault("gold_vocabulary_density", {}).update(out["gold_vocabulary_density"])
    OUT_JSON.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{args.lang}: {n_words} words")
    print(f"  stems:   {density_stems['morpheme_types_total']} types, density_by_type={density_stems['density_by_type']}, density_by_token={density_stems['density_by_token']}")
    print(f"  affixes: {density_affixes['morpheme_types_total']} types, density_by_type={density_affixes['density_by_type']}, density_by_token={density_affixes['density_by_token']}")
    print(f"  pooled:  {density_all['morpheme_types_total']} types, density_by_type={density_all['density_by_type']}, density_by_token={density_all['density_by_token']}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
