#!/usr/bin/env python3
"""Pairwise syllabics comparisons from saved predictions (seed 42): syl_morph vs
syl_bpe vs syl_char, plus en->ikt rescored after romanizing outputs and refs so it
can be compared with the romanized pilot."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from sacrebleu.metrics import CHRF, BLEU
from phase0_bootstrap_general import corpus_paired_bootstrap
from translit_inuktitut import transliterate
E = Path(__file__).resolve().parents[1]
base = E / "data_pipeline/inuktitut/pilot_data_syl/blocks/cleaned/test/cleaned"
read = lambda p: Path(p).read_text(encoding="utf-8").splitlines()
out = {}
for src, tgt, f in (("ikt", "en", "tgt.txt"), ("en", "ikt", "src.txt")):
    refs = read(base / f)
    P = {c: read(E / f"results/preds/{src}_{tgt}_{c}_e25_len256.txt") for c in ("syl_char", "syl_bpe", "syl_morph")}
    for a, b in (("syl_morph", "syl_bpe"), ("syl_morph", "syl_char"), ("syl_bpe", "syl_char")):
        out[f"{src}_{tgt}|{a}_vs_{b}"] = corpus_paired_bootstrap(CHRF(), P[a], P[b], refs)
    out[f"{src}_{tgt}|bleu"] = {c: round(BLEU().corpus_score(p, [refs]).score, 2) for c, p in P.items()}
    if tgt == "ikt":
        rr = [transliterate(r) for r in refs]
        R = {c: [transliterate(x) for x in p] for c, p in P.items()}
        out["en_ikt_romanized|scores"] = {c: round(CHRF().corpus_score(p, [rr]).score, 2) for c, p in R.items()}
        out["en_ikt_romanized|syl_morph_vs_syl_bpe"] = corpus_paired_bootstrap(CHRF(), R["syl_morph"], R["syl_bpe"], rr)
        out["en_ikt_romanized|syl_char_vs_syl_bpe"] = corpus_paired_bootstrap(CHRF(), R["syl_char"], R["syl_bpe"], rr)
(E / "results/ikt_syl_posthoc.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
