#!/usr/bin/env python3
"""Retrieval-proxy baseline (after Chen et al., 2026, "Translation or Recitation?"):
for each test sentence, return the reference translation of its nearest
training-set neighbor, judged on the SOURCE side. No model, no training. If a
fine-tuned system does not clearly beat this, its score reflects train-test
overlap more than translation.

Deviation from the paper, disclosed: they pick the neighbor with a
BLEU/chrF similarity over the source; here similarity is cosine over TF-IDF
character 1-4-grams (sparse, tractable for 9,367 x 56,197 pairs). The two
should rank neighbors similarly but are not identical, so treat the number as
comparable in spirit, not to the decimal.
"""
import json
from pathlib import Path

from sacrebleu import corpus_bleu, corpus_chrf
from sacrebleu.metrics import CHRF
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_pipeline" / "mapudungun" / "cleaned"
OUT = ROOT / "results" / "phase0_retrieval_baseline.json"


def read(p):
    return Path(p).read_text(encoding="utf-8").splitlines()


def run(src_ext, tgt_ext):
    tr_src, tr_tgt = read(DATA / f"train.{src_ext}"), read(DATA / f"train.{tgt_ext}")
    te_src, te_tgt = read(DATA / f"test.{src_ext}"), read(DATA / f"test.{tgt_ext}")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), sublinear_tf=True)
    xtr = vec.fit_transform(tr_src)
    xte = vec.transform(te_src)
    preds = []
    for i in range(0, xte.shape[0], 500):
        sims = xte[i : i + 500] @ xtr.T
        preds.extend(tr_tgt[j] for j in sims.argmax(axis=1).A1)
    chrf = CHRF()
    return {
        "n_test": len(te_src),
        "corpus_chrf": round(corpus_chrf(preds, [te_tgt]).score, 2),
        "corpus_bleu": round(corpus_bleu(preds, [te_tgt]).score, 2),
        "mean_sentence_chrf": round(sum(chrf.sentence_score(p, [r]).score for p, r in zip(preds, te_tgt)) / len(te_tgt), 2),
    }


res = {"generated_by": "scripts/phase0_retrieval_baseline.py",
       "es_to_arn": run("es", "arn"), "arn_to_es": run("arn", "es")}
OUT.write_text(json.dumps({"phase0_retrieval_baseline": res}, indent=2) + "\n", encoding="utf-8")
print(json.dumps(res, indent=2))
