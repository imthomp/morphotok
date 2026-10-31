#!/usr/bin/env python3
"""General paired-bootstrap comparison for Phase 0 pilot checkpoints -- any
direction, any set of tokenizer-approach checkpoints, against a chosen
baseline. Supersedes phase0_predict_and_bootstrap.py's hardcoded es->arn-only
version for the 25-epoch and other follow-up runs.
"""
import argparse
import json
import random
import re
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sacrebleu.metrics import CHRF

DATA = Path("/home/it238/projects/morphotok/experiments/data_pipeline/mapudungun/pilot_data")
MODELS = Path("/home/it238/projects/morphotok/experiments/models/phase0_pilot")
LANG_CODES = {"arn": "arn_Latn", "es": "spa_Latn", "ikt": "ike_Latn", "en": "eng_Latn"}
LOW_RES = {"arn", "ikt"}  # language stored on the src.txt side


def desegment(text: str) -> str:
    return re.sub(r"@@ ?", "", text)


def load_test(direction_src: str, approach: str):
    base = DATA / "blocks" / ("cleaned" if approach == "standard" else approach) / "test"
    subdir = "cleaned" if approach == "standard" else ""
    p = base / subdir if subdir else base
    fname = "src.txt" if direction_src in LOW_RES else "tgt.txt"
    lines = (p / fname).read_text(encoding="utf-8").splitlines()
    return lines


def ref_lines():
    base = DATA / "blocks" / "cleaned" / "test" / "cleaned"
    # reference is whichever side is NOT the source; determined by caller
    return (base / "src.txt").read_text(encoding="utf-8").splitlines(), \
           (base / "tgt.txt").read_text(encoding="utf-8").splitlines()


def generate(model_dir, src_lines, src_code, tgt_code, batch_size=32, max_length=128):
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()
    tok.src_lang = src_code
    tgt_id = tok.convert_tokens_to_ids(tgt_code)
    preds = []
    with torch.no_grad():
        for i in range(0, len(src_lines), batch_size):
            enc = tok(src_lines[i:i + batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length).to(dev)
            out = model.generate(**enc, forced_bos_token_id=tgt_id, max_length=max_length, num_beams=4)
            preds.extend(tok.batch_decode(out, skip_special_tokens=True))
    del model
    torch.cuda.empty_cache()
    return preds


def bootstrap_diff(a, b, n_boot=1000, seed=42):
    random.seed(seed)
    n = len(a)
    diffs = [x - y for x, y in zip(a, b)]
    obs = sum(diffs) / n
    means = []
    for _ in range(n_boot):
        means.append(sum(diffs[random.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo, hi = means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]
    p = 2 * min(sum(1 for m in means if m <= 0), sum(1 for m in means if m >= 0)) / n_boot
    return {"observed_mean_diff": round(obs, 4), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
            "two_sided_p_approx": round(min(p, 1.0), 4), "significant_at_0.05": bool(lo > 0 or hi < 0)}


def corpus_paired_bootstrap(metric, preds_a, preds_b, refs, n_boot=1000, seed=42):
    """Koehn (2004)-style paired bootstrap on CORPUS chrF: resample sentences,
    recompute corpus chrF for both systems from summed sufficient statistics."""
    import numpy as np
    sa = np.array(metric._extract_corpus_statistics(preds_a, [refs]))
    sb = np.array(metric._extract_corpus_statistics(preds_b, [refs]))
    score = lambda s: metric._compute_score_from_stats(list(s.sum(axis=0)))
    obs_a, obs_b = score(sa), score(sb)
    rng = np.random.default_rng(seed)
    n = len(refs)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(score(sa[idx]).score - score(sb[idx]).score)
    diffs.sort()
    lo, hi = diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]
    p = 2 * min(sum(d <= 0 for d in diffs), sum(d >= 0 for d in diffs)) / n_boot
    return {"corpus_a": round(obs_a.score, 2), "corpus_b": round(obs_b.score, 2),
            "observed_diff": round(obs_a.score - obs_b.score, 4),
            "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
            "two_sided_p_approx": round(min(p, 1.0), 4), "significant_at_0.05": bool(lo > 0 or hi < 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, choices=sorted(LANG_CODES))
    ap.add_argument("--tgt", required=True, choices=sorted(LANG_CODES))
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--suffix", default="_e25")
    ap.add_argument("--baseline", default="standard")
    ap.add_argument("--conditions", nargs="+", default=["standard", "morfessor_vc", "vocab_expansion"])
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--max-length", type=int, default=128)
    args = ap.parse_args()
    global DATA, MODELS
    if args.data_dir: DATA = Path(args.data_dir)
    if args.models_dir: MODELS = Path(args.models_dir)

    src_code, tgt_code = LANG_CODES[args.src], LANG_CODES[args.tgt]
    src_all, tgt_all = ref_lines()
    src_lines = src_all if args.src in LOW_RES else tgt_all
    refs = src_all if args.tgt in LOW_RES else tgt_all

    chrf = CHRF()
    per_sentence = {}
    all_preds = {}
    pred_dir = Path(args.out_json).parent / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for cond in args.conditions:
        mdir = MODELS / f"nllb-600M-{args.src}-{args.tgt}-{cond}{args.suffix}" / "best"
        # Conditions without their own pre-segmented blocks/ dir (e.g. syllabics added-token
        # conditions) read the raw cleaned text; their tokenizer lives in the checkpoint.
        approach = cond if (DATA / "blocks" / cond).exists() else "standard"
        # segmented source for VC/VE conditions, on the whichever side is the model's INPUT
        if approach != "standard":
            seg_src = load_test(args.src, approach)
        else:
            seg_src = src_lines
        print(f"=== {cond}: {mdir} ===", flush=True)
        preds = generate(mdir, seg_src, src_code, tgt_code, max_length=args.max_length)
        if approach != "standard" and args.tgt in LOW_RES:
            preds = [desegment(p) for p in preds]
        (pred_dir / f"{args.src}_{args.tgt}_{cond}{args.suffix}.txt").write_text("\n".join(preds) + "\n", encoding="utf-8")
        all_preds[cond] = preds
        scores = [chrf.sentence_score(p, [r]).score for p, r in zip(preds, refs)]
        per_sentence[cond] = scores
        print(f"  mean sentence-chrF: {sum(scores)/len(scores):.2f}", flush=True)

    comparisons = {}
    for cond in args.conditions:
        if cond == args.baseline:
            continue
        comparisons[f"{cond}_vs_{args.baseline}"] = {
            "sentence_mean": bootstrap_diff(per_sentence[cond], per_sentence[args.baseline]),
            "corpus_chrf": corpus_paired_bootstrap(CHRF(), all_preds[cond], all_preds[args.baseline], refs),
        }

    out = {
        "direction": f"{args.src}->{args.tgt}",
        "suffix": args.suffix,
        "n_sentences": len(refs),
        "mean_sentence_chrf": {k: round(sum(v)/len(v), 2) for k, v in per_sentence.items()},
        "corpus_chrf_regenerated": {k: round(CHRF().corpus_score(v, [refs]).score, 2) for k, v in all_preds.items()},
        "comparisons": comparisons,
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
