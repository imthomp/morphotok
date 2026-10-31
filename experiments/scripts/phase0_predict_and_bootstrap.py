#!/usr/bin/env python3
"""Generate per-sentence predictions for each Phase 0 pilot condition's best
checkpoint, compute per-sentence chrF, then bootstrap resample (n=1000,
paired) to test whether the differences between conditions reported in
plan.md are distinguishable from noise. Pre-registered in the Evaluation
section (bootstrap resampling, n=1000); not run at training time since the
training script only saves aggregate test metrics.
"""

import argparse
import json
import random
import re
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sacrebleu.metrics import CHRF

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_JSON = SCRIPT_DIR / ".." / "results" / "phase0_pilot_bootstrap.json"

CONDITIONS = {
    "standard": "standard",
    "morfessor_vc": "morfessor_vc",
    "vocab_expansion": "vocab_expansion",
}


def desegment(text: str) -> str:
    return re.sub(r"@@ ?", "", text)


def load_test_src(data_dir: Path, tokenizer_approach: str) -> list[str]:
    if tokenizer_approach == "standard":
        base = data_dir / "blocks" / "cleaned" / "test" / "cleaned"
    else:
        base = data_dir / "blocks" / tokenizer_approach / "test"
    # es->arn: model source is Spanish, stored as tgt.txt in each tokenizer-approach dir
    # (only the arn/src.txt side is pre-segmented; Spanish is always copied unchanged).
    return (base / "tgt.txt").read_text(encoding="utf-8").splitlines()


def load_test_ref(data_dir: Path) -> list[str]:
    base = data_dir / "blocks" / "cleaned" / "test" / "cleaned"
    return (base / "src.txt").read_text(encoding="utf-8").splitlines()


def generate(model_dir: Path, src_lines: list[str], batch_size: int, max_length: int) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    tgt_id = tokenizer.convert_tokens_to_ids("arn_Latn")
    tokenizer.src_lang = "spa_Latn"

    preds = []
    with torch.no_grad():
        for i in range(0, len(src_lines), batch_size):
            batch = src_lines[i : i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            out = model.generate(**enc, forced_bos_token_id=tgt_id, max_length=max_length, num_beams=4)
            preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    return preds


def bootstrap_diff(scores_a, scores_b, n_boot=1000, seed=42):
    random.seed(seed)
    n = len(scores_a)
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    obs_mean_diff = sum(diffs) / n
    boot_means = []
    for _ in range(n_boot):
        sample = [diffs[random.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo = boot_means[int(0.025 * n_boot)]
    hi = boot_means[int(0.975 * n_boot)]
    p_le_zero = sum(1 for m in boot_means if m <= 0) / n_boot
    p_ge_zero = sum(1 for m in boot_means if m >= 0) / n_boot
    two_sided_p = 2 * min(p_le_zero, p_ge_zero)
    return {
        "observed_mean_diff": round(obs_mean_diff, 4),
        "ci95_lo": round(lo, 4),
        "ci95_hi": round(hi, 4),
        "two_sided_p_approx": round(min(two_sided_p, 1.0), 4),
        "significant_at_0.05": bool(lo > 0 or hi < 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--max-sentences", type=int, default=None, help="cap test set for speed; None = full 9,367")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    refs = load_test_ref(data_dir)
    if args.max_sentences:
        refs = refs[: args.max_sentences]

    chrf = CHRF()
    per_sentence_scores = {}
    for name, approach in CONDITIONS.items():
        src_lines = load_test_src(data_dir, approach)
        if args.max_sentences:
            src_lines = src_lines[: args.max_sentences]
        model_dir = models_dir / f"nllb-600M-es-arn-{name}" / "best"
        print(f"=== {name}: generating {len(src_lines)} predictions from {model_dir} ===", flush=True)
        preds = generate(model_dir, src_lines, args.batch_size, args.max_length)
        if approach != "standard":
            preds = [desegment(p) for p in preds]
        scores = [chrf.sentence_score(p, [r]).score for p, r in zip(preds, refs)]
        per_sentence_scores[name] = scores
        print(f"  mean sentence-chrF: {sum(scores)/len(scores):.2f}", flush=True)

    comparisons = {
        "morfessor_vc_vs_standard": bootstrap_diff(per_sentence_scores["morfessor_vc"], per_sentence_scores["standard"]),
        "vocab_expansion_vs_standard": bootstrap_diff(per_sentence_scores["vocab_expansion"], per_sentence_scores["standard"]),
        "vocab_expansion_vs_morfessor_vc": bootstrap_diff(per_sentence_scores["vocab_expansion"], per_sentence_scores["morfessor_vc"]),
    }

    out = {
        "phase0_pilot_bootstrap": {
            "n_sentences": len(refs),
            "n_bootstrap_resamples": 1000,
            "mean_sentence_chrf": {k: round(sum(v) / len(v), 2) for k, v in per_sentence_scores.items()},
            "comparisons": comparisons,
        }
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
