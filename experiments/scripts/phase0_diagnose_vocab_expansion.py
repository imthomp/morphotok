#!/usr/bin/env python3
"""Diagnose why Vocab Expansion (Morfessor-VE) significantly underperformed
Standard BPE / Morfessor-VC in the Phase 0 pilot (results/phase0_pilot_bootstrap.json).

Question: does the model actually use its 13,024 new embedding tokens at
inference time, or do they sit unused while the model falls back to
standard subword fragments? Two possible readings:
  - New tokens rarely appear in generated output -> the new embeddings never
    became useful (init/warmup problem), fixable before Phase 1.
  - New tokens appear often and it's still worse -> the method itself is
    hurting, not an implementation bug.

Compares new-token usage rate in generated output against the rate in the
gold reference's own pre-segmented text (the "ceiling" -- how often a
perfect model would be expected to use them), not just an arbitrary baseline.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_JSON = SCRIPT_DIR / ".." / "results" / "phase0_vocab_expansion_diagnosis.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--new-tokens-json", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=128)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    new_tokens = set(json.loads(Path(args.new_tokens_json).read_text(encoding="utf-8")))

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    new_token_ids = {tokenizer.convert_tokens_to_ids(t) for t in new_tokens}
    new_token_ids.discard(tokenizer.unk_token_id)

    # Reference: gold Mapudungun, already pre-segmented with the @@ convention
    # by prep_vocab_expansion_data.py -- re-tokenize it to see how often the
    # tokenizer's own new vocab items appear in the GOLD text (the ceiling).
    ref_lines = (data_dir / "blocks" / "vocab_expansion" / "test" / "src.txt").read_text(encoding="utf-8").splitlines()
    ref_new_token_hits = 0
    ref_total_tokens = 0
    for line in ref_lines:
        ids = tokenizer(line.replace("@@ ", ""), add_special_tokens=False)["input_ids"]
        ref_total_tokens += len(ids)
        ref_new_token_hits += sum(1 for i in ids if i in new_token_ids)

    # Generated: run the model on the Spanish source, see how often ITS
    # output actually uses the new tokens.
    src_lines = (data_dir / "blocks" / "vocab_expansion" / "test" / "tgt.txt").read_text(encoding="utf-8").splitlines()
    tgt_id = tokenizer.convert_tokens_to_ids("arn_Latn")
    tokenizer.src_lang = "spa_Latn"

    gen_new_token_hits = 0
    gen_total_tokens = 0
    n_sentences_with_new_token = 0
    with torch.no_grad():
        for i in range(0, len(src_lines), args.batch_size):
            batch = src_lines[i : i + args.batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length).to(device)
            out = model.generate(**enc, forced_bos_token_id=tgt_id, max_length=args.max_length, num_beams=4)
            for seq in out.tolist():
                # strip padding/special tokens for counting
                ids = [t for t in seq if t not in tokenizer.all_special_ids]
                gen_total_tokens += len(ids)
                hits = sum(1 for t in ids if t in new_token_ids)
                gen_new_token_hits += hits
                if hits > 0:
                    n_sentences_with_new_token += 1
            if i % (args.batch_size * 20) == 0:
                print(f"  {i}/{len(src_lines)} done, running gen new-token rate so far: "
                      f"{gen_new_token_hits}/{gen_total_tokens}", flush=True)

    result = {
        "n_new_tokens_in_vocab": len(new_tokens),
        "reference_new_token_rate": round(ref_new_token_hits / ref_total_tokens, 5) if ref_total_tokens else None,
        "reference_new_token_hits": ref_new_token_hits,
        "reference_total_tokens": ref_total_tokens,
        "generated_new_token_rate": round(gen_new_token_hits / gen_total_tokens, 5) if gen_total_tokens else None,
        "generated_new_token_hits": gen_new_token_hits,
        "generated_total_tokens": gen_total_tokens,
        "n_sentences_total": len(src_lines),
        "n_sentences_with_ge1_new_token": n_sentences_with_new_token,
        "frac_sentences_with_ge1_new_token": round(n_sentences_with_new_token / len(src_lines), 4),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
