#!/usr/bin/env python3
"""
Quantify the es->arn "source-copy" failure mode described qualitatively in
the Mapudungun paper (main.tex, Analysis section): the model sometimes
reproduces the Spanish source verbatim in Mapudungun orthography instead of
producing an actual Mapudungun translation.

Flags a prediction as likely source-copy if it's much more similar to the
Spanish source (ES) than to the Mapudungun reference (REF), using normalized
character-level SequenceMatcher ratio as a cheap proxy for near-verbatim
copying. This is a heuristic, not a semantic judgment -- read the flagged
examples, don't just trust the count.

Input: predictions readable file. Output: SSOT JSON under results/.
"""

import argparse
import difflib
import json
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = SCRIPT_DIR / ".." / "results" / "source_copy_failure_analysis.json"


def parse_readable(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n(?=\[\d+\])", text)
    examples = []
    for b in blocks:
        m = re.search(r"ES:\s*(.*)\n\s*REF:\s*(.*)\n\s*PRED:\s*(.*)", b)
        if m:
            examples.append({
                "es": m.group(1).strip(),
                "ref": m.group(2).strip(),
                "pred": m.group(3).strip(),
            })
    return examples


def norm(s):
    return re.sub(r"[^a-záéíóúñü ]", "", s.lower())


def flag_source_copy(examples, sim_es_threshold=0.6, margin=0.2):
    flagged = []
    for ex in examples:
        es_n, ref_n, pred_n = norm(ex["es"]), norm(ex["ref"]), norm(ex["pred"])
        if len(pred_n) < 5:
            continue
        sim_es = difflib.SequenceMatcher(None, es_n, pred_n).ratio()
        sim_ref = difflib.SequenceMatcher(None, ref_n, pred_n).ratio()
        if sim_es > sim_es_threshold and sim_es > sim_ref + margin:
            flagged.append({**ex, "sim_es_pred": round(sim_es, 3), "sim_ref_pred": round(sim_ref, 3)})
    flagged.sort(key=lambda x: -x["sim_es_pred"])
    return flagged


def merge_json(path, new_data):
    path = Path(path)
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    existing.update(new_data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("readable_predictions", help="Path to a *_readable.txt predictions file (ES/REF/PRED triples)")
    ap.add_argument("--condition", default="unspecified", help="Label for which model/tokenization condition this file is from")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    examples = parse_readable(args.readable_predictions)
    flagged = flag_source_copy(examples)

    result = {
        "source_copy_failure_analysis": {
            args.condition: {
                "generated_by": "scripts/source_copy_failure_analysis.py",
                "input_file": str(args.readable_predictions),
                "n_examples": len(examples),
                "n_flagged": len(flagged),
                "pct_flagged": round(100 * len(flagged) / len(examples), 3) if examples else None,
                "method": (
                    "Normalized character-level SequenceMatcher ratio; flagged if "
                    "sim(ES,PRED) > 0.6 and sim(ES,PRED) - sim(REF,PRED) > 0.2. "
                    "Heuristic proxy for near-verbatim source copying, not a semantic judgment."
                ),
                "top_examples": flagged[:20],
            }
        }
    }
    merge_json(args.out, result)
    print(f"{args.condition}: {len(flagged)}/{len(examples)} ({100*len(flagged)/len(examples):.2f}%) flagged as likely source-copy")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
