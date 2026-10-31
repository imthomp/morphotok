#!/usr/bin/env python3
"""Decisive test of the AmericasNLP paper's weak arn->es Standard baselines.

Replaying the pre-fix finetune_nllb.py under transformers 5.2.0 (the Mar 1
environment those three checkpoints record) shows the training examples had
NO language tags: input = <text> </s> <unk> (no arn_Latn prefix), labels =
<text> </s> <unk> (no spa_Latn). The paper evaluated them the standard way:
arn_Latn-prefixed input, decoder forced to start with spa_Latn. This scores
each checkpoint both ways on the same test set:
  published_format : arn_Latn + text + </s>, forced_bos=spa_Latn (paper's eval)
  training_format  : text + </s> + <unk>, decoder starts at </s>, nothing forced
If training_format is much higher, the published baseline numbers reflect a
train/eval format mismatch, not the models' ability.
"""
import json
from pathlib import Path
import torch
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE = "/home/it238/.cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/snapshots/f8d333a098d19b4fd9a8b18f94170487ad3f821d"
PAPER = Path("/nobackup/archive/usr/it238/mapudungun_final_models")
DATA = Path("/home/it238/projects/morphotok/experiments/data_pipeline/mapudungun/pilot_data/blocks/cleaned/test/cleaned")
OUT = Path("/home/it238/projects/morphotok/experiments/results/paper_baseline_format_test.json")
PRED = Path("/home/it238/projects/morphotok/experiments/results/preds")

src = (DATA / "src.txt").read_text(encoding="utf-8").splitlines()
refs = (DATA / "tgt.txt").read_text(encoding="utf-8").splitlines()
tok = AutoTokenizer.from_pretrained(BASE)
tok.add_special_tokens({"additional_special_tokens": ["arn_Latn"]})
assert tok.convert_tokens_to_ids("arn_Latn") == 256204
EOS, UNK, PAD = tok.eos_token_id, tok.unk_token_id, tok.pad_token_id
ARN, SPA = tok.convert_tokens_to_ids("arn_Latn"), tok.convert_tokens_to_ids("spa_Latn")
dev = "cuda"

def encode(lines, fmt):
    rows = []
    for l in lines:
        ids = tok(l, add_special_tokens=False)["input_ids"][:125]
        rows.append([ARN] + ids + [EOS] if fmt == "published_format" else ids + [EOS, UNK])
    n = max(map(len, rows))
    ids = torch.tensor([r + [PAD] * (n - len(r)) for r in rows])
    return ids.to(dev), (ids != PAD).long().to(dev)

def run(model, fmt, bs):
    model.generation_config.forced_bos_token_id = SPA if fmt == "published_format" else None
    preds, first = [], {}
    with torch.no_grad():
        for i in range(0, len(src), bs):
            ids, mask = encode(src[i:i + bs], fmt)
            out = model.generate(input_ids=ids, attention_mask=mask, max_length=128, num_beams=4,
                                 decoder_start_token_id=EOS)
            for row in out.tolist():
                t = tok.convert_ids_to_tokens(row[1]); first[t] = first.get(t, 0) + 1
            preds += tok.batch_decode(out, skip_special_tokens=True)
    return preds, sorted(first.items(), key=lambda x: -x[1])[:3]

result = json.loads(OUT.read_text()) if OUT.exists() else {}
for name, bs in [("nllb-600M-blocks-arn-es", 32), ("nllb-1.3B-blocks-arn-es", 32), ("nllb-200-3.3B-blocks-arn-es", 16)]:
    model = AutoModelForSeq2SeqLM.from_pretrained(str(PAPER / name)).to(dev).eval()
    for fmt in ["published_format", "training_format"]:
        key = f"{name}|{fmt}"
        if key in result: continue
        preds, first = run(model, fmt, bs)
        PRED.mkdir(parents=True, exist_ok=True)
        (PRED / f"paper_{name}_{fmt}.txt").write_text("\n".join(preds) + "\n", encoding="utf-8")
        result[key] = {"chrF++": round(CHRF(word_order=2).corpus_score(preds, [refs]).score, 2),
                       "chrF": round(CHRF().corpus_score(preds, [refs]).score, 2),
                       "BLEU": round(BLEU().corpus_score(preds, [refs]).score, 2),
                       "first_generated_tokens": first}
        print(key, result[key], flush=True)
        OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    del model; torch.cuda.empty_cache()
