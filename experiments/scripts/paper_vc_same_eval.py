#!/usr/bin/env python3
"""Score the paper's arn->es Morfessor-VC checkpoints at 1.3B and 3.3B with the SAME
evaluation settings as paper_baseline_format_test.py (beam 4, max_length 128, our
regenerated test set, paper's original Morfessor model + cascade for the input),
so the corrected Standard baselines can be compared apples-to-apples."""
import importlib.util, json
from pathlib import Path
import morfessor, sentencepiece as spm, torch
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoModelForSeq2SeqLM

spec = importlib.util.spec_from_file_location("cc", "/home/it238/projects/morphotok/experiments/scripts/phase0_crosscheck_paper.py")
cc = importlib.util.module_from_spec(spec); spec.loader.exec_module(cc)
OUT = Path("/home/it238/projects/morphotok/experiments/results/paper_vc_same_eval.json")
refs = cc.read(cc.DATA / "cleaned/test/cleaned/tgt.txt")
raw = cc.read(cc.DATA / "cleaned/test/cleaned/src.txt")
seg, cas = cc.load_module("segment_morfessor"), cc.load_module("segment_morfessor_vc")
mm = morfessor.MorfessorIO().read_binary_model_file(cc.OLD_MORFESSOR)
sp = spm.SentencePieceProcessor(); sp.Load(cc.SP_MODEL); vocab = cas.build_vocab_set(sp)
vc_src = [cas.cascade_line(seg.segment_line(l, mm), vocab).rstrip("\n") for l in raw]
res = json.loads(OUT.read_text()) if OUT.exists() else {}
for name, path, bs in [
    ("nllb-1.3B-blocks-arn-es-morfessor_vc", cc.PAPER_MODELS / "nllb-1.3B-blocks-arn-es-morfessor_vc", 32),
    ("nllb-200-3.3B-blocks-arn-es-morfessor_vc", Path("/nobackup/archive/usr/it238/mapudungun_models/nllb-200-3.3B-blocks-arn-es-morfessor_vc/best"), 16)]:
    if name in res: continue
    preds = cc.generate(path, vc_src, tok_source=cc.BASE_TOK, batch_size=bs)
    (cc.PRED_DIR / f"paper_{name}.txt").write_text("\n".join(preds) + "\n", encoding="utf-8")
    res[name] = {"chrF++": round(CHRF(word_order=2).corpus_score(preds, [refs]).score, 2),
                 "chrF": round(CHRF().corpus_score(preds, [refs]).score, 2),
                 "BLEU": round(BLEU().corpus_score(preds, [refs]).score, 2)}
    print(name, res[name], flush=True)
    OUT.write_text(json.dumps(res, indent=2) + "\n")
