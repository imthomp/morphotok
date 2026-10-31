#!/usr/bin/env python3
"""Probe: does the paper's arn->es Standard checkpoint (trained 2026-03-01 with the
pre-fix script, transformers 5.2.0, no tokenizer.tgt_lang set when building
labels) expect a different first target token than spa_Latn?

For each system, on the first N test sentences: (1) generate with NO forced
first token and report the most common first generated tokens; (2) score
chrF++ with forced_bos in {spa_Latn, eng_Latn, arn_Latn, none}. Post-fix
systems (ours, paper_vc) are controls: spa_Latn should win for them.
"""
import argparse, importlib.util, json, collections
from pathlib import Path
import torch
from sacrebleu.metrics import CHRF
from transformers import AutoModelForSeq2SeqLM

spec = importlib.util.spec_from_file_location("cc", "/home/it238/projects/morphotok/experiments/scripts/phase0_crosscheck_paper.py")
cc = importlib.util.module_from_spec(spec); spec.loader.exec_module(cc)
import morfessor, sentencepiece as spm

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=300); args = ap.parse_args()
refs = cc.read(cc.DATA/"cleaned"/"test"/"cleaned"/"tgt.txt")[:args.n]
raw = cc.read(cc.DATA/"cleaned"/"test"/"cleaned"/"src.txt")[:args.n]
seg, cas = cc.load_module("segment_morfessor"), cc.load_module("segment_morfessor_vc")
mm = morfessor.MorfessorIO().read_binary_model_file(cc.OLD_MORFESSOR)
sp = spm.SentencePieceProcessor(); sp.Load(cc.SP_MODEL); vocab = cas.build_vocab_set(sp)
paper_vc_src = [cas.cascade_line(seg.segment_line(l, mm), vocab).rstrip("\n") for l in raw]

systems = {
    "paper_standard (2026-03-01, pre-fix)": (cc.PAPER_MODELS/"nllb-600M-blocks-arn-es", raw, cc.BASE_TOK),
    "paper_vc (2026-03-16, post-fix)": (cc.PAPER_MODELS/"nllb-600M-blocks-arn-es-morfessor_vc", paper_vc_src, cc.BASE_TOK),
    "ours_standard (post-fix)": (cc.ROOT/"models/phase0_pilot/nllb-600M-arn-es-standard/best", raw, None),
}
dev = "cuda" if torch.cuda.is_available() else "cpu"
out = {}
for name, (mdir, src, toksrc) in systems.items():
    tok = cc.load_tokenizer(toksrc or mdir); tok.src_lang = "arn_Latn"
    model = AutoModelForSeq2SeqLM.from_pretrained(str(mdir)).to(dev).eval()
    ids = {k: tok.convert_tokens_to_ids(k) for k in ["spa_Latn", "eng_Latn", "arn_Latn"]}
    ids["none"] = None
    res = {}
    for label, bos in ids.items():
        preds, first = [], collections.Counter()
        with torch.no_grad():
            for i in range(0, len(src), 32):
                enc = tok(src[i:i+32], return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev)
                kw = {"forced_bos_token_id": bos} if bos is not None else {}
                o = model.generate(**enc, max_length=128, num_beams=4, **kw)
                for row in o.tolist():
                    first[tok.convert_ids_to_tokens(row[1])] += 1
                preds += tok.batch_decode(o, skip_special_tokens=True)
        res[label] = {"chrF++": round(CHRF(word_order=2).corpus_score(preds, [refs]).score, 2),
                      "first_token_counts": first.most_common(3)}
        print(name, label, res[label], flush=True)
    out[name] = res
    del model; torch.cuda.empty_cache()
Path("/home/it238/projects/morphotok/experiments/results/probe_forced_bos.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
