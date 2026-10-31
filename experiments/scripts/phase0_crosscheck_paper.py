#!/usr/bin/env python3
"""Apples-to-apples check of the Phase 0 arn->es result against the
AmericasNLP paper's own 600M checkpoints.

Why: the paper's table (chrF++, word_order=2) has 600M Standard BPE arn->es at
34.9 and Morfessor-VC at 43.16; the Phase 0 pilot got 47.67 / 46.48 but logged
plain chrF (word_order=0) during training. Different metric, possibly
different training setup. This scores all four systems on the SAME test set
with the SAME metrics (chrF++, chrF, BLEU) and saves predictions so later
bootstraps don't need to regenerate them.

Systems (all arn->es, our regenerated cleaned test set, 9,367 sentences):
  ours_standard, ours_vc      Phase 0 pilot checkpoints
  paper_standard              paper's nllb-600M-blocks-arn-es
  paper_vc                    paper's nllb-600M-blocks-arn-es-morfessor_vc, fed
                              our test set segmented with the paper's ORIGINAL
                              Morfessor model + its cascade merge (its own
                              segment_morfessor.py / segment_morfessor_vc.py logic)
"""
import argparse
import importlib.util
import json
from pathlib import Path

import morfessor
import sentencepiece as spm
import torch
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path("/home/it238/projects/morphotok/experiments")
PAPER_MODELS = Path("/nobackup/archive/usr/it238/mapudungun_final_models")
PAPER_REPO = Path("/home/it238/projects/mapudungun/mapudungun-mt/scripts/tokenization")
OLD_MORFESSOR = "/nobackup/archive/usr/it238/mapudungun_morfessor_model.bin"
BASE_TOK = ("/home/it238/.cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/snapshots/"
            "f8d333a098d19b4fd9a8b18f94170487ad3f821d")
SP_MODEL = ("/home/it238/.cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/snapshots/"
            "f8d333a098d19b4fd9a8b18f94170487ad3f821d/sentencepiece.bpe.model")
DATA = ROOT / "data_pipeline" / "mapudungun" / "pilot_data" / "blocks"
OUT = ROOT / "results" / "phase0_crosscheck_paper.json"
PRED_DIR = ROOT / "results" / "preds"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, PAPER_REPO / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def read(p):
    return Path(p).read_text(encoding="utf-8").splitlines()


def load_tokenizer(tok_source):
    """The paper's checkpoints declare a tokenizer class (TokenizersBackend) that
    this transformers version lacks, so -- as the paper's own predict_test.py did --
    use the base NLLB tokenizer and append arn_Latn (id 256204, per the
    checkpoint's added_tokens.json)."""
    tok = AutoTokenizer.from_pretrained(str(tok_source))
    if "arn_Latn" not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": ["arn_Latn"]})
    assert tok.convert_tokens_to_ids("arn_Latn") == 256204, tok.convert_tokens_to_ids("arn_Latn")
    return tok


def generate(model_dir, src_lines, tok_source=None, batch_size=32, max_length=128):
    tok = load_tokenizer(tok_source or model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()
    tok.src_lang = "arn_Latn"
    tgt_id = tok.convert_tokens_to_ids("spa_Latn")
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


def score(preds, refs):
    return {
        "chrF++": round(CHRF(word_order=2).corpus_score(preds, [refs]).score, 2),
        "chrF": round(CHRF().corpus_score(preds, [refs]).score, 2),
        "BLEU": round(BLEU().corpus_score(preds, [refs]).score, 2),
    }


def main():
    argparse.ArgumentParser().parse_args()
    refs = read(DATA / "cleaned" / "test" / "cleaned" / "tgt.txt")
    raw_src = read(DATA / "cleaned" / "test" / "cleaned" / "src.txt")
    ours_vc_src = read(DATA / "morfessor_vc" / "test" / "src.txt")

    # paper's original segmentation pipeline applied to OUR test set
    seg = load_module("segment_morfessor")
    cas = load_module("segment_morfessor_vc")
    model = morfessor.MorfessorIO().read_binary_model_file(OLD_MORFESSOR)
    sp = spm.SentencePieceProcessor()
    sp.Load(SP_MODEL)
    vocab = cas.build_vocab_set(sp)
    paper_vc_src = [cas.cascade_line(seg.segment_line(l, model), vocab).rstrip("\n") for l in raw_src]

    systems = {
        "ours_standard": (ROOT / "models/phase0_pilot/nllb-600M-arn-es-standard/best", raw_src),
        "ours_vc": (ROOT / "models/phase0_pilot/nllb-600M-arn-es-morfessor_vc/best", ours_vc_src),
        "paper_standard": (PAPER_MODELS / "nllb-600M-blocks-arn-es", raw_src),
        "paper_vc": (PAPER_MODELS / "nllb-600M-blocks-arn-es-morfessor_vc", paper_vc_src),
    }
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    result = json.loads(OUT.read_text())["phase0_crosscheck_paper"] if OUT.exists() else {}
    for name, (mdir, src) in systems.items():
        if name in result:
            print(f"skip {name} (already scored)", flush=True)
            continue
        print(f"=== {name}: {mdir} ===", flush=True)
        preds = generate(mdir, src, tok_source=BASE_TOK if name.startswith("paper_") else None)
        (PRED_DIR / f"arn_es_{name}.txt").write_text("\n".join(preds) + "\n", encoding="utf-8")
        result[name] = score(preds, refs)
        print(name, result[name], flush=True)
        OUT.write_text(json.dumps({"phase0_crosscheck_paper": result}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
