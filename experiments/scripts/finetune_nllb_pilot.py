#!/usr/bin/env python3
"""Phase 0 pilot fine-tuning: NLLB-200 on Mapudungun<->Spanish, three
conditions (Standard BPE, Morfessor-VC, Vocab Expansion), pre-registered
undertraining check at 15 GPU-hours/run (notes/plan.md, "Hyperparameter
grid -- pre-registered 2026-09-15").

Adapted from mapudungun-mt/scripts/training/finetune_nllb.py (same core
training loop, reused directly for the standard/morfessor_vc conditions)
with one addition: real vocab-expansion support for tokenizer_approach=
"vocab_expansion" (Morfessor-VE), which has no precedent in the original
script -- new OOV morpheme tokens are added to the tokenizer and their
embeddings initialized by averaging the embeddings of the constituent
subword tokens the *old* tokenizer would have used to encode that same
surface string (the "average of constituent subword-token embeddings" init
described in notes/plan.md's Methods section).
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import sacrebleu

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LANG_CODES = {"arn": "arn_Latn", "es": "spa_Latn", "ikt": "ike_Latn", "en": "eng_Latn"}
# Languages absent from NLLB-200 get a new tag token initialized as the mean of
# these existing tags. For Inuktitut there is no related NLLB language; reusing
# the same proxy set is an arbitrary but fixed choice (a single embedding row).
PROXY_LANGS = ["ayr_Latn", "grn_Latn", "quy_Latn"]
NEW_LANG_TAGS = {"arn_Latn", "ike_Latn"}
# Which side of the stored pairs (src.txt) holds the non-Spanish/English language.
LOW_RES = {"arn", "ikt"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    p.add_argument("--src", choices=sorted(LANG_CODES), required=True)
    p.add_argument("--tgt", choices=sorted(LANG_CODES), required=True)
    p.add_argument("--data-dir", required=True, help="pilot_data root (contains blocks/)")
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--tokenizer-approach",
        required=True,
        choices=["standard", "morfessor_vc", "vocab_expansion"],
    )
    p.add_argument("--gpu-hours-budget", type=float, default=15.0, help="logged only, for run_config.json")
    p.add_argument("--new-tokens-file", default=None, help="vocab_expansion only: override new_tokens.json (e.g. a frequency-capped subset)")
    p.add_argument("--added-tokens-file", default=None, help="JSON list of plain tokens to add (syllabics coverage conditions); new rows use transformers' mean-resizing init")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--early-stopping-patience", type=int, default=3)
    p.add_argument("--num-beams", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_split(data_dir, tokenizer_approach, split, src):
    if tokenizer_approach == "standard":
        base = Path(data_dir) / "blocks" / "cleaned" / split / "cleaned"
    else:
        base = Path(data_dir) / "blocks" / tokenizer_approach / split
    arn_lines = (base / "src.txt").read_text(encoding="utf-8").splitlines()
    es_lines = (base / "tgt.txt").read_text(encoding="utf-8").splitlines()
    assert len(arn_lines) == len(es_lines)
    if src not in LOW_RES:
        return Dataset.from_dict({"src": es_lines, "tgt": arn_lines})
    return Dataset.from_dict({"src": arn_lines, "tgt": es_lines})


def add_lang_token(tokenizer, model, tag):
    if tag in tokenizer.get_vocab():
        return
    tokenizer.add_special_tokens({"additional_special_tokens": [tag]})
    model.resize_token_embeddings(len(tokenizer))
    proxy_ids = [tokenizer.convert_tokens_to_ids(l) for l in PROXY_LANGS]
    new_id = tokenizer.convert_tokens_to_ids(tag)
    with torch.no_grad():
        shared = model.model.shared.weight
        shared[new_id] = shared[proxy_ids].mean(dim=0)


def add_vocab_expansion_tokens(tokenizer, model, new_tokens_path: Path):
    """Add OOV morpheme tokens found by prep_vocab_expansion_data.py, with
    average-embedding init from each token's own encoding under the *old*
    (pre-expansion) tokenizer.
    """
    new_tokens = json.loads(new_tokens_path.read_text(encoding="utf-8"))
    new_tokens = [t for t in new_tokens if t not in tokenizer.get_vocab()]
    if not new_tokens:
        logger.info("No new vocab-expansion tokens to add.")
        return

    # Compute each new token's init embedding BEFORE resizing, using the old tokenizer
    # to encode the surface string it represents (strip the "@@" continuation marker).
    old_embeddings = model.model.shared.weight.detach().clone()
    init_vectors = []
    for tok in new_tokens:
        surface = tok[:-2] if tok.endswith("@@") else tok
        piece_ids = tokenizer.encode(surface, add_special_tokens=False)
        if not piece_ids:
            piece_ids = [tokenizer.unk_token_id]
        init_vectors.append(old_embeddings[piece_ids].mean(dim=0))

    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        shared = model.model.shared.weight
        for tok, vec in zip(new_tokens, init_vectors):
            shared[tokenizer.convert_tokens_to_ids(tok)] = vec

    logger.info(f"Vocab expansion: added {len(new_tokens)} new tokens, embeddings averaged from old tokenizer encoding.")


def build_preprocess_fn(tokenizer, src_lang_code, tgt_lang_code, max_length):
    def preprocess(examples):
        tokenizer.src_lang = src_lang_code
        tokenizer.tgt_lang = tgt_lang_code
        return tokenizer(examples["src"], text_target=examples["tgt"], max_length=max_length, truncation=True)
    return preprocess


def desegment(text: str) -> str:
    import re
    return re.sub(r"@@ ?", "", text)


def build_compute_metrics(tokenizer, desegment_output: bool):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = [p.strip() for p in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        decoded_labels = [l.strip() for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        if desegment_output:
            decoded_preds = [desegment(p) for p in decoded_preds]
            decoded_labels = [desegment(l) for l in decoded_labels]
        return {
            "chrf": round(sacrebleu.corpus_chrf(decoded_preds, [decoded_labels]).score, 2),
            "bleu": round(sacrebleu.corpus_bleu(decoded_preds, [decoded_labels]).score, 2),
        }
    return compute_metrics


def main():
    args = parse_args()
    assert args.src != args.tgt

    src_code, tgt_code = LANG_CODES[args.src], LANG_CODES[args.tgt]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    desegment_output = (args.tokenizer_approach != "standard" and args.tgt in LOW_RES)

    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    for code in (src_code, tgt_code):
        if code in NEW_LANG_TAGS:
            add_lang_token(tokenizer, model, code)

    if args.added_tokens_file:
        from tokenizers import pre_tokenizers
        # With prepend_scheme="always", every text span between added tokens gets its own
        # leading "▁", so "ᓯᑕᒥᖅ, ..." would decode as "ᓯᑕᒥᖅ , ...". "first" only marks the
        # start of the string; real spaces still become "▁".
        tokenizer._tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="first", split=True)
        added = json.loads(Path(args.added_tokens_file).read_text(encoding="utf-8"))
        n = tokenizer.add_tokens(added)
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Added {n} tokens from {args.added_tokens_file}")

    if args.tokenizer_approach == "vocab_expansion":
        new_tokens_path = Path(args.new_tokens_file) if args.new_tokens_file else Path(args.data_dir) / "blocks" / "vocab_expansion" / "new_tokens.json"
        add_vocab_expansion_tokens(tokenizer, model, new_tokens_path)

    tgt_token_id = tokenizer.convert_tokens_to_ids(tgt_code)
    model.generation_config.forced_bos_token_id = tgt_token_id

    raw = {split: load_split(args.data_dir, args.tokenizer_approach, split, args.src) for split in ("train", "dev", "test")}
    logger.info(f"train: {len(raw['train']):,}  dev: {len(raw['dev']):,}  test: {len(raw['test']):,}")

    preprocess = build_preprocess_fn(tokenizer, src_code, tgt_code, args.max_length)
    tokenized = {s: d.map(preprocess, batched=True, remove_columns=["src", "tgt"]) for s, d in raw.items()}
    # Regression guard for the AmericasNLP-paper bug: every example must carry the
    # source tag first and the label must start with the target tag.
    s_id, t_id = tokenizer.convert_tokens_to_ids(src_code), tokenizer.convert_tokens_to_ids(tgt_code)
    for ex in tokenized["train"].select(range(min(200, len(tokenized["train"])))):
        assert ex["input_ids"][0] == s_id, f"input missing {src_code}: {ex['input_ids'][:5]}"
        assert ex["labels"][0] == t_id, f"labels missing {tgt_code}: {ex['labels'][:5]}"
    logger.info(f"Language-tag check passed: inputs start with {src_code}, labels with {tgt_code}")
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="chrf",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=args.max_length,
        generation_num_beams=args.num_beams,
        logging_steps=200,
        fp16=torch.cuda.is_available(),
        seed=args.seed,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["dev"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer, desegment_output),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()
    trainer.save_model(str(output_dir / "best"))
    tokenizer.save_pretrained(str(output_dir / "best"))

    test_results = trainer.predict(tokenized["test"], metric_key_prefix="test", num_beams=args.num_beams, max_length=args.max_length)
    logger.info(f"Test results: {test_results.metrics}")
    (output_dir / "test_results.json").write_text(json.dumps(test_results.metrics, indent=2))


if __name__ == "__main__":
    main()
