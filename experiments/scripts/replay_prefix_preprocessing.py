#!/usr/bin/env python3
"""Replay the AmericasNLP paper's training preprocessing to show what the model saw.

Usage (needs a venv with the given transformers version):
  python replay_prefix_preprocessing.py prefix   # pre-fix script logic (no tokenizer.tgt_lang)
  python replay_prefix_preprocessing.py fixed    # fixed script logic (tgt_lang set)

Run 'prefix' under transformers==5.2.0 (the version the three affected arn->es
Standard checkpoints record): inputs/labels come out with NO language tags and a
trailing '</s> <unk>'. Under 4.57.6, 'fixed' gives 'arn_Latn ... </s>' /
'spa_Latn ... </s>' and 'prefix' crashes. Originally run as /tmp/replay_prep.py on
2026-09-28 with a scratch venv (/tmp/tf52: pip install transformers==5.2.0
sentencepiece tokenizers protobuf). See notes/plan.md, "Code/history audit".
"""
import sys, transformers
from transformers import AutoTokenizer
BASE = "/home/it238/.cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/snapshots/f8d333a098d19b4fd9a8b18f94170487ad3f821d"
fixed = sys.argv[1] == "fixed"
tok = AutoTokenizer.from_pretrained(BASE)
print("transformers", transformers.__version__, "| class", type(tok).__name__, "| mode", sys.argv[1])
tok.add_special_tokens({"additional_special_tokens": ["arn_Latn"]})
tok.src_lang = "arn_Latn"
if fixed:
    tok.tgt_lang = "spa_Latn"
src = ["Pichi ngütramkayayu.", "Chem l'awen' ñi kümen."]
tgt = ["Voy a conversar un poco.", "¿Qué remedio es bueno?"]
out = tok(src, text_target=tgt, max_length=128, truncation=True)
for i in range(2):
    print(" input :", tok.convert_ids_to_tokens(out["input_ids"][i]))
    print(" labels:", tok.convert_ids_to_tokens(out["labels"][i]))
