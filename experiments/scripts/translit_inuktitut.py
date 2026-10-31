#!/usr/bin/env python3
"""Deterministic syllabics -> romanized Inuktitut transliteration.

Why: NLLB-200's vocabulary covers almost none of Canadian Aboriginal Syllabics
(42.6% of tokens on Nunavut Hansard text become <unk>), so experiments in
syllabics would measure script coverage, not morphology. Romanized Inuktitut is
the standard alternative orthography and is what earlier Nunavut Hansard
releases shipped.

The mapping is derived from Unicode character names with Inuktitut conventions
(O-series -> u, C -> g, F -> v, Y -> j, LH -> lh as in the Nunavut Hansard
romanization, NUNAVUT H -> h, doubled letters for long vowels), then validated
against the row-aligned syllabics/romanized pairs in
EdinburghNLP/nunavut-hansard-plusplus (see validate()).
"""
import argparse
import re
import unicodedata

CONS = {"": "", "P": "p", "T": "t", "K": "k", "C": "g", "M": "m", "N": "n", "S": "s",
        "L": "l", "Y": "j", "F": "v", "R": "r", "Q": "q", "NG": "ng", "NNG": "nng", "LH": "lh"}
VOW = {"I": "i", "II": "ii", "O": "u", "OO": "uu", "A": "a", "AA": "aa"}


def build_table():
    table = {}
    for cp in range(0x1400, 0x1680):
        name = unicodedata.name(chr(cp), "")
        if not name.startswith("CANADIAN SYLLABICS "):
            continue
        body = name[len("CANADIAN SYLLABICS "):]
        if body == "NUNAVUT H":
            table[chr(cp)] = "h"
            continue
        m = re.fullmatch(r"(NNG|NG|LH|[PTKCMNSLYFRQ]?)(II|OO|AA|I|O|A)?", body)
        if not m:
            continue
        c, v = m.group(1), m.group(2) or ""
        if c not in CONS or (v and v not in VOW) or (not c and not v):
            continue
        table[chr(cp)] = CONS[c] + VOW.get(v, "")
    return table


TABLE = build_table()


def transliterate(text: str) -> str:
    out = "".join(TABLE.get(ch, ch) for ch in text)
    # Orthographic convention: syllabic final Q (ᖅ) before a K-series syllable is
    # written "qq" in romanization (ᖅᑲ -> qqa), not "qk".
    return out.replace("qk", "qq")


def validate(n=20000):
    from datasets import load_dataset
    rom = load_dataset("EdinburghNLP/nunavut-hansard-plusplus", "inuktitut")["train"]
    syl = load_dataset("EdinburghNLP/nunavut-hansard-plusplus", "inuktitut_syllabics")["train"]
    step = max(1, len(syl) // n)
    words = ok = 0
    diffs = {}
    for i in range(0, len(syl), step):
        a = transliterate(syl[i]["text"]).lower().split()
        b = rom[i]["text"].lower().split()
        if len(a) != len(b):
            continue
        for x, y in zip(a, b):
            words += 1
            if x == y:
                ok += 1
            elif len(diffs) < 25:
                diffs[x] = y
    print(f"word-level exact match: {ok}/{words} = {100*ok/words:.2f}%")
    for x, y in list(diffs.items())[:25]:
        print(f"  ours={x!r:30} ref={y!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--infile")
    ap.add_argument("--outfile")
    a = ap.parse_args()
    if a.validate:
        validate()
    if a.infile:
        with open(a.infile, encoding="utf-8") as f, open(a.outfile, "w", encoding="utf-8") as g:
            for line in f:
                g.write(transliterate(line))
