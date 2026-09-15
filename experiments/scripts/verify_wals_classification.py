#!/usr/bin/env python3
"""
Verify count_nllb_morphological_complexity.py's WALS_20A_SOURCE-tagged entries
against the actual WALS dataset, replacing a manual spot-check with a full
programmatic join.

Downloads the WALS CLDF dataset (cldf-datasets/wals on GitHub: values.csv,
languages.csv, codes.csv) and joins WALS 20A ("Fusion of Selected Inflectional
Formatives") datapoints to our classification via ISO 639-3 code, then reports
matches, mismatches, and unmatched entries.

Run once with network access (BYU login nodes have it; compute nodes may not).
Caches the three WALS CSVs under data/wals_cache/ so subsequent runs don't
re-fetch.
"""

import csv
import json
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / ".." / "data" / "wals_cache"
MORPH_JSON = SCRIPT_DIR / ".." / "results" / "nllb_morphology_stats.json"
OUT_JSON = SCRIPT_DIR / ".." / "results" / "wals_verification.json"

WALS_BASE = "https://raw.githubusercontent.com/cldf-datasets/wals/master/cldf/"
FILES = ["values.csv", "languages.csv", "codes.csv"]


def fetch_cached(name):
    path = CACHE_DIR / name
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(WALS_BASE + name, path)
    return path


def wals_to_simple(wals_val):
    if wals_val is None:
        return None
    v = wals_val.lower()
    if "isolating" in v and "concatenative" not in v and "tonal" not in v:
        return "isolating"
    if v == "exclusively concatenative":
        return "concatenative(agglutinative-or-fusional)"
    if "ablaut" in v:
        return "introflexive"
    if v == "exclusively tonal":
        return "tonal"
    if "isolating" in v and "concatenative" in v:
        return "isolating/concatenative(mixed)"
    if "tonal" in v and "isolating" in v:
        return "isolating/tonal"
    if "tonal" in v and "concatenative" in v:
        return "tonal/concatenative(mixed)"
    return v


def main():
    codes_path = fetch_cached("codes.csv")
    languages_path = fetch_cached("languages.csv")
    values_path = fetch_cached("values.csv")

    codes = {}
    with open(codes_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            codes[row["ID"]] = row["Name"]

    lang_to_iso = {}
    with open(languages_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            iso = row["ISO639P3code"]
            if iso:
                lang_to_iso.setdefault(iso, []).append(row["ID"])

    values_20a = {}
    with open(values_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["Parameter_ID"] == "20A":
                values_20a[row["Language_ID"]] = codes.get(row["Code_ID"], row["Value"])

    morph = json.loads(MORPH_JSON.read_text(encoding="utf-8"))["nllb_morphology_stats"]
    per_lang = morph["per_language"]
    wals_direct_entries = [l for l in per_lang if l["source"] and l["source"].startswith("WALS_20A")]

    results = []
    for l in wals_direct_entries:
        base = l["base_code"]
        found_val, found_wals_id = None, None
        for wid in lang_to_iso.get(base, []):
            if wid in values_20a:
                found_val, found_wals_id = values_20a[wid], wid
                break
        simplified = wals_to_simple(found_val)
        consistent = None
        if simplified is not None:
            if simplified == "isolating":
                consistent = l["morph_type"] == "isolating"
            elif simplified.startswith("concatenative"):
                consistent = l["morph_type"] in ("agglutinative", "fusional")
            elif simplified == "introflexive":
                consistent = l["morph_type"] == "introflexive"
            else:
                consistent = None  # mixed/ambiguous WALS categories, not auto-judged
        results.append({
            "base_code": base, "name": l["name"], "our_type": l["morph_type"],
            "wals_lang_id": found_wals_id, "wals_20a_value": found_val,
            "wals_simplified": simplified, "found": found_val is not None,
            "consistent_with_ours": consistent,
        })

    n_found = sum(1 for r in results if r["found"])
    n_checked = sum(1 for r in results if r["consistent_with_ours"] is not None)
    n_consistent = sum(1 for r in results if r["consistent_with_ours"] is True)
    n_mismatch = sum(1 for r in results if r["consistent_with_ours"] is False)

    out = {
        "wals_verification": {
            "generated_by": "scripts/verify_wals_classification.py",
            "source": "cldf-datasets/wals on GitHub (values.csv, languages.csv, codes.csv), joined via ISO 639-3",
            "n_wals_direct_tagged_entries": len(wals_direct_entries),
            "n_matched_to_real_wals_datapoint": n_found,
            "n_auto_checkable": n_checked,
            "n_consistent": n_consistent,
            "n_mismatch": n_mismatch,
            "mismatches": [r for r in results if r["consistent_with_ours"] is False],
            "unmatched": [r["base_code"] for r in results if not r["found"]],
            "all_results": results,
        }
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"WALS-direct-tagged entries: {len(wals_direct_entries)}")
    print(f"Matched to a real WALS 20A datapoint: {n_found}")
    print(f"Auto-checkable for consistency: {n_checked} (consistent: {n_consistent}, mismatch: {n_mismatch})")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
