#!/usr/bin/env python3
"""
Cross NLLB-200's official per-language resource level (Table 1 of the NLLB
paper) against this project's morphological-type classification, to answer
the question that actually motivates the thesis: how many NLLB-200 languages
are BOTH morphologically complex AND low-resource -- i.e. the population a
vocabulary-aligned tokenization method could plausibly help.

Why this script exists (2026-08-19): the raw "80.6% of NLLB-200 is
morphologically complex" statistic (see count_nllb_morphological_complexity.py)
turned out to be a base-rate artifact -- WALS Chapter 20A's own summary
reports ~75% of the world's languages are non-isolating, so NLLB-200's 80.6%
isn't telling you anything specific about NLLB. The number that actually
supports the "who cares" motivation is the resource x morphology intersection
computed here.

Source of resource labels: NLLB paper (Team et al. 2022), Table 1, pages
12-15 of the PDF (arxiv.org/pdf/2207.04672), extracted via pypdf and hand
-verified: the parsed 150 Low / 54 High split exactly matches the paper's own
stated "150 low-resource languages (LRLs) and 54 high-resource languages
(HRLs)" -- this exact match on a total the paper states independently is a
real (if partial) validation of the extraction, not just an assumption.
Raw extracted table text: data/nllb_paper_table1_raw_extract.txt
Parsed code->resource-level mapping: data/nllb_paper_table1_resource_level.json

SSOT output: results/nllb_morphology_stats.json (merged in, not overwritten --
adds a new "resource_x_morphology_crossing" key alongside the existing
morphology-only stats).
"""

import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_JSON = os.path.join(SCRIPT_DIR, "..", "data", "nllb_paper_table1_resource_level.json")
MORPH_JSON = os.path.join(SCRIPT_DIR, "..", "results", "nllb_morphology_stats.json")
OUT_JSON = MORPH_JSON  # same SSOT file, merged in

COMPLEX_TYPES = {"agglutinative", "fusional", "introflexive", "polysynthetic"}


def merge_json(path, new_data):
    existing = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}
    existing.update(new_data)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main():
    resource = json.load(open(RESOURCE_JSON, encoding="utf-8"))
    morph_data = json.load(open(MORPH_JSON, encoding="utf-8"))
    per_lang = morph_data["nllb_morphology_stats"]["per_language"]

    joined = []
    unmatched = []
    for l in per_lang:
        code = l["flores_code"]
        if code not in resource:
            unmatched.append(code)
            continue
        joined.append({
            "flores_code": code,
            "base_code": l["base_code"],
            "morph_type": l["morph_type"],
            "resource_level": resource[code],
            "is_complex": l["morph_type"] in COMPLEX_TYPES,
        })

    n_total = len(joined)
    n_low = sum(1 for j in joined if j["resource_level"] == "Low")
    n_high = sum(1 for j in joined if j["resource_level"] == "High")
    n_low_complex = sum(1 for j in joined if j["resource_level"] == "Low" and j["is_complex"])
    n_high_complex = sum(1 for j in joined if j["resource_level"] == "High" and j["is_complex"])
    n_complex_total = n_low_complex + n_high_complex

    out = {
        "resource_x_morphology_crossing": {
            "generated_by": "scripts/cross_nllb_resource_morphology.py",
            "resource_source": (
                "NLLB paper (Team et al. 2022), Table 1, pp.12-15 of "
                "arxiv.org/pdf/2207.04672 -- extracted via pypdf 2026-08-19. "
                "Parsed 150 Low / 54 High exactly matches the paper's own "
                "stated total, a real cross-check on the extraction."
            ),
            "n_total_matched": n_total,
            "n_unmatched": len(unmatched),
            "unmatched_codes": unmatched,
            "n_low_resource": n_low,
            "n_low_resource_and_complex": n_low_complex,
            "pct_low_resource_that_is_complex": round(100 * n_low_complex / n_low, 1) if n_low else None,
            "n_high_resource": n_high,
            "n_high_resource_and_complex": n_high_complex,
            "pct_high_resource_that_is_complex": round(100 * n_high_complex / n_high, 1) if n_high else None,
            "n_complex_and_low_resource_of_all_204": n_low_complex,
            "pct_of_all_204_that_is_complex_and_low_resource": round(100 * n_low_complex / n_total, 1),
            "headline_finding": (
                "116 of 204 FLORES-200 languages (56.9%) are BOTH morphologically "
                "complex AND low-resource by NLLB's own <1M-bitext threshold -- "
                "this is the population a vocabulary-aligned tokenization method "
                "could plausibly help, and it does NOT reduce to the raw "
                "'80.6% complex' base-rate stat, which conflates high- and "
                "low-resource languages and is largely explained by the global "
                "typological base rate (WALS 20A: ~75% of all languages "
                "non-isolating), not anything NLLB-specific."
            ),
            "counterintuitive_note": (
                "Complexity rate is actually slightly HIGHER among NLLB's "
                "54 high-resource languages (85.2%) than among its 150 "
                "low-resource languages (77.3%). Complex morphology is NOT "
                "disproportionately concentrated in NLLB's low-resource tier -- "
                "so the 'who cares' argument should rest on the raw count (116 "
                "languages) and the tokenization-quality mechanism, not on a "
                "claim that complexity and low-resource-ness are correlated "
                "within NLLB-200, because they aren't (if anything, mildly "
                "anti-correlated in this sample)."
            ),
            "per_language": joined,
        }
    }

    merge_json(OUT_JSON, out)

    print(f"Matched {n_total}/204 languages to a resource level (unmatched: {unmatched})")
    print(f"Low-resource: {n_low}, of which complex: {n_low_complex} ({100*n_low_complex/n_low:.1f}%)")
    print(f"High-resource: {n_high}, of which complex: {n_high_complex} ({100*n_high_complex/n_high:.1f}%)")
    print(f"Complex AND low-resource (the actual motivating population): {n_low_complex}/{n_total} ({100*n_low_complex/n_total:.1f}%)")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
