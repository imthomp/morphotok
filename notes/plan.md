# Vocabulary-Aligned Morphological Tokenization for Multilingual MT
## Research Plan

**Last synced with thesis_proposal.md: 2026-09-14.** This document owns operational planning (compute, timeline, red-team disposition, logistics, decision log). For current hypotheses, evaluation design, methods, and citations, `thesis_proposal.md` is the source of truth, not this section. Update this note whenever a sync pass is done so staleness is visible at a glance.

**Working title:** "Vocabulary-Aligned Morphological Tokenization Across Morphological Types: A Typological Study" (see `thesis_proposal.md` for the current title, "When Does Morpheme-Aware Tokenization Help? A Predictive Framework Across Language Types")

**Target venue:** ARR (ACL/EMNLP/NAACL cycle) → expand to TACL if results support full typological coverage

**Origin:** Follow-up to "Bringing Mapudungun into the Modern MT Ecosystem" (AmericasNLP 2026). The Mapudungun paper established Morfessor-VC as a proof-of-concept; this paper generalizes it into a principled framework, tests it across morphological types and model architectures, and releases a reusable toolkit.

---

## Core Thesis & Research Questions

**See `thesis_proposal.md`** (Problem, Hypotheses, Motivation sections) for the current, canonical statement of the thesis, research questions, and the two formal hypotheses (vocabulary density vs. typological category as the driver of benefit; source/target asymmetry). Do not restate or re-derive that content here; if it changes, update the pointer date above rather than copying prose into this section.

---

## Methods

### Existing (from Mapudungun paper)
**Morfessor-VC** — Run Morfessor unsupervised morpheme segmentation; merge any resulting piece not in the pretrained model's vocabulary with its neighbor until it is in-vocabulary. Constrains segmentation to the existing vocab; helps source-side only.

### New Methods

**Vocab-biased BPE** — Modified BPE training that upweights merges producing tokens already present in the pretrained model's vocabulary. The merge score becomes:

```
score(merge) = freq(merge) × (1 + λ · 1[result ∈ pretrained_vocab])
```

where λ is a hyperparameter tuned per language (Optuna). No Morfessor dependency; purely BPE-based. Expected to help source-side tokenization without requiring a morphological analyzer.

**Vocab Expansion (Morfessor-VE)** — Run Morfessor; for morphemes *not* in the pretrained vocabulary, add them as new tokens and initialize their embeddings by averaging the embeddings of their constituent subword tokens. Fine-tune the full model including new embeddings. Addresses both source (better input segmentation) and target (model can generate morpheme tokens directly). WECHSEL is the closest prior work; Morfessor-VE differs in that the new vocabulary is morpheme-inventory-driven rather than cross-lingual-transfer-driven.

**Vocab Expansion + Vocab-biased BPE (Combined)** — Apply Vocab-biased BPE for segmentation, then expand vocabulary for any remaining out-of-vocab morphemes. Tests whether the two mechanisms compound.

### Baselines
- Standard BPE (pretrained model default) — primary baseline
- Plain Morfessor (no VC constraint) — isolates segmentation from vocab alignment
- WECHSEL — prior work on vocab adaptation for low-resource MT; tests representation/embedding transfer, not vocabulary alignment, so a WECHSEL win or loss is not directly interpreted as evidence for or against the tokenization hypothesis (this clarification now lives in thesis_proposal.md's footnote 10, since a reader needs it to correctly interpret results)
- **MorphBPE (Asgari et al., 2025, added 2026-09)** — constrains BPE merges to respect morpheme boundaries; the mirror-image method to Morfessor-VC (MorphBPE bends BPE to fit morphemes; Morfessor-VC bends morphemes to fit the existing BPE vocabulary). Core Tier 1 baseline in thesis_proposal.md; add here too.
- **Stretch-goal baselines (Tier 3, not core):** Optuna BPE (tuned BPE from Mapudungun paper, already implemented) and UnigramLM. Demoted from core baselines 2026-09 to match thesis_proposal.md's Tier system — both still valuable comparison points, just not required for a defensible Tier 1 result.
- **Character-level tokenization (added per advisor comment 2026-08-19)** — advisor's objection: standard BPE can represent *any* unseen sentence via its byte/char fallback, so it has universal coverage; none of the three new methods (Morfessor-VC, Vocab-biased BPE, Morfessor-VE) currently make that guarantee — Morfessor segmenters can hit truly novel word forms with no clean fallback, and Vocab-biased BPE's bias term doesn't change BPE's underlying coverage story either way. A character/byte-level tokenizer (e.g. ByT5-style) has coverage by construction and belongs in the grid as the other extreme of the granularity spectrum, both as a coverage sanity-check baseline and as a data point on the fertility/quality tradeoff curve. Recent literature to engage:
  - Edman et al. (TACL 2023), "Are Character-level Translations Worth the Wait? Comparing ByT5 and mT5 for Machine Translation" — direct precedent for a byte-level MT baseline at scale.
  - Libovický et al. (Findings of ACL 2022), "Why Don't People Use Character-level Machine Translation?" — catalogs why char-level underperforms in practice; useful for framing expectations.
  - Ndayegamiye, Roberts & Frey (FLAIRS 2026), "Systematic Analysis of Tokenization Properties in Low-Resource Polysynthetic NMT" (English–Cherokee) — directly relevant negative result: extreme compression/character-level tokenization did *not* improve BLEU for a polysynthetic low-resource pair in their grid. Cite as prior evidence the char-level baseline may lose, not as a reason to skip it.
  - **Ammon's paper — could not confirm which one the advisor meant.** Checked Ammon C. Shurtz's known publications (BYU MT lab): Kreyòl-MT (NAACL 2024), "Neuron-Level Language Tag Injection..." (zero-shot MT), and "When Scripts Diverge: Strengthening Low-Resource NMT Through Phonetic Cross-Lingual Transfer" (MRL 2025, aclanthology.org/2025.mrl-main.22) — the last is about phonetic/transliteration-based cross-script transfer for Khmer/Thai/Lao↔English, not character-level tokenization or OOV coverage, so it doesn't seem to be the intended reference. User flagged the intended paper may not be published yet — ask advisor directly for the citation before the related-work section is finalized.
- **OOV coverage note:** state explicitly in the Methods section (not just here) which methods have provable universal coverage (Standard BPE, Vocab-biased BPE — still BPE under the hood; character-level) vs. which don't out of the box (plain Morfessor, Morfessor-VC, Morfessor-VE — need a defined fallback for segments Morfessor can't handle, e.g. falling back to BPE or characters for that span) and document the fallback used for each.

---

## Phase 0: Single-Language Pilot (added per advisor comment 2026-08-19)

**Decided 2026-09: Mapudungun.** Resolved after further deliberation below; no longer an open item. thesis_proposal.md reflects this directly (Phase 0 pilot on Mapudungun, no PathSay mention in the core plan). The PathSay/Kaqchikel line of inquiry isn't abandoned, it's demoted to a Tier 3 stretch item ("an exploratory PathSay-language addition, one or a few") in thesis_proposal.md, consistent with the reasoning below that PathSay's value is lab-visibility and data reuse, not filling a typological gap Phase 0 needs.

Advisor's objection to launching the full 12-language grid directly: pick one language, get a clean result, *then* generalize — don't build the typological claim on the first pass. Two live candidates, considered as of 2026-08-19 before the decision above:

- **Mapudungun** — pro: existing AmericasNLP-paper infra (data pipeline, Morfessor-VC baseline, fertility analysis), a gold-standard morpheme segmentation in progress via the `mapudungun-syntax` collaboration, fastest path to a first clean result. Con: polysynthetic only — a Phase 0 result here says nothing yet about whether the framework generalizes to agglutinative/fusional types, and dialect-pooling in the AVENUE corpus (see "Connection to Mapudungun Paper" below) is a known wrinkle to disclose.
- **A PathSay language** (lab's active African-language data-gathering project) — pro: ties into an actively-growing lab data effort, likely more novel/visible. Con: no existing tokenization/eval pipeline to reuse yet, and neither the specific language, its morphological type, nor current data volume is confirmed.

**Decision needed before Phase 1 design is finalized.** Whichever is chosen becomes the single-language validation of the vocab-alignment hypothesis before committing to the full multi-language grid (see red-team note below on why that grid isn't ready to launch as-is regardless).

**Recommendation (2026-08-19): Mapudungun.** The point of a Phase 0 pilot is speed to a trustworthy first result before committing to the full grid — and on that axis Mapudungun wins decisively: data pipeline, eval harness, and a Morfessor-VC baseline already exist from the AmericasNLP paper, so Phase 0 is mostly a re-run with the two new methods added, not a build. PathSay's advantage (active lab data effort) doesn't actually help Phase 0's goal, because none of its candidate languages are polysynthetic (see below) — Phase 0 doesn't need typological breadth, Phase 1 does. The one real cost of picking Mapudungun is that it re-treads ground the AmericasNLP paper already touched; that's fine for a fast sanity check but shouldn't be oversold as new when Phase 0 results get written up.

A cleaner way to get both: keep Mapudungun for Phase 0, and **use Phase 1 to fold in a PathSay language as a partial swap-in for one of the existing agglutinative slots** (e.g. Nyanja or Kikongo in place of Swahili) — reuses PathSay's already-cleaned corpora, gives the lab-visibility win, and doesn't cost typological diversity since it's occupying the same agglutinative-Bantu category Swahili already covers. Concrete PathSay candidates and their status, from `/grphome/grp_mtlab/projects/pathsay-cleaning`:

| Code | Language | Family/type | In NLLB-200? | Data status |
|---|---|---|---|---|
| nso | Northern Sotho | Bantu, agglutinative | Yes (`nso_Latn`) | Active — trained SentencePiece model exists |
| nya | Chichewa/Nyanja | Bantu, agglutinative | Yes (`nya_Latn`) | Active — trained model exists |
| kon | Kikongo | Bantu, agglutinative | Yes (`kon_Latn`) | Active — trained model exists |
| orm | Oromo | Cushitic, agglutinative | Yes, as `gaz_Latn` | Active — trained model exists |
| pcm | Nigerian Pidgin | English creole, analytic | **No** | Active but off the table for this grid — not in NLLB-200/FLORES-200 |
| fon, kam, lua, luo | Fon, Kamba, Luba-Kasai, Luo | Gbe/Bantu/Nilotic | All in NLLB-200 | Past work — cleaned corpora exist but not the current focus |

None are polysynthetic, so none change the paper's headline typological claim either way — this swap is about leveraging existing cleaned data and lab collaboration, not about generalizing the polysynthetic result.

**Correction (2026-08-19, same day) — the above table understated PathSay.** It was built from what's actively running in `pathsay-cleaning`'s `lang_filter/` directory, which turns out to be a small, mostly-Bantu working subset. Samuel Bird (PathSay data lead) sent the actual current list — **~100 languages**, updated daily, with per-language row counts (row = a Church/scripture-domain parallel sentence, not FLORES-style general text — flag that domain as a caveat, same category of issue as Nunavut Hansard's legislative-domain caveat below). It includes real polysynthetic candidates the Bantu-only view missed entirely:

| Code | Language | Family | Type (needs WALS verification, not yet sourced) | Rows |
|---|---|---|---|---|
| apw | Western Apache | Athabaskan (Na-Dené) | Polysynthetic — same family as Navajo, textbook case | 7,945 |
| bla | Blackfoot | Algonquian | Polysynthetic | 1,151 |
| arp | Arapaho | Algonquian | Polysynthetic | 1,151 |
| cak | Kaqchikel | Mayan | Highly synthetic/agglutinative, ergative — richest-resourced synthetic language in the whole list | **168,846** |
| quc | K'iche' | Mayan | Highly synthetic/agglutinative, ergative | 73,434 |
| quh | South Bolivian Quechua | Quechuan | Agglutinative | 124,043 |
| cuk / kvn | Kuna / Border Kuna | Chibchan | Agglutinative-to-polysynthetic | 30,569 / 7,926 |
| + Achi, Aguateco, Chuj, Jakalteko, Mam, Poqomch'i, Tzutujil, Tzotzil, W. Kanjobal, Yucatec Maya, Zapotec | | Mayan / Oto-Manguean | Various, unverified | 1,879–42,065 |

**This reopens the Phase 0 recommendation above.** Western Apache/Blackfoot/Arapaho are genuinely polysynthetic (unlike anything in the earlier PathSay table) but tiny (1,151–7,945 rows, comparable to or worse than Mapudungun). Kaqchikel is the standout: 168,846 rows is an order of magnitude more data than Mapudungun has, in a highly-synthetic Mayan language — if it holds up as a legitimate polysynthetic-adjacent data point after typological verification, it could be a *stronger* Phase 0 candidate than Mapudungun on data-availability grounds, while Mapudungun still wins on existing-pipeline grounds. Recommend actually raising this with the advisor rather than deciding it here: the "Mapudungun vs. PathSay" framing he posed assumed PathSay meant the Bantu-only picture; Kaqchikel changes the tradeoff.

**Access note:** Sam's list updates daily; he recommended getting an account from Joseph (BYU MT lab) for live access rather than working from this one-time snapshot. Do that before finalizing any PathSay language choice — this table is already a point-in-time snapshot, not the live source of truth.

---

## Language Selection

Selection criteria: (1) in NLLB-200 coverage, (2) parallel data available (FLORES-200 minimum; ideally flores+ or other corpus), (3) typological diversity, (4) range of resource levels.

**How many NLLB-200 languages are morphologically complex? (advisor question, 2026-08-19)** Computed via `scripts/count_nllb_morphological_complexity.py` against the official 204-entry FLORES-200 language table, classified by WALS 20A where available (~35/196 languages directly) and family-level typological consensus otherwise (Comrie 1989; Velupillai 2012). SSOT: `results/nllb_morphology_stats.json`.

**Result: 158/196 languages (80.6%) are morphologically complex** (agglutinative + fusional + introflexive, i.e. everything but isolating/analytic — threshold is a variable in the script, not hardcoded). Not yet human-verified beyond the agent's own sourcing; spot-check ≥15 rows against wals.info before this goes in a paper draft.

**Dr. Ringger dug into the 80.6% claim (2026-08-19) — it doesn't hold up as a motivating statistic, but a better one does.** Checked the WALS Chapter 20A summary directly (Bickel & Nichols): ~75% of the world's languages rely exclusively on concatenative (non-isolating) morphology, areally concentrated exceptions aside (isolating languages cluster in the West African Sahel belt and Southeast Asia/the Pacific). NLLB-200's 80.6% is statistically indistinguishable from that global base rate — it's not telling you anything NLLB-specific, just restating that most human languages aren't isolating. Also: WALS 20A itself doesn't distinguish agglutinative from fusional (that's WALS 21A, not fetched for this classification), so most of the agglutinative/fusional calls behind the 80.6% are family-level typological consensus, not individually-sourced WALS datapoints — only ~35/196 languages have a direct WALS 20A value at all.

**The number that actually motivates the paper: extracted NLLB's own official per-language resource classification (Table 1, pp.12-15 of the NLLB paper PDF, arxiv.org/pdf/2207.04672) via pypdf and crossed it against the morphology classification** (`scripts/cross_nllb_resource_morphology.py`, merged into the same SSOT JSON). The parsed split (150 Low-resource / 54 High-resource) exactly matches the paper's own stated total — a real, if partial, validation of the extraction.

- **116 of 204 FLORES-200 languages (56.9%) are BOTH morphologically complex AND low-resource** by NLLB's own <1M-bitext threshold. This is the actual population a vocabulary-aligned tokenization method could plausibly help — and it's a materially different, better-grounded claim than "80% are complex," which conflates high- and low-resource languages and mostly reduces to the base rate above.
- **Counterintuitive finding, worth stating honestly rather than hiding:** complexity rate is actually slightly *higher* among NLLB's 54 high-resource languages (85.2%) than its 150 low-resource ones (77.3%). Complex morphology is not disproportionately concentrated in NLLB's low-resource tier — if anything mildly the opposite in this sample. So the "who cares" argument should rest on the raw count (116 languages) and the tokenization-quality mechanism, not on a claim that complexity and under-resourcedness are correlated within NLLB-200 — they aren't, in this data.
- **Recommendation: replace "80% of NLLB-200 languages are morphologically complex" with "116 of NLLB-200's 204 languages (57%) are both morphologically complex and low-resource"** as the motivating statistic in the paper's intro. It's smaller-sounding but it's the one that actually supports the argument, and it survives the base-rate objection the first number doesn't.

**Critical finding that changes the framing — flag to advisor:** Zero of the 204 FLORES-200 languages classify as polysynthetic, and more specifically **Mapudungun (arn), Nahuatl (nah), and Inuktitut (ikt) are not in the official FLORES-200/NLLB-200 language list at all** (confirmed independently against `data/flores200_languages_raw.tsv` — no matching codes). This contradicts this plan's Language Selection criterion #1 ("in NLLB-200 coverage") and the "who cares" framing above ("NLLB has 200 languages and the morphologically complex ones could be improved") for exactly the three polysynthetic languages the paper leans on hardest. It doesn't invalidate the project — it likely means the actual framing is stronger: these are polysynthetic languages NLLB was never pretrained on at all, so the paper's contribution is extending vocabulary-aligned tokenization to *NLLB-uncovered* polysynthetic languages via fine-tuning + vocab methods (consistent with how the Mapudungun/AmericasNLP paper already fine-tuned NLLB for arn↔es), not improving NLLB's existing coverage of polysynthetic languages. But every sentence in the intro that currently implies NLLB already covers polysynthetic languages needs rewriting, and the Language Selection table's inclusion criteria need a second bullet for "or added via fine-tuning, if not in NLLB-200 natively."

### Primary Languages (full grid)

| Language | Code | Morphological Type | Resource Level | Direction |
|---|---|---|---|---|
| Mapudungun | arn | Polysynthetic | Very low | arn↔es |
| Quechua | quy | Agglutinative *(reclassified 2026-08-19 — was mislabeled Polysynthetic; GLM-5.2 red-team, 6/8 samples)* | Low | quy↔es |
| Nahuatl | nah | Polysynthetic | Very low | nah↔es |
| Inuktitut | ikt | Polysynthetic | Low | ikt↔en |
| Turkish | tur | Agglutinative | High | tur↔en |
| Finnish | fin | Agglutinative | High | fin↔en |
| Swahili | swh | Agglutinative | Medium | swh↔en |
| Czech | ces | Fusional | High | ces↔en |
| Russian | rus | Fusional | High | rus↔en |

### Control Languages (analytic, expected null result)
| Language | Code | Type | Direction |
|---|---|---|---|
| Vietnamese | vie | Analytic/isolating | vie↔en |
| Mandarin | zho | Analytic/isolating | zho↔en |
| Yoruba | yor | Analytic/isolating (+ tonal) *(reclassified 2026-08-19 — was mislabeled Agglutinative; GLM-5.2 red-team, 3/8 samples)* | yor↔en |

**Rationale for controls:** If vocab-aligned tokenization doesn't help analytic languages (where morpheme = word), that validates the morphological complexity hypothesis and rules out a general fine-tuning artifact.

### Polysynthetic notes

**Verified 2026-08-19 against the official 204-entry FLORES-200 table (`data/flores200_languages_raw.tsv`): none of Mapudungun (arn), Nahuatl (nah), or Inuktitut (ikt) are in NLLB-200/FLORES-200 at all** — confirmed by direct grep, not memory. Quechua *is* in FLORES-200 (as `quy_Latn`, Ayacucho Quechua) but is agglutinative, not polysynthetic (see reclassification above). So the "in NLLB-200 coverage" selection criterion at the top of this section literally cannot be satisfied by any polysynthetic language — this is expected once you accept the reframing above (extending NLLB to languages it doesn't cover, not improving its existing coverage), but it means eval can't lean on FLORES-200 devtest for these three the way it can for every other language in the grid.

**Recommendation:** don't substitute Basque/Wolof for Nahuatl/Inuktitut — that would quietly shrink the polysynthetic category from 3 languages to 2 (Mapudungun, Quechua-if-still-counted... except Quechua's reclassified out too, so it'd leave only Mapudungun) right when the paper's central typological claim depends on having more than one polysynthetic data point. Instead, keep Nahuatl and Inuktitut in-scope using project-specific test sets in place of FLORES-200 devtest — Axolotl/OPUS corpora for Nahuatl, Nunavut Hansard for Inuktitut (already well-studied in NLP, per the original note) — and disclose the resulting test-set inconsistency explicitly in Limitations (three languages evaluated on non-FLORES test sets is a real, statable caveat, not a hidden one).

**Data audit (2026-08-19, verified via web search, not memory):**
- **Nahuatl — Axolotl corpus:** real, public, downloadable (UNAM, `corpus.unam.mx/axolotl`, also mirrored on HuggingFace as `somosnlp-hackathon-2022/Axolotl-Spanish-Nahuatl`). ~1.19M tokens, tens of thousands of Spanish-Nahuatl sentence pairs. Caveat directly relevant to a Morfessor-based method: the corpus spans **multiple Nahuatl varieties and orthographies** (dialectal + diachronic + orthographic variation, per the original LREC 2016 paper) — this could itself degrade morpheme segmentation quality independent of the tokenization method, so this needs disclosure alongside any Nahuatl result, and possibly a variety-filtering preprocessing step. Sufficient in size for Phase 1; **not a blocker.**
- **Inuktitut — Nunavut Hansard 3.0:** real, public, CC-BY-4.0 (NRC Digital Repository). ~1.3M aligned sentence pairs — described in its own paper as the largest parallel corpus of any polysynthetic or Indigenous-Americas language released to date. Domain is legislative proceedings (1999-2017), so there's a domain-mismatch caveat versus FLORES-200's Wikipedia-style register if it's ever compared cross-language, but as a standalone Inuktitut test set it's large and well-documented. **Not a blocker**, easily the best-resourced of the three non-FLORES polysynthetic languages.
- **Net result: both fallback datasets check out.** Neither Nahuatl nor Inuktitut needs replacing on data-availability grounds — the earlier "if data is insufficient, replace with Basque/Wolof" contingency in this doc's original draft doesn't trigger. Remaining work before Phase 1 is mechanical: download both, run them through the same cleaning/dedup pipeline as Mapudungun, and decide on a Nahuatl variety-filtering policy given the dialectal spread.

---

## Model Scope

### Primary model: NLLB-200
- Sizes: 600M, 3.3B (skip 1.3B to halve compute; run 1.3B only on winners)
- Already set up in Mapudungun pipeline; all scripts reusable

### Secondary model: mBART-50
- Different architecture (denoising pretraining vs. translation pretraining)
- Different vocabulary (250K SPM vs. NLLB's 256K)
- Run on subset: polysynthetic languages + 2 agglutinative (Turkish, Finnish) + 1 fusional (Czech)
- Tests whether results generalize across architectures

### Tertiary/Quaternary: mT5 and Aya Expanse 8B — **revised to Tier 3 stretch goals (decided 2026-09), superseding the 2026-08-19 "committed for ARR" call**
- Reason for the reversal: as thesis_proposal.md matured into the Tier 1/2/3 scope system, architecture replication got reassessed against actual thesis time constraints. mBART-50 alone (Tier 2) is sufficient to test "results generalize beyond one architecture"; mT5 and Aya add real value but aren't required for a defensible thesis (Tier 1).
- mT5: different pretraining objective (span-corruption T5) and a genuinely different SPM vocabulary from both NLLB and mBART. If pursued, run on the same reduced subset as mBART-50 (polysynthetic + 2 agglutinative + 1 fusional).
- Aya Expanse 8B: tests the methods in an instruction-tuned LLM context. If pursued, run on the most reduced subset: Phase 0's winning language + 1 agglutinative + 1 analytic control, method-limited to Standard BPE vs. the single best-performing new method.
- Both are explicitly the first things cut if the timeline slips, and the first things picked up if Tier 1 finishes with time to spare.

---

## Experimental Design

**Predates the Tier system, same caveat as Compute Estimate/Timeline below.** Phases 1-5 here describe the original full-grid design; thesis_proposal.md's Tier 1 is the actual required floor (narrower — see Compute Estimate note). Treat this section as the maximal/Tier-2-3 version of the plan, not the committed one.

### Phase 1: NLLB 600M, all languages, all methods
Full grid: 12 languages × 2 directions × 7 methods (Standard BPE, Morfessor, Morfessor-VC, Vocab-biased BPE, Vocab Expansion, WECHSEL, + backtranslation for polysynthetic languages) = ~148 training runs at 600M. This is the paper's core result table.

**Backtranslation scope (revised 2026-08-19 per GLM-5.2 red-team, item 2/8-unanimous confound):** BT is now **excluded from the core Phase 1 cross-language-type comparison table** — pre-registered here, before any runs launch, so there's no post-hoc temptation to include it if it happens to help the headline result. It still runs, but only as a separate, clearly-labeled polysynthetic-only ablation (arn, quy*, nah, ikt — *quy is agglutinative post-reclassification, so scope this ablation to genuinely polysynthetic languages: arn, nah, ikt) reported in its own subsection, testing whether BT + Vocab Expansion compounds for es→arn specifically. Any claim comparing polysynthetic vs. other typological groups in the main results table uses non-BT conditions only.

### Phase 2: NLLB 3.3B, winner + Standard BPE per language
Run Standard BPE and the best-performing new method per language at 3.3B to confirm scale effects. ~24 additional runs.

### Phase 3: mBART-50 replication
Subset of languages at 600M equivalent, all methods. ~48 runs. Tests architecture generalization.

### Phase 4: Predictive analysis
**Synced 2026-09 to thesis_proposal.md's more developed design**, which supersedes the description below.
- Vocabulary density computed two ways: gold-derived (fraction of gold/reference morphemes already in the pretrained vocab) for languages with gold segmentation (Turkish, Finnish, Czech, Russian, possibly Quechua), and an unsupervised estimate (Morfessor-output-vs-vocab overlap, no gold annotation needed) for the rest — reported both ways to quantify what's lost by removing the gold-annotation requirement, since the unsupervised estimate is what actually scales to the full 116-language NLLB population.
- Density-vs-typology test: mixed-effects model, density as fixed effect, language as random intercept, fit on within-language density-benefit pairs per morpheme-frequency bucket (not one point per language, avoids conflating typology with per-language idiosyncrasy at n≈12). Covariates: raw pretraining-corpus token count and final tokenizer vocabulary size, to rule out exposure and vocab-size confounds respectively.
- Predictive framework validated via leave-one-language-out: fit on all languages but one, predict the held-out language's best method, report top-1 accuracy against a 25% chance baseline (4 candidate methods) and calibration (predicted-vs-observed chrF++ gain correlation, r > 0.5 pre-specified as "meaningfully predictive"). Quechua and/or Swahili held out entirely as a cold genus-level test (Quechuan; Bantu specifically, distinct from the leave-one-out pass).
- If this holds up, the toolkit can recommend a method without training — this is now treated as a primary contribution (see Open Questions below), not just a post-hoc chart.

### Phase 5: Ablations
- Vocab Expansion: embedding initialization strategy (average vs. nearest-neighbor vs. random)
- Vocab-biased BPE: λ sensitivity
- Morfessor hyperparameter sensitivity across languages

---

## Toolkit (the "framing" contribution)

**Package name TBD** (e.g., `morphotok`, `vamt`, `morphalign`)

**Inputs:** language code, pretrained model name/path, monolingual corpus (optional)
**Outputs:** tokenized training data using the recommended method; optionally, an extended tokenizer + embedding initialization

**API sketch:**
```python
from morphotok import MorphoTokenizer

tok = MorphoTokenizer(
    lang="tur",
    model="facebook/nllb-200-distilled-600M",
    method="auto"  # or "morfessor-vc", "vocab-expansion", "vocab-biased-bpe"
)
tokenized = tok.tokenize(corpus)
```

**"auto" mode:** uses Phase 4's predictive analysis to recommend a method based on vocab density without training. Fall back to Morfessor-VC as safe default.

**Release:** pip package + HuggingFace integration (HF tokenizer-compatible). Documented with language-specific examples for all 12 paper languages.

---

## Evaluation

**Revised 2026-09 to match thesis_proposal.md's AmanaMT-informed approach** (superseding the generic "COMET-23 (CK23, not COMET-22)" framing below, which predates that project's completed findings).

### Primary metric
chrF++ (sacrebleu) — consistent with Mapudungun paper and standard in low-resource MT. Character-only chrF is additionally reported alongside chrF++, to check whether results are driven by chrF++'s word-n-gram component in typologically uneven ways (chrF++ behaves closer to character-only for long-word polysynthetic languages, closer to word-level matching for short-word analytic ones — GLM-5.2 red-team item 5).

### Secondary metrics
- **Per direction/morphology cell, CometKiwi-23 or MetricX-24**, as recommended by AmanaMT (Thompson, 2026)'s typological metric-selection guide, rather than one fixed COMET checkpoint across all language/direction pairs. AmanaMT found translation direction (not morphological type) is the dominant predictor of which metric is reliable, and that COMET-22/23-family reliability itself varies by direction, which is a genuine confound for this project's own direction-asymmetry hypothesis (Hypothesis 2 in thesis_proposal.md): an apparent lack of target-side tokenization benefit could partly reflect the metric being less reliable evaluating into a given language, not a real null effect. Cross-check results against the direction-appropriate metric where possible. **For the six languages overlapping with AmanaMT's own benchmark (Turkish, Finnish, Swahili, Czech, Russian, Mandarin), use that study's actual per-language measured reliability rather than the general guide** — free, since the data already exists.
- BLEU for comparison with prior work.
- COMET-22 is still avoided as a fixed default for OOD languages not in its training data (Mapudungun, Nahuatl) for the same reason as before; the per-cell CometKiwi-23/MetricX-24 approach supersedes needing a single blanket "use COMET-23" rule.

### Metric disagreement risk
Thompson et al. (subchar-mt, forthcoming) find that morpheme segmentation improves chrF++/BLEU but COMET flips the ranking (baseline > morphemes) for Chinese MT. This may replicate here. Paper A should report both metrics and address the disagreement directly if it appears — this is prior work to cite, not a gap to fill.

### Human evaluation
**Revised 2026-09: back in scope, split minimal (Tier 1) / extensive (Tier 3 stretch)**, superseding the earlier "cut entirely" call. The fix for "don't know speakers of all 12 languages" isn't dropping human eval, it's scoping it by typological category (one language per category: polysynthetic, agglutinative, fusional, analytic) rather than by individual language — this keeps the design symmetric at the level the paper actually argues at (typology), rather than either attempting all 12 (infeasible) or singling out Mapudungun alone (asymmetric attention on one of twelve languages, which risks reading as a Mapudungun paper with 11 languages attached rather than a genuine typological study).
- **Minimal (Tier 1):** Mapudungun (existing AmericasNLP annotator) fills the polysynthetic slot. Agglutinative/fusional/analytic slots need an actual annotator audit before committing — Turkish/Finnish plausible for agglutinative, Czech/Russian for fusional, Vietnamese/Mandarin/Yoruba for analytic, all worth checking against BYU's language population rather than assumed. ~30-50 sentences per language, preference ranking between Standard BPE and best new method, MQM-lite error typology.
- **Extensive (Tier 3 stretch):** broader language coverage, multiple annotators with measured inter-annotator agreement — this is what actually answers the Mapudungun paper's own stated need for a stronger human-eval design (its pilot had one annotator, no IAA, 20 sentences for linguistic annotation).
- Goal: validate whether the chrF++ ceiling finding from the Mapudungun paper replicates (its own pilot found a 3-point chrF++ gap produced no consistent human preference — cited directly in thesis_proposal.md's footnote 4 as a caveat on the ≥1 chrF++ "help" threshold).
- **AmanaMT synergy:** if the analytic-category slot is evaluated in the en→X direction (already tested for Vietnamese/Mandarin/Yoruba per the language table), the resulting judgments contribute a real data point toward AmanaMT (Thompson, 2026)'s documented empty en→X isolating cell — free secondary value from already-planned work.

### Statistical testing
- Bootstrap resampling (n=1000) for significance
- **Benjamini-Hochberg (FDR control), scoped within-language, not Bonferroni (revised 2026-09).** Plain Bonferroni across the full 12-language × multi-method grid is too conservative given the power problem already flagged for the lowest-resource languages (Mapudungun, Nahuatl, Inuktitut) — stacking a maximally conservative correction on top of already-marginal power risks making it statistically impossible to detect real effects exactly where the thesis needs power most. BH is standard for this kind of pre-registered (not exploratory) comparison set, and scoping the correction within-language (across the handful of methods compared for that language) rather than globally avoids over-correcting for comparisons that aren't actually the relevant multiplicity.
- Report effect sizes, not just p-values

---

## Analysis

### Why it works (or doesn't)
- Fertility analysis across all languages (already implemented for Mapudungun)
- Vocabulary coverage: % of morphemes in pretrained vocab, before and after expansion
- Attention entropy: does morpheme segmentation concentrate attention on more meaningful spans?

### The direction asymmetry
- Test whether Vocab Expansion resolves the polysynthetic-target problem
- If not, characterize what remains: is it decoding (beam search over morpheme tokens)? Sparse target-side training signal?

### Morphological type as predictor
- Does mean morphological complexity score (e.g., morpheme-per-word ratio) predict effect size?
- Cluster languages by method performance, see if clusters align with typological categories

### Tokenization × resource level
Stratify Phase 1 results by resource level (very low / low / medium / high). Does morpheme-aware tokenization help most at very-low resource, and fade at higher resource levels? subchar-mt (Thompson et al., forthcoming) finds morpheme representations peak at 50–100 training examples and degrade at higher data volumes — test whether this pattern holds across morphological types. This analysis is free from Phase 1 data and previews Paper B naturally.

### Failure modes
- For languages where no method helps: why? Corpus size ceiling? Domain mismatch? NLLB pretraining coverage?

### Cheap additions from Mapudungun paper limitations
- **BOUQuET eval:** Mapudungun is already in BOUQuET (omnilingual2026). Run all Mapudungun models on BOUQuET at inference time — no new training, directly addresses a stated future-work item from the AmericasNLP paper.
- **Source-copy failure mode:** Check whether the es→arn source-copy failure observed in Mapudungun (model reproduces Spanish source verbatim in Mapudungun orthography) appears in other polysynthetic language pairs. Purely analytical — no new experiments beyond inspecting outputs.

---

## Compute Estimate

**Stale as of 2026-09 — needs re-scoping against the Tier system before being trusted.** This table was built around the old Phase 1-5 structure (full 12-language × 7-method grid, mBART as Phase 3, mT5/Aya not yet in the picture). thesis_proposal.md's Tier 1 (the "defensible thesis" floor) is narrower — Phase 0 + core grid on Morfessor-VC/Morfessor-VE/baselines across 12 languages, NLLB-200 only, no mBART/mT5/Aya, no Optuna BPE/UnigramLM. The table below likely overstates required compute for Tier 1 and doesn't cleanly map costs onto Tier 2/3 additions. Recalculate once Phase 0's actual per-run wall-clock time is known (same dependency already noted for the undertraining pilot decision below).

| Phase | Runs | GPU-hours (est.) |
|---|---|---|
| Phase 1 (NLLB 600M, all) | 144 | ~720 |
| Phase 2 (NLLB 3.3B, winners) | 24 | ~480 |
| Phase 3 (mBART replication) | 48 | ~240 |
| Phase 4 (predictive analysis) | 0 training | ~10 |
| Phase 5 (ablations) | ~40 | ~200 |
| **Total** | **~256 training runs** | **~1,650 GPU-hours** |

GPU-hours estimated at ~5h/run for 600M, ~20h/run for 3.3B on A100. These are rough; Mapudungun training times should calibrate actual estimates.

**SLURM strategy:**
- Phase 1 is fully parallelizable — queue all 144 jobs simultaneously
- Use `HF_HUB_OFFLINE=1` and pre-download all model weights
- Cache Morfessor and tokenizer outputs; don't recompute across methods for same language

---

## Data Pipeline

For each language:
1. Download parallel corpus (FLORES-200 as minimum; supplement with OPUS, AmericasNLP corpora, Nunavut Hansard, etc.)
2. Deduplicate and clean (reuse Mapudungun pipeline where possible)
3. Run Morfessor to get morpheme inventory
4. Compute NLLB vocab coverage (Phase 4 analysis)
5. Generate tokenized training sets for each method
6. Train models
7. Evaluate on FLORES-200 devtest (standardized across all languages)

**FLORES-200** as the universal test set ensures comparability across languages and with prior work. Where available, supplement with language-specific test sets.

---

## Related Work to Engage

**Rewritten 2026-09 to match thesis_proposal.md's actual reference list**, which has diverged substantially from this older list as the project matured. Dropped from the old list: Passban et al. (2018) and Sennrich & Haddow (2016) (generic survey citations — the per-paper hypothesis-testing framing in thesis_proposal.md's Prior Work section does this job better now); Gerz et al. (2018) and Cotterell & Heigold (2017) (typology-in-NLP framing — now handled by the AmanaMT Ridge/WALS negative-result citation below, which makes the same point with a directly relevant empirical result rather than a general citation); Fan et al. (2022) for NLLB (the correct citation is "NLLB Team, et al. 2022," not "Fan et al." — Angela Fan is a co-author but the paper's own convention is NLLB Team).

- **Morfessor**: Virpioja et al. (2013) — morphological segmentation tooling
- **WECHSEL**: Minixhofer et al. (2022) — vocabulary transfer for cross-lingual adaptation
- **NLLB**: NLLB Team, et al. (2022) — the base model
- **mBART**: Liu et al. (2020) — secondary base model
- **BPE**: Sennrich et al. (2016a); Gage (1994)
- **Backtranslation**: Sennrich et al. (2016b) — distinct paper from the BPE one, same authors/year, disambiguated as 2016a/2016b in thesis_proposal.md
- **UnigramLM**: Kudo (2018)
- **MorphBPE**: Asgari et al. (2025) — morpheme-boundary-constrained BPE, the mirror-image method to Morfessor-VC
- **Fertility/tokenization quality**: Rust et al. (2021) — the correct source for the fertility-degrades-quality claim (not Ahia et al. 2023, which is about tokenization cost/pricing inequity, a related but distinct claim, caught and fixed 2026-08-21)
- **Morphological analyzer alternative**: Nzeyimana (2024) — Kinyarwanda, explicit morphological analyzer/synthesizer replacing tokenization entirely
- **Polysynthetic MT**: Mager et al. (2022, four-language BPE-vs-Morfessor case study); Mager et al. (2023 survey); Ebrahimi et al. (2023 AmericasNLP shared task)
- **Linguistic diversity in NLP**: Joshi et al. (2020)
- **Typology in NLP / metric reliability**: Thompson (2026, AmanaMT) — direction dominates over morphological type as a predictor of metric reliability; explicitly names polysynthetic languages (Inuktitut) as unstudied future work, which this project directly answers
- **Speaker population data**: Eberhard et al. (2026, Ethnologue)

---

## Timeline (target: ARR October 2026 cycle)

**Also stale as of 2026-09, and now overdue, not just structurally outdated.** Today is 2026-09-15 — the table below has "Phase 2 training complete (NLLB 3.3B, winners)" dated exactly today, while Phase 0 hasn't started. Every date below through 2026-09-28 has already passed or is passing with none of the underlying work done. This needs a full re-date, not just a re-scoping, before it's useful for anything beyond historical record. Milestones should be re-sequenced around Tier 1 first, Tier 2/3 only if time allows, rather than the full grid as originally laid out.

| Milestone | Target Date |
|---|---|
| subchar-mt + needle-in-a-haystack submitted (prerequisite) | 2026-08-03 |
| Data pipeline for all 12 languages | 2026-08-15 |
| Phase 1 training complete (NLLB 600M, all methods) | 2026-09-01 |
| Phase 4 predictive analysis | 2026-09-05 |
| Phase 2 training complete (NLLB 3.3B, winners) | 2026-09-15 |
| Phase 3 training complete (mBART replication) | 2026-09-20 |
| Phase 5 ablations | 2026-09-28 |
| Toolkit packaged and documented | 2026-10-02 |
| Paper draft complete | 2026-10-08 |
| Internal review + revision | 2026-10-13 |
| **ARR submission** | **2026-10-15** |

If results across all 12 languages + 2 models are strong, expand to full TACL treatment (add language varieties, deeper analysis, extended related work) for a TACL submission in early 2027.

---

## GLM-5.2 Exploration Phase (pre-experimental reasoning)

Logged 2026-08-04. Before/alongside Phase 1 data work, use the self-hosted
GLM-5.2 server (744B MoE, vLLM on BYU m13h) for open-ended reasoning on the
research questions above — not a substitute for the actual experiments, just
a check on whether the model's own reasoning surfaces anything non-obvious
before committing compute to the full grid.

**Methodology** (human-out-of-the-loop mode): no automatic verifier exists for
these questions the way a proof checker exists for math, so substitute
self-consistency (N=8–16 parallel samples per question, look for convergence
across samples rather than trusting any single generation) plus, where
possible, grounding in real data already in hand (Mapudungun fertility
analysis, existing Morfessor outputs). Full reasoning traces logged and
audited after the run, not checkpointed live.

### Starter questions (ready to run once server is up)

1. Is the Mapudungun direction asymmetry (Morfessor-VC helps source-side, not
   target-side) a general property of polysynthetic-target translation, or an
   artifact specific to Mapudungun's morphology/corpus? What mechanism would
   predict which polysynthetic languages show it?
2. Given the three candidate methods (Morfessor-VC, Vocab-biased BPE, Vocab
   Expansion), reason through which should win for each morphological type
   (polysynthetic, agglutinative, fusional) from first principles, before
   any training — then compare against Phase 1 results once they exist as a
   sanity check on whether the model's predictions were defensible.
3. Can NLLB pretraining vocabulary density over a language's morpheme
   inventory (RQ5) plausibly predict method choice without training? Reason
   about what mechanism would make this true or false, and what a
   counterexample would look like.
4. For analytic control languages (Vietnamese, Mandarin): argue the case for
   *why* vocab-aligned tokenization should show a null result there — is
   "morpheme = word" actually sufficient reasoning, or is there a way the
   control could still show an effect for a confounded reason (e.g.
   fine-tuning artifact) that the current design wouldn't catch?
5. Toolkit "auto" mode: given only a language code and a pretrained model,
   sketch what a decision procedure (not requiring training) would look like
   for recommending Morfessor-VC vs. Vocab Expansion vs. Vocab-biased BPE —
   does this converge with the Phase 4 predictive-analysis plan already in
   the doc, or suggest a different signal?

---

## Known Design Issues to Fix Before Phase 1 (GLM-5.2 red-team synthesis, 2026-08-19)

8-sample self-consistency red-team of the Phase 1 grid design (`/home/it238/nobackup/autodelete/glm52/logs/red_team_synthesis/morphology-mt-phase1_synthesis.md`) converged on issues that could produce a false-positive confirmation of the core hypothesis. All are cheap to fix before compute is spent:

- **Corpus-size confound (unanimous, 8/8 samples):** polysynthetic languages in the grid are almost all low-resource while analytic/fusional languages are higher-resource, so a clean polysynthetic→analytic improvement gradient could just be data volume, not morphology. Fix: subsample high-resource languages to match the lowest-resource polysynthetic language's sentence count, or add a low-resource isolating-language control.
- **Backtranslation confound (7/8):** BT is applied only to polysynthetic languages, making cross-type comparisons uninterpretable if BT conditions win. Fix: drop BT from Phase 1's core grid, or pre-register that BT runs are excluded from cross-type comparisons and reported as a polysynthetic-only ablation.
- **Quechua misclassification (6/8):** Quechua is agglutinative, not polysynthetic — factual error in the language table above, zero-cost fix.
- **Yoruba misclassification (3/8):** Yoruba is analytic/isolating, not agglutinative — also zero-cost, reclassify as a control-group language.
- **COMET-23 coverage gap (6/8):** XLM-R (COMET-23's backbone) likely has minimal-to-no training coverage for Mapudungun, Nahuatl, Inuktitut — exactly the languages carrying the primary claim. Verify coverage; add chrF (character-only, not chrF++) or BLEU as a metric that doesn't depend on it.
- **chrF++ differential sensitivity (5/8):** chrF++ mixes char and word n-grams; for long-word polysynthetic languages it degenerates to effectively character-only, while for short-word analytic languages it behaves more like word-level matching — a metric that isn't comparable across the typological groups being compared. Fix: report chrF (char-only) alongside chrF++ and decompose where needed.
- **Phylogenetic pseudo-replication (2/8):** Czech+Russian are both Slavic, Vietnamese+Mandarin share areal features — with n=2 per category, claims should be scoped ("Slavic fusional," not "fusional") rather than generalized to the whole typological class.
- **Morfessor degenerate-segmentation risk (2-3/8):** low monolingual data for Mapudungun/Nahuatl could make Morfessor segmentation degenerate before any training even starts. Fix: manually inspect segmentations for all languages (~1hr CPU) before launching Phase 1.
- Single-sample flags worth a look but not yet convergent: WECHSEL may be a category error in the comparison (it changes embedding init, not tokenization); 5 GPU-hrs/run may undertrain larger-vocab methods, systematically favoring Morfessor; vocabulary-size itself is a confound between Morfessor and BPE outputs at "the same" merge count.

**Net effect on this plan:** corroborates the advisor's Phase 0 instinct above — a single-language pilot is a natural place to shake out the Quechua/Yoruba reclassification and Morfessor-inspection fixes cheaply before they get baked into a 144-run grid.

**Full disposition of every red-team item (2026-08-19 — user asked to address all of them, not just the blocking three):**

1. **Corpus-size confound → fixed by design.** Add a matched-size condition to Phase 1: for every non-Phase-0-tier language, run an additional training condition subsampled to the sentence count of the lowest-resource polysynthetic language in the grid (Mapudungun or Nahuatl, whichever ends up smaller). Report both the natural-size and matched-size results side by side in the main table — if the effect survives at matched size, corpus size is ruled out directly rather than argued around. Adds roughly one subsampled run per non-low-resource language/method cell; scope this against the Compute Estimate table once Phase 0 timing is known.
2. **Backtranslation confound → fixed above** (moved to its own pre-registered ablation, excluded from cross-type claims).
3. **Quechua / Yoruba misclassification → fixed above** (language table corrected).
4. **COMET-23 coverage gap → add a verification step.** Before Phase 1 launches, check XLM-R's (COMET-23's backbone) training-language list for arn/nah/ikt/quy coverage — this is a lookup, not an experiment. Whatever the result, report chrF (character-only, sacrebleu default word_order, distinct from chrF++) as a metric that doesn't depend on COMET-23's backbone coverage at all, alongside chrF++ and COMET-23, for every language.
5. **chrF++ differential sensitivity across typology → add the decomposition.** Report character-only chrF and word-only chrF components separately (not just the blended chrF++) in an appendix table, so a reader can see whether the headline effect is driven by the character component (which would suggest it behaves similarly across polysynthetic and analytic languages) or diverges structurally.
6. **Phylogenetic pseudo-replication (Czech/Russian both Slavic; Vietnamese/Mandarin areal) → fixed by scoping claims, not by adding languages.** Any generalization in the paper's prose uses "Slavic fusional" / the specific language pair, not "fusional languages in general," until n≥3 unrelated languages per category exist. If budget allows after Phase 1, the single highest-value addition would be one more, unrelated fusional language (e.g. Finnish is already agglutinative — consider Georgian or Icelandic as a non-Slavic fusional add for Phase 2/TACL, not Phase 1).
7. **Morfessor degenerate-segmentation risk → pre-flight check done for Mapudungun (2026-08-19), still open for the other 11 languages.** Loaded the existing trained model (`/nobackup/archive/usr/it238/mapudungun_morfessor_model.bin`, from the AmericasNLP pipeline) and ran `viterbi_segment` on 40 real word forms sampled from the raw AVENUE transcription corpus (`projects/mapudungun/mapudungun-corpus`, not synthetic test strings). Result: **1/40 (2.5%) came back near-character-level** (`fep` → `fe+p`, a 3-letter word — expected noise on very short forms, not a systematic failure); the other 39 produced linguistically plausible-looking morpheme splits (`amuleayin` → `amule+ay+in`, `witranalwue` → `witran+al+wue`, `ngillaentukelaymi` → `ngilla+en+tukelaymi`). **Verdict: Mapudungun's existing Morfessor model is not degenerate** — Phase 0 can proceed on the current segmentation without re-tuning. One artifact found and worth fixing regardless: one sample word (`nmlchnfmcm2x0155nfmcm00SPAno`) was a leaked transcription-ID token, not real text — the corpus-cleaning step needs a tighter filter for the AVENUE `.trl` transcript-ID format before this corpus is reused as-is. The other 11 languages (Nahuatl, Inuktitut, Quechua, Turkish, Finnish, Swahili, Czech, Russian, Vietnamese, Mandarin, Yoruba) haven't been checked yet — none have a trained Morfessor model to inspect until their data pipelines exist, so this stays open until each language's corpus lands.
   - Housekeeping note: the actual training-ready corpus files under `/nobackup/autodelete/usr/it238/mapudungun/data-processed/` (the `morfessor`, `cleaned`, `morfessor_vc` subdirs referenced by this plan) have **already been auto-deleted from `nobackup/autodelete` scratch** (BYU's autodelete policy) — only trained model checkpoints and predictions survive in `nobackup/archive`. The raw AVENUE transcripts in the git-tracked `mapudungun-corpus` repo are safe, but the *processed/cleaned/tokenized* intermediate files need regenerating from scratch before Phase 0 training can actually run. Worth confirming the corpus-prep scripts that produced them still exist somewhere reproducible (not just their output).
8. **WECHSEL category-error concern → clarify in the paper, not the design.** WECHSEL changes embedding initialization, not tokenization/segmentation — keep it in the baseline table but add one sentence in Methods explicitly stating it tests a different axis (representation transfer, not vocabulary alignment), so a WECHSEL win doesn't get mistakenly read as evidence against or for the tokenization hypothesis.
9. **Undertraining at ~5 GPU-hours/run → add a pilot.** Before committing the full 720 GPU-hours, run a 3-config pilot (Standard BPE, Morfessor-VC, Vocab Expansion) at 3x the planned budget (~15 GPU-hours each, ~45 GPU-hours total) on the Phase 0 language, and check whether method rankings match the 5-hour budget's rankings. If they don't, increase the Phase 1 per-run budget before launching — this is the one red-team fix that costs meaningful extra compute, but it's cheap insurance against a systematic bias toward smaller-vocabulary methods.
10. **Vocabulary-size confound (Morfessor vs. BPE at "the same" merge count) → report vocab size as a covariate.** For every trained tokenizer (all methods, all languages), log final vocabulary size alongside the results table, so a reviewer can check whether performance differences track vocabulary size rather than the segmentation strategy itself. No new runs needed, just an additional logged statistic per existing run.

Items 1 and 9 are the only two that cost additional compute; everything else is design/reporting discipline. Recommend resolving 3, 4 (the lookup), 7, and 8 during Phase 0 since they're free and fast; decide on 1 and 9's added compute once the Phase 0 language's actual corpus size and per-run wall-clock time are known.

---

## Open Questions / Decisions Needed

- [ ] **Final language list (Nahuatl/Inuktitut):** recommendation given above under "Polysynthetic notes" — keep both, evaluate on project-specific test sets (Axolotl/OPUS, Nunavut Hansard) since neither is in FLORES-200, disclose as a Limitations item. Still open: actually auditing corpus size/quality for each before Phase 1 commits compute.
- [ ] **Toolkit name — recommendation: `morphotok`.** Short, descriptive without being cryptic, reads naturally as a pip package name (`pip install morphotok`, `from morphotok import MorphoTokenizer` — matches the API sketch already in this doc). Check PyPI/GitHub namespace availability before committing; `vamt` and `morphalign` are the fallback candidates already listed above if it's taken.
- [x] Whether to include mT5 or Aya as a third model — **revised 2026-09: both are Tier 3 stretch goals**, superseding the 2026-08-19 "both, committed for ARR" decision. See Model Scope above for reasoning and the reduced-subset scoping if pursued.
- [ ] **Human evaluation languages and annotators (revised 2026-09):** minimal eval needs one confirmed annotator per typological category (Mapudungun confirmed for polysynthetic; agglutinative/fusional/analytic slots still need an actual audit, not just the assumption that "Turkish and Finnish are easy to staff" — verify against BYU's actual language population before committing).
- [ ] **Whether Phase 4 (predictive analysis) is a primary contribution or stays as analysis — recommendation: elevate it to primary.** The toolkit's "auto" mode literally depends on this signal existing, so if it stays a descriptive analysis, the toolkit's headline feature is resting on an untested correlation. Make it rigorous instead: hold out a subset of Phase 1 languages, fit the vocab-density predictor on the rest, and report leave-one-language-out recommendation accuracy (does the predicted winning method match the actually-observed winner?) as its own result, not just a post-hoc chart. That's a real, falsifiable claim worth a subsection of its own, and it's the thing that makes "auto mode" defensible as a contribution rather than a convenience feature.
- [ ] Co-authorship: currently Isaac + advisor confirmed; Brandon and others still TBD. Likely add Rogers if the mapudungun-syntax gold-standard section (below) is used, since that's their data contribution.
- [ ] **Whether to open-source training scripts as part of the toolkit or separately — recommendation: one repo, two surfaces.** Keep the `morphotok` package (the clean tokenization/auto-mode API) as the polished public-facing piece, but put paper-reproduction scripts (data pipeline, training configs, eval harness) in an `experiments/` subdirectory of the *same* repo rather than a second repo. Splitting them adds sync overhead (two READMEs, two issue trackers, version drift between "the toolkit used in the paper" and "the toolkit people pip-install") for no real benefit at this project's size — ARR/TACL reviewers and future users both just want one link.

---

## Connection to mapudungun-syntax Project

The Rogers collaboration (`projects/mapudungun/mapudungun-syntax`) produces a gold-standard morpheme segmentation of Mapudungun verb words. This feeds morphology-mt in two ways:

1. **Linguistic validity section:** Evaluate Morfessor-VC's boundary placement against the gold standard (precision/recall on morpheme boundaries). This shows the segmentation is linguistically meaningful, not just MT-useful — a stronger "why it works" argument.
2. **Dialect-separated data point:** The morphotactics project separates the 3 AVENUE dialects (Lafkenche, Nguluche, Pewenche). If dialect-level segmentation accuracy differs, that informs how to interpret Mapudungun as a data point in morphology-mt.

The gold standard is built by the morphotactics project on its own timeline; morphology-mt uses it as a downstream consumer. Coordinate timing so the gold set exists before morphology-mt's analysis section is written.

---

## Future Paper: Speech Translation for Mapudungun

**Working title:** "Cascaded and End-to-End Speech Translation for Mapudungun"

**Relationship:** Paper 3 in the Mapudungun sequence (after AmericasNLP MT paper and morphology-mt). AVENUE corpus has audio — opens ASR → Morfessor-VC → NLLB pipeline or end-to-end approach.

**Novel angle:** Morfessor-VC applied at the ASR decoding stage — constraining the subword lattice to NLLB vocab. Untested and potentially impactful.

**Venue:** Interspeech or ACL.

**Status:** Idea only — no experiments planned yet. Lower priority than morphology-mt Paper A and Paper B.

---

## Connection to Mapudungun Paper

This paper cites the Mapudungun paper as the origin case study. Morfessor-VC results from that paper are included as the starting point (with permission — it's your own prior work). The new paper does not re-run Mapudungun from scratch; it uses the existing results and extends the method.

The Mapudungun paper's limitation "Morfessor-VC is language-agnostic and applies to other polysynthetic languages" is the explicit motivation for this paper.

**Dialect note (sharpened 2026-09 against the actual corpus, not the paper's vaguer framing):** The AVENUE corpus is not evenly pooled across its 3 dialect communities. Verified directly against the raw transcription files (`mapudungun-corpus/TRANSCRIPTION`): by transcription volume, ~97% comes from a single source (31,110 of 32,202 transcription lines; 328 of 344 files), with the other two sources combined under 3.5%. "Dialect-pooled" undersells it — Mapudungun-as-tested is much closer to one dialect/speaker community's usage than a balanced 3-way sample. This is now stated precisely in thesis_proposal.md's footnote 8. Considered and rejected: dropping the two minority sources entirely — at this scale they're not a real confound, but they are real training data in an already very-low-resource setting, and removing them would require re-deriving splits and retraining the existing baseline models that Phase 0 is specifically trying to reuse for speed. Disclosure, not removal, is the right fix.

**Scope separation from subchar-mt:** Both papers study morpheme-based tokenization for MT. The separation is clean by design: Chinese is an analytic (control) language in Paper A, so no experiments overlap. The metric disagreement finding from subchar-mt is prior work to cite, not a result to reproduce.

---

## Future Paper: Data Strategies for Morphologically Complex Low-Resource MT

**Working title:** "Data Strategies for Morphologically Complex Low-Resource MT: Augmentation and Typological Transfer"

**Relationship to this paper:** Companion paper. Paper A (this paper) tells you *how to tokenize* and owns the direction asymmetry story (including backtranslation for polysynthetic languages). The companion asks: given a new low-resource language, what data strategy maximizes quality as a function of resource level and typological profile?

**Core thesis:** Optimal data strategy (augmentation type, transfer source) for morphologically complex MT is a function of resource level and typological profile — not just raw data volume. Paper A handles the direction asymmetry; Paper B handles the resource-level strategy question across the full typological range.

### Research Questions

1. Does backtranslation help more at very-low vs. low vs. medium resource levels, and where does the ceiling kick in — corpus size or tokenization?
2. Does typological similarity predict transfer quality better than resource level? (e.g., Quechua → Mapudungun vs. Spanish → Mapudungun — typologically distant but geographically/lexically close)
3. Is there a generalizable decision procedure: given a new low-resource morphologically complex language and known resource level, what data strategy should you use?

### Methods to Compare

- Backtranslation (self-training on monolingual target-side data)
- Multilingual fine-tuning (pool all languages together)
- Sequential transfer (fine-tune on typologically similar language first, then target)
- Data mixing ratios (how much high-resource data to include)
- Synthetic morphological augmentation (generate inflected forms via paradigm completion)

### Language Design

Reuse the language set from Paper A, stratified by resource level:

| Resource Level | Languages |
|---|---|
| Very low (<50K pairs) | Mapudungun, Nahuatl |
| Low (50K–500K) | Quechua, Inuktitut, Yoruba |
| Medium (500K–5M) | Swahili |
| High (>5M) | Turkish, Finnish, Czech, Russian |

Transfer pairs to test: Quechua → Mapudungun, Turkish → Uzbek (if in scope), Czech → Slovak, Finnish → Estonian.

### Key Claim

Typological similarity is a better predictor of useful transfer than geographic proximity or script similarity — and this holds even controlling for resource level. If confirmed, this gives practitioners a principled way to choose a transfer source language without exhaustive search.

### Venue Target

ARR cycle following Paper A submission. Pipeline and language data already built; incremental compute cost only for augmentation and transfer experiments.
