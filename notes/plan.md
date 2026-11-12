# Vocabulary-Aligned Morphological Tokenization for Multilingual MT
## Research Plan

**Last synced with thesis_proposal.md: 2026-09-15 (confirmed current — BOUQuET dropped from Tier 3, estimator-validation gap disclosed in both docs).** This document owns operational planning (compute, timeline, red-team disposition, logistics, decision log). For current hypotheses, evaluation design, methods, and citations, `thesis_proposal.md` is the source of truth, not this section. Update this note whenever a sync pass is done so staleness is visible at a glance.

**Working title:** "Vocabulary-Aligned Morphological Tokenization Across Morphological Types: A Typological Study" (see `thesis_proposal.md` for the current title, "When Does Morpheme-Aware Tokenization Help? A Predictive Framework Across Language Types")

**Target venue:** ARR (ACL/EMNLP/NAACL cycle) → expand to TACL if results support full typological coverage

**Origin:** Follow-up to "Bringing Mapudungun into the Modern MT Ecosystem" (AmericasNLP 2026). The Mapudungun paper established Morfessor-VC as a proof-of-concept; this paper generalizes it into a principled framework, tests it across morphological types and model architectures, and releases a reusable toolkit.

---

## Core Thesis & Research Questions

**See `thesis_proposal.md`** (Problem, Hypotheses, Motivation sections) for the current, canonical statement of the thesis, research questions, and the two formal hypotheses (vocabulary density vs. typological category as the driver of benefit; source/target asymmetry). Do not restate or re-derive that content here; if it changes, update the pointer date above rather than copying prose into this section.

---

## Methods

### Existing (from Mapudungun paper)
**Morfessor-VC** — Run Morfessor unsupervised morpheme segmentation, then greedily merge adjacent morpheme pairs (left-to-right within each word) whenever their concatenation is itself already a single vocabulary item, regardless of whether the two original pieces were in- or out-of-vocabulary (this is the exact algorithm the Mapudungun paper's `segment_morfessor_vc.py` implements and published results from -- verified against that code 2026-09-16, correcting an earlier wrong description here that said the opposite: "merge OOV pieces with a neighbor until in-vocab"). Does **not** guarantee full in-vocab coverage of the output -- a piece can remain OOV if no adjacent merge happens to land in-vocab; see the OOV coverage note above. Helps source-side only, per the Mapudungun paper's own result.

### New Methods

**Vocab-biased BPE** — Modified BPE training that upweights merges producing tokens already present in the pretrained model's vocabulary. The merge score becomes:

```
score(merge) = freq(merge) × (1 + λ · 1[result ∈ pretrained_vocab])
```

where λ is a hyperparameter tuned per language (Optuna). No Morfessor dependency; purely BPE-based. Expected to help source-side tokenization without requiring a morphological analyzer.

**Vocab Expansion (Morfessor-VE)** — Run Morfessor; for morphemes *not* in the pretrained vocabulary, add them as new tokens and initialize their embeddings by averaging the embeddings of their constituent subword tokens. Fine-tune the full model including new embeddings. Addresses both source (better input segmentation) and target (model can generate morpheme tokens directly). WECHSEL is the closest prior work; Morfessor-VE differs in that the new vocabulary is morpheme-inventory-driven rather than cross-lingual-transfer-driven.

**Morfessor-VM (Vocabulary-Maximized) — added 2026-09-16, Tier 2.** Run Morfessor; unlike Morfessor-VC's opportunistic merging (which stops once no more in-vocab merges are available and can leave pieces OOV), VM forces every output piece into the existing vocabulary by continuing to merge an OOV piece with its neighbor until the whole word is either fully in-vocab or collapses to a single whole-word fallback token. Direct within-family counterpoint to Morfessor-VC along the same existing-vocabulary-only axis (no new tokens, unlike Morfessor-VE): if VM outperforms VC, partial alignment was leaving benefit on the table; if it doesn't, opportunistic merging already captures what matters. This is the algorithm the toolkit's `MorfessorVC` class implemented by mistake until 2026-09-16 (see Toolkit implementation status below) -- now correctly split into two distinct classes/methods, `MorfessorVC` (matches the published paper's algorithm) and `MorfessorVM` (this one, the original self-invented algorithm, kept because it's a real, testable method in its own right, not discarded as a bug).

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
| Icelandic | isl | Fusional | High | isl↔en |
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
- **Nahuatl — Axolotl corpus:** real, public, downloadable (UNAM, `corpus.unam.mx/axolotl`, also mirrored on HuggingFace as `somosnlp-hackathon-2022/Axolotl-Spanish-Nahuatl`). ~1.19M tokens, tens of thousands of Spanish-Nahuatl sentence pairs. Caveat directly relevant to a Morfessor-based method: the corpus spans **multiple Nahuatl varieties and orthographies** (dialectal + diachronic + orthographic variation, per the original LREC 2016 paper) — this could itself degrade morpheme segmentation quality independent of the tokenization method. **Resolved 2026-09, committed rather than hedged:** filter to the single most common variety/orthography in Axolotl before it becomes training data, decided now while still designing rather than deferred as a "possibly" for later. Sufficient in size for Phase 1 even after filtering to one variety; **not a blocker.**
- **Inuktitut — Nunavut Hansard 3.0:** real, public, CC-BY-4.0 (NRC Digital Repository). ~1.3M aligned sentence pairs — described in its own paper as the largest parallel corpus of any polysynthetic or Indigenous-Americas language released to date. Domain is legislative proceedings (1999-2017), so there's a domain-mismatch caveat versus FLORES-200's Wikipedia-style register if it's ever compared cross-language, but as a standalone Inuktitut test set it's large and well-documented. **Not a blocker**, easily the best-resourced of the three non-FLORES polysynthetic languages.
- **Net result: both fallback datasets check out.** Neither Nahuatl nor Inuktitut needs replacing on data-availability grounds — the earlier "if data is insufficient, replace with Basque/Wolof" contingency in this doc's original draft doesn't trigger. Remaining work before Phase 1 is mechanical: download both, run them through the same cleaning/dedup pipeline as Mapudungun, applying the single-variety filter to Nahuatl per the committed policy above.

**Data pipelines actually built, 2026-09-18 (`experiments/data_pipeline/{nahuatl,quechua}/cleaned/`):**
- **Nahuatl -- done, with one real deviation from the plan, disclosed rather than silently substituted.** The HF mirror's `var` (variety) column is entirely null -- no variety metadata exists to filter on as literally planned. Used `src` (source document) as a proxy instead, since different sources are produced by different translation efforts and plausibly have internally-consistent orthography even without an explicit variety label. Filtered to the single largest source, `opus-bible-uedin` (7,821 of 20,028 rows, ~39%) -- a Bible translation, typically produced by one team with consistent conventions. After removing empty rows and exact-duplicate pairs: 7,817 unique pairs, split 7,037/390/390 (train/dev/test, 90/5/5, deterministic shuffle seed=42, no official split existed to reuse).
- **Quechua -- done.** `somosnlp-hackathon-2022/spanish-to-quechua` on HuggingFace: confirmed genuine Ayacucho Quechua (matches `quy_Latn`, the FLORES-200/NLLB-200-covered variety already used in the grid), with its own train/val/test splits already provided. After cleaning (drop empty rows) and deduping: 99,293/12,674/12,689 (train/dev/test) -- a real, sizeable corpus, no fallback needed. (The official AmericasNLP shared task loader, `vgaraujov/americasnlp-mt-21`, uses a deprecated HF dataset-script format the current `datasets` library won't load; not pursued further since this source already works and is verified as the right variety.)
- **Inuktitut -- blocked, not done, caught before it became a real problem.** The only HF-available option, `EdinburghNLP/nunavut-hansard-plusplus`, turned out to have a broken parallel alignment: 673,680 English rows vs. 670,378 Inuktitut rows (not equal length), no shared alignment ID in either file, and the dataset card only says it was "deduped" without confirming that was done jointly across the pair -- the row-count mismatch confirms it wasn't. Zipping these together positionally would have silently fabricated a parallel corpus; caught and stopped before doing that, not discovered after. **Needs the original Nunavut Hansard 3.0 directly from the NRC Digital Repository (DOI: 10.4224/40001819)** rather than this derived mirror -- likely a manual download, not pip/HF-accessible. Still the right source per the original data audit above; just needs fetching properly.
  - **Resolved 2026-09-21.** Fetched the actual original directly from the NRC Digital Repository (`https://nrc-digital-repository.canada.ca/eng/view/dataset/?id=c7e34fa7-7629-43c2-bd6d-19b32bf64f60`, 202 MiB .tgz, verified real via Content-Disposition header before trusting the download, not just the DOI resolving). This version comes with proper pre-made, equal-length, officially-aligned train/dev/test splits (`split/{train,dev,test}.{en,iu}`) -- no alignment risk this time, unlike the broken HF mirror. Text is in Canadian Aboriginal Syllabics (the corpus's native script, per its own README -- "unlike previous versions... in syllabics"), not romanized Latin script; that's fine here since Inuktitut isn't in NLLB-200 at all (confirmed above), so there's no existing-vocab script to match against -- script choice is a free design decision for this language, unlike for the other 9 grid languages. After removing empty lines and exact-duplicate pairs (`experiments/data_pipeline/inuktitut/cleaned/`): **771,382 train / 3,587 dev / 4,219 test pairs** -- comfortably the largest of the three non-FLORES polysynthetic corpora (Mapudungun ~56K, Nahuatl ~7.8K train). Grid data status is now 11/12 languages with real, checked corpora -- only Mandarin (needs a dedicated CJK segmenter) remains. **Inuktitut's own degenerate-segmentation check completed 2026-09-21/22 (job 13834932): 0/40 near-character-level** (n=523,568 unique word types from the first 200K training lines) -- clean, same as every other language checked.
- **Degenerate-segmentation pre-flight check extended to Nahuatl and Quechua** using these real corpora (see Known Design Issues item 7 below for results).

---

## Model Scope

### Primary model: NLLB-200
- Sizes: 600M, 3.3B (skip 1.3B to halve compute; run 1.3B only on winners)
- Already set up in Mapudungun pipeline; all scripts reusable

### Secondary model: mBART-50
- Different architecture (denoising pretraining vs. translation pretraining)
- Different vocabulary (250K SPM vs. NLLB's 256K)
- Run on subset: polysynthetic languages + 2 agglutinative (Turkish, Finnish) + 1 fusional (Icelandic)
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
- Vocabulary density computed two ways: gold-derived (fraction of gold/reference morphemes already in the pretrained vocab) and an unsupervised estimate (Morfessor-output-vs-vocab overlap, no gold annotation needed) — reported both ways to quantify what's lost by removing the gold-annotation requirement, since the unsupervised estimate is what actually scales to the full 116-language NLLB population.
- **Estimator-validation result, computed 2026-09 (`scripts/unsupervised_vocabulary_density.py`, `results/unsupervised_vocabulary_density.json`, 100K-word deterministic subsample per language): the unsupervised estimator substantially underestimates gold density in all four validation languages, consistently.**

  | Language | Gold (affixes-only) type/token | Unsupervised type/token | Gap (type) |
  |---|---|---|---|
  | Finnish | 62% / 72% | 19.7% / 38.3% | −42 pts |
  | Hungarian | 20% / 94% | 14.8% / 39.3% | −5 pts |
  | Russian | 50% / 81% | 13.0% / 31.0% | −37 pts |
  | Czech | 86% / 99% | 18.9% / 52.6% | −67 pts |

  Direction is consistent (unsupervised always lower) but magnitude varies a lot (5–67 points), so this isn't a simple constant offset that could be corrected with one calibration factor. Plausible mechanism: Morfessor's own segmentation is coarser/different from gold linguistic morpheme boundaries (it optimizes a corpus-likelihood objective, not linguistic accuracy), so its predicted "morphemes" are frequently multi-morpheme chunks or boundary-shifted relative to gold — meaning fewer of them individually match a single vocab entry even when the underlying content is in-vocab. Hungarian's small gap vs. Czech's large one suggests this interacts with morphological type (Hungarian's more transparent agglutinative segmentation may align with Morfessor's output better than Czech's more fusional pattern) — matches the density-vs-typology test already planned above, and is itself now a candidate finding, not just a methods-validation footnote. **Not yet written up in thesis_proposal.md; the "Scalable density estimation" paragraph currently doesn't disclose this gap. Needs a decision: report the unsupervised estimate as-is with this validation caveat, or explore a correction (e.g. token-weighted density, or restricting Morfessor to a fixed operating point matched to gold's affix-count) before using it as the paper's actual scaled-up estimator for the 116-language population.**
- **Gold-validation language set, revised 2026-09 with real data.** Originally assumed UniMorph would supply gold segmentation for Turkish, Finnish, Icelandic, Russian, and Quechua; checked directly and UniMorph doesn't give morpheme boundaries at all (just lemma/form/feature-tag triples). MorphyNet does, but only covers Finnish and Russian of that original five. Since this step validates the *estimator* against real gold data, not the 12-language typological hypothesis, it doesn't need to match the experimental grid, so swapped to whichever MorphyNet-covered languages best span the typology: **Finnish + Hungarian (agglutinative), Russian + Czech (fusional)** — 2 real data points per category instead of 1, and all four now have actually-computed density numbers (`experiments/results/gold_vocabulary_density.json`): Finnish 62%/72% (type/token density), Hungarian 20%/94%, Russian 50%/81%, Czech 86%/99%.
- **Proposed, not yet done (from Chen et al. 2026):** (a) a retrieval-proxy baseline -- for each test sentence, the reference translation of its nearest training-set neighbor -- computed for Phase 0 and each grid language (no training needed); if a fine-tuned model doesn't clearly beat it, the chrF gap between methods means less than it looks. (b) their pretraining-exposure score (4-gram counts of the target text against NLLB's public bitext via infini-gram) as a concrete implementation of the pretraining-exposure covariate the test below already names but has not built.
- Density-vs-typology test: mixed-effects model, density as fixed effect, language as random intercept, fit on within-language density-benefit pairs per morpheme-frequency bucket (not one point per language, avoids conflating typology with per-language idiosyncrasy at n≈12). Covariates: raw pretraining-corpus token count and final tokenizer vocabulary size, to rule out exposure and vocab-size confounds respectively.
- Predictive framework validated via leave-one-language-out: fit on all languages but one, predict the held-out language's best method, report top-1 accuracy against a 25% chance baseline (4 candidate methods) and calibration (predicted-vs-observed chrF++ gain correlation, r > 0.5 pre-specified as "meaningfully predictive"). Quechua and/or Swahili held out entirely as a cold genus-level test (Quechuan; Bantu specifically, distinct from the leave-one-out pass).
- If this holds up, the toolkit can recommend a method without training — this is now treated as a primary contribution (see Open Questions below), not just a post-hoc chart.

### Phase 5: Ablations
- Vocab Expansion: embedding initialization strategy (average vs. nearest-neighbor vs. random)
- Vocab-biased BPE: λ sensitivity
- Morfessor hyperparameter sensitivity across languages

---

## Toolkit (the "framing" contribution)

**Package name: `morphotok`** (resolved, see Open Questions below).

**Implementation status, 2026-09: all three methods (Morfessor-VC, Morfessor-VE, Vocab-biased BPE) implemented for real in `src/morphotok/methods/`, superseding the earlier stub-only state.** Verified end-to-end via `MorphoTokenizer.tokenize()` against the real NLLB-200 600M tokenizer and MorphyNet Finnish word data (`.venv` sanity checks, 2026-09-15):
- **Morfessor-VC**: **corrected 2026-09-16.** The first implementation used a self-invented algorithm (merge OOV pieces with a neighbor until full in-vocab coverage) that does not match the Mapudungun paper's actual `segment_morfessor_vc.py` -- that mismatch was caught when reviewing the original code directly for the Phase 0 pilot (see below) and is exactly what produced the "cascades to whole-word fallback" behavior observed in yesterday's Finnish testing; that behavior was an artifact of the wrong algorithm, not a real property of Morfessor-VC. Rewritten to match the paper's actual cascade-merge algorithm (greedily merge adjacent pairs when their concatenation is already in-vocab, regardless of the pieces' own vocab status; does not guarantee full coverage of the output, unlike the earlier version). Toolkit class and Methods description above both now match.
- **Morfessor-VE**: trains Morfessor, collects OOV morphemes (word-initial "▁"-marked variants tracked separately) as `new_tokens()`, segments via Morfessor's own output directly (no merge-back, since post-expansion everything is in-vocab by construction). Embedding-averaging itself is left to the training pipeline that consumes `new_tokens()`, not this class's job.
- **Vocab-biased BPE**: real from-scratch BPE trainer (no existing library exposes a bias term in merge scoring) implementing `score = freq * (1 + lambda * 1[merged in vocab])`. First version used a full-corpus rescan per merge (O(words * merges)) and became impractically slow at real scale (100K words, 8000 merges: didn't finish in 90s); rewritten with the standard incremental pair-count-update trick (only re-derive counts for words containing the just-merged pair), reducing full runs to ~400-540s/language via SLURM (job 13706393, `scripts/train_vocab_biased_bpe.py`, `results/vocab_biased_bpe_smoke.json` -- 99.6-100% of words split into >1 piece across Finnish/Hungarian/Russian/Czech at 8000 merges, confirming the trainer produces real segmentation, not degenerate no-ops).
- Auto-mode (`resolve_method()` for `method="auto"`) still correctly raises `NotImplementedError` -- unblocked only once Phase 4's predictive framework exists, not before.

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
- **Per direction/morphology cell, CometKiwi-23 or MetricX-24**, as recommended by AmanaMT (Thompson, 2026)'s typological metric-selection guide, rather than one fixed COMET checkpoint across all language/direction pairs. AmanaMT found translation direction (not morphological type) is the dominant predictor of which metric is reliable, and that COMET-22/23-family reliability itself varies by direction, which is a genuine confound for this project's own direction-asymmetry hypothesis (Hypothesis 2 in thesis_proposal.md): an apparent lack of target-side tokenization benefit could partly reflect the metric being less reliable evaluating into a given language, not a real null effect. Cross-check results against the direction-appropriate metric where possible. **For the six languages overlapping with AmanaMT's own benchmark (Turkish, Finnish, Swahili, Icelandic, Russian, Mandarin), use that study's actual per-language measured reliability rather than the general guide** — free, since the data already exists.
- BLEU for comparison with prior work.
- COMET-22 is still avoided as a fixed default for OOD languages not in its training data (Mapudungun, Nahuatl) for the same reason as before; the per-cell CometKiwi-23/MetricX-24 approach supersedes needing a single blanket "use COMET-23" rule.

### Metric disagreement risk
Thompson et al. (subchar-mt, forthcoming) find that morpheme segmentation improves chrF++/BLEU but COMET flips the ranking (baseline > morphemes) for Chinese MT. This may replicate here. Paper A should report both metrics and address the disagreement directly if it appears — this is prior work to cite, not a gap to fill.

### Human evaluation
**Revised 2026-09: back in scope, split minimal (Tier 1) / extensive (Tier 3 stretch)**, superseding the earlier "cut entirely" call. The fix for "don't know speakers of all 12 languages" isn't dropping human eval, it's scoping it by typological category (one language per category: polysynthetic, agglutinative, fusional, analytic) rather than by individual language — this keeps the design symmetric at the level the paper actually argues at (typology), rather than either attempting all 12 (infeasible) or singling out Mapudungun alone (asymmetric attention on one of twelve languages, which risks reading as a Mapudungun paper with 11 languages attached rather than a genuine typological study).
- **Minimal (Tier 1):** Mapudungun fills the polysynthetic slot with **two annotators (resolved 2026-09):** the existing AmericasNLP annotator plus Rogers, the Mapudungun linguist already collaborating on the mapudungun-syntax gold-standard project — gives real inter-annotator agreement for this category essentially for free, rather than inheriting the Mapudungun paper's own single-annotator/no-IAA limitation. Agglutinative/fusional/analytic slots need an actual annotator audit before committing — Turkish/Finnish plausible for agglutinative, Icelandic/Russian for fusional, Vietnamese/Mandarin/Yoruba for analytic, all worth checking against BYU's language population rather than assumed. ~30-50 sentences per language, preference ranking between Standard BPE and best new method, MQM-lite error typology.
- **Extensive (Tier 3 stretch):** broader language coverage, multiple annotators with measured inter-annotator agreement — this is what actually answers the Mapudungun paper's own stated need for a stronger human-eval design (its pilot had one annotator, no IAA, 20 sentences for linguistic annotation).
  - **Revised 2026-09-16: promoted out of Tier 3 for the categories where the annotator-scarcity blocker is actually resolved, rather than kept stretch-only across the board.** The Tier 3 demotion existed because of assumed annotator scarcity; that's no longer true everywhere. **Swahili (agglutinative) and Yoruba (analytic): Extensive tier confirmed feasible via PathSay** -- real active contributors confirmed (BYU PathSay annotator-activity data, 2026-09-16: e.g. Swahili IDs 7/28/70/76/90/91/92/115, Yoruba IDs 20/47/49/50/51/77/78/81/83/94), and PathSay is the path-of-least-resistance recruiting channel for these two since it's already staffed and active there. One caveat not yet resolved: that activity data shows people doing translation-production work, not MT-output evaluation -- still need to confirm at least one or two of these contributors are willing to do a preference-ranking/MQM-lite task, not just assume the task type transfers.
  - **Icelandic/Russian (fusional) and Turkish/Finnish (agglutinative alternative): confirmed CLOSED 2026-09-18 -- not in scope for PathSay.** Sam Bird checked directly against the Church's confidential language-development list (held by Dr. Steve Richardson): none of the four are on it, so they're not currently in line with PathSay's actual mission regardless of NLLB resource-level. The earlier "mission-mismatch... but High-resource languages do exist in PathSay" reasoning (2026-09-16) was the right instinct but the wrong prediction -- real answer beats the plausibility argument. These four stay Minimal-tier via a different recruiting channel (BYU population), not PathSay. Core grid unaffected either way (see above -- training/automatic-metric experiments for these four don't depend on PathSay at all, only the deep human-eval layer does).
  - **Sam's language-selection priority framework (2026-09-18), useful beyond just this question:** (1) Active PathSay -- languages with real current annotator hours (confirmed: Swahili, Yoruba, Zulu, Xhosa, Southern Sotho, Tswana per the annotator-activity CSV; Sam says only ~5 languages are actively being translated at any time). (2) PathSay languages with data but no annotators yet ("lots of segments, no hours" -- needs hiring, but the pipeline already exists, no new data-gathering required). (3) Something new, gated on getting Dr. Richardson's permission to see the actual confidential Church list. **Worth revisiting the earlier fusional-candidate search (Indo-Aryan cluster: Gujarati, Marathi, Odia, Punjabi, Assamese, etc. -- see the rejected-candidates note below) against real segment-count data under category (2)** -- that search only checked typology/resource-level and confirmed-annotator-activity, not the "has data, zero hours yet" middle tier Sam just described, so it may be incomplete. Not yet done.
  - **Considered and rejected: swapping in a PathSay-available language for the fusional slot to sidestep the recruiting gap.** Amharic looked promising at first (real active PathSay annotators, Afroasiatic family so no phylogenetic redundancy with Russian/Icelandic) but checked against this project's own existing WALS-based classification (`results/nllb_morphology_stats.json`) and found Amharic is classified **introflexive** (root-and-pattern/templatic Semitic morphology), not fusional -- a structurally distinct category, not a subtype, per WALS and this project's own Definitions section. Using it as a "fusional" stand-in would be a real mislabeling, the same class of error already caught and fixed for Quechua/Yoruba earlier. Belarusian/Bosnian (Slavic, redundant with Russian) and German (Germanic, redundant with Icelandic) were also considered and rejected for reintroducing the phylogenetic-pseudo-replication problem the Czech->Icelandic swap already fixed. No clean PathSay-available fusional substitute exists; requesting PathSay add and recruit for Icelandic/Russian directly remains the right path, not a language swap.
  - **Revised 2026-09-18, now that Icelandic/Russian/Turkish/Finnish are confirmed closed for PathSay (see above): Catalan and Basque are live options, not rejected.** Systematically checked all of PathSay's priority-2 languages (has data per Sam, no confirmed active hours -- computed as PathSay's full coverage list minus the confirmed-active-annotator subset from the CSV) against this project's own WALS-based classification and NLLB resource-level data. **Catalan (cat): fusional, High-resource, Romance family** -- clears every bar the earlier candidates failed on (not introflexive like Amharic, not family-redundant with Russian/Icelandic like Belarusian/Bosnian/German), and has a real PathSay data/recruiting path per Sam's priority-2 framework. The standout candidate for the fusional slot if Icelandic/Russian recruiting via BYU doesn't pan out. **Basque (eus): agglutinative, High-resource, language isolate** -- not urgently needed (Turkish/Finnish/Swahili already cover agglutinative, Swahili is a confirmed fallback), but a real backup if Turkish/Finnish annotator recruiting also fails. Neither yet proposed to Sam or confirmed for actual segment volume -- "has data" per Sam's blanket statement doesn't confirm real quantity, same caveat as everything else in this section.
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
- **Fertility results for the 4 gold-validation languages, computed 2026-09 (`scripts/fertility_analysis.py`, SLURM job 13706031, 100K-word samples, `results/fertility_analysis.json`):** Finnish 6.364 (standard BPE) vs 2.452 (Morfessor); Hungarian 5.921 vs 2.110; Russian 5.800 vs 2.111; Czech 5.262 vs 2.059. Consistent pattern across all four: Morfessor produces roughly 2-2.5 pieces/word, well under half of NLLB's own frequency-driven BPE fertility (5.3-6.4) -- Morfessor's morphological segmentation is structurally coarser than NLLB's BPE, not finer, which is a useful sanity-check data point for the fertility/quality tradeoff story (Rust et al., 2021): the plausible benefit mechanism isn't "more tokens closer to real morphology," it's "fewer, more meaningful tokens." 
- Vocabulary coverage: % of morphemes in pretrained vocab, before and after expansion
- Attention entropy: does morpheme segmentation concentrate attention on more meaningful spans?
- **Unsupervised vocabulary density + fertility computed across all 10 remaining grid languages, 2026-09-21/22 (SLURM array job 13835049, `results/per_lang/{density,fertility}_*.json`) -- first time density spans all four typological categories, not just the fin/hun/rus/ces validation set:**

  | Language | Category | Density by type | Density by token | Standard BPE fertility | Morfessor fertility |
  |---|---|---|---|---|---|
  | Vietnamese | analytic | 60.5% | 86.5% | 3.91 | 1.90 |
  | Swahili | agglutinative | 46.3% | 82.6% | 4.43 | 2.25 |
  | Yoruba | analytic | 40.5% | 73.7% | 4.86 | 2.25 |
  | Turkish | agglutinative | 42.3% | 69.2% | 4.61 | 2.07 |
  | Icelandic | fusional | 33.4% | 65.3% | 5.07 | 2.19 |
  | Mapudungun | polysynthetic | 27.2% | 52.7% | 5.70 | 2.35 |
  | Mandarin | analytic | 24.5%* | 36.7%* | 4.70* | 1.20* |
  | Quechua | agglutinative/polysynthetic | 24.2% | 41.6% | 5.82 | 2.21 |
  | Nahuatl | polysynthetic | 16.4% | 41.3% | 7.36 | 2.58 |
  | Inuktitut | polysynthetic | **1.5%** | **5.1%** | 4.33 | 2.85 |

  *Mandarin used jieba word segmentation (not whitespace splitting like every other language here), so its numbers aren't directly comparable methodology-wise -- flagged, not silently pooled with the rest.

  **Notable: Inuktitut's density is dramatically lower than every other language** (1.5% by type vs. the next-lowest, Nahuatl, at 16.4%) -- consistent with it being both polysynthetic and entirely absent from NLLB's pretraining data (unlike Mapudungun and Nahuatl, which at least share Spanish-adjacent orthographic/lexical overlap via loanwords and Latin script; Inuktitut's syllabics script shares almost nothing with NLLB's vocabulary). This is itself a testable prediction under Hypothesis 1: if density drives benefit, Inuktitut should show the *largest* benefit from vocab-alignment methods of any language in the grid, polysynthetic or not -- worth flagging as a specific, falsifiable expectation before Phase 1 results come in, not just noting the number.
  - **Caveat (2026-10-07):** the 1.5% figure was a script-coverage artifact. 42.6% of syllabics tokens are `<unk>` under NLLB. After deterministic romanization (`experiments/scripts/translit_inuktitut.py`, 99.67% word-exact against the Hansard++ romanized release), density is 9.6% by type and 25.2% by token. That is still the lowest in the grid.
  - **RESULT (2026-10-08): prediction tested and not supported.** 600M, 25 epochs, 256 tokens, size-matched (56,197 random train pairs, seed 42), romanized, one seed. Results JSONs: `results/ikt_bootstrap_{ikt_en,en_ikt}_e25_len256.json`. Morfessor-VC is significantly *worse* than Standard BPE in both directions, on corpus chrF, sentence-mean chrF, and BLEU. ikt->en: -0.55 corpus chrF [-0.83, -0.25]. en->ikt: -0.52 [-0.78, -0.24]. Together with Mapudungun's two-seed null at 256 tokens, Morfessor-VC shows no benefit for NLLB fine-tuning at the lowest density tested.
  - **Caveat carried over from the fin/hun/rus/ces validation:** these are all unsupervised (Morfessor-based) density estimates, which that validation found substantially underestimate gold density (5-67 points, direction consistent, magnitude not). Read these as a consistent *ordering* signal across languages, not as calibrated absolute density values.
  - **Word-list caveat:** Turkish/Swahili/Icelandic/Vietnamese/Yoruba/Mandarin word lists come from FLORES+ dev/devtest (7K-16K unique types), much thinner than the 100K-word samples used for Mapudungun/Nahuatl/Quechua/Inuktitut (drawn from their own training corpora) -- noted per-language via `n_words_used` in the results JSON, not pooled as if equally reliable.
  - **Still missing for the full Hypothesis 1 test:** actual MT benefit (chrF gain from vocab-aligned methods) for these 9 languages -- only Mapudungun has real benefit data so far (Phase 0). Density alone doesn't test the hypothesis; it's one half of the density-vs-benefit correlation this framework is built on.

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
- **BOUQuET eval — closed/deferred 2026-09, doesn't apply as scoped.** Confirmed by actually loading `facebook/bouquet` (gated access obtained 2026-09-15): the dataset is English-centric by construction — every row is `src_lang → eng_Latn` (verified structurally: the same `par_id` groups the same source paragraph translated into English from ~180 source languages; arn_Latn appears only as a source language, paired with eng_Latn, never with spa_Latn). All existing trained Mapudungun models are arn↔es (checked every checkpoint dir under `mapudungun_final_models/` and `mapudungun_models/`). So BOUQuET can't evaluate the existing models as-is without adding an English direction to model scope, which isn't otherwise motivated (mT5/Aya English-pivot work is already Tier 3). Not worth a two-hop arn→es→eng pivot eval either — compounding errors from an off-the-shelf es→eng hop would be a weak, noisy signal, not a clean benchmark number. Dropped rather than pursued; if model scope ever expands to include English, revisit.
- **Source-copy failure mode:** Check whether the es→arn source-copy failure observed in Mapudungun (model reproduces Spanish source verbatim in Mapudungun orthography) appears in other polysynthetic language pairs. Purely analytical — no new experiments beyond inspecting outputs. **Done for Mapudungun 2026-09:** `scripts/source_copy_failure_analysis.py`, run on the 1.3B es→arn-morfessor condition — 46/9,382 (0.49%) of predictions flagged as likely source-copy (normalized SequenceMatcher ratio thresholding). Low rate; not a dominant failure mode for this condition. Extending to other polysynthetic pairs is blocked on those models/predictions existing.

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
- **Pretraining-time vocabulary coverage / plasticity**: Abagyan et al. (2026, ACL long, Cohere Labs), "One Tokenizer To Rule Them All" -- a tokenizer trained on more languages than the model is pretrained on gives ~19% higher win-rate adapting to new languages, 8x faster adaptation, and +5.3 BLEU on FLORES-200 for expanded languages, with ~no cost on primary ones. The upstream, pretraining-time mirror of this project's post-hoc framing: same mechanism (vocabulary coverage drives adaptation ease), opposite end of the pipeline. Added 2026-09-23.
- **Contextualizing XLR MT scores**: Chen et al. (2026, ACL short), "Translation or Recitation?" -- FRED difficulty metrics (Fertility, Retrieval proxy, pretraining Exposure, corpus Diversity); finds the retrieval-proxy (nearest-neighbor) score is the dominant predictor of reported BLEU (R2=0.58) and many XLR systems fail to beat it. Relevant as a caution against comparing raw chrF across this project's 12-language grid without contextualizing it. Their table 10 also gives an external reference point for Quechua (quy->es, 125K sentences, BLEU 8.76 / chrF++ 33.83, AmericasNLP 2025). Note their "fertility" is tokens/character, not this project's tokens/word -- same word, different metric. Added 2026-09-23.

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
- **Phylogenetic pseudo-replication (2/8):** ~~Czech+Russian are both Slavic~~ **resolved 2026-09 by swapping Czech → Icelandic** (Germanic, not Slavic; confirmed fusional per our own WALS-based classification, unlike Georgian which is actually agglutinative despite being the other candidate originally floated — see Model Scope). Vietnamese+Mandarin sharing areal features is less of a concern given Yoruba (Niger-Congo, unrelated family/area) is a third analytic-control language, not just two.
- **Morfessor degenerate-segmentation risk (2-3/8):** low monolingual data for Mapudungun/Nahuatl could make Morfessor segmentation degenerate before any training even starts. Fix: manually inspect segmentations for all languages (~1hr CPU) before launching Phase 1.
- Single-sample flags worth a look but not yet convergent: WECHSEL may be a category error in the comparison (it changes embedding init, not tokenization); 5 GPU-hrs/run may undertrain larger-vocab methods, systematically favoring Morfessor; vocabulary-size itself is a confound between Morfessor and BPE outputs at "the same" merge count.

**Net effect on this plan:** corroborates the advisor's Phase 0 instinct above — a single-language pilot is a natural place to shake out the Quechua/Yoruba reclassification and Morfessor-inspection fixes cheaply before they get baked into a 144-run grid.

**Full disposition of every red-team item (2026-08-19 — user asked to address all of them, not just the blocking three):**

1. **Corpus-size confound → fixed by design.** Add a matched-size condition to Phase 1: for every non-Phase-0-tier language, run an additional training condition subsampled to the sentence count of the lowest-resource polysynthetic language in the grid (Mapudungun or Nahuatl, whichever ends up smaller). Report both the natural-size and matched-size results side by side in the main table — if the effect survives at matched size, corpus size is ruled out directly rather than argued around. Adds roughly one subsampled run per non-low-resource language/method cell; scope this against the Compute Estimate table once Phase 0 timing is known.
2. **Backtranslation confound → fixed above** (moved to its own pre-registered ablation, excluded from cross-type claims).
3. **Quechua / Yoruba misclassification → fixed above** (language table corrected).
4. **COMET-23 coverage gap → add a verification step.** Before Phase 1 launches, check XLM-R's (COMET-23's backbone) training-language list for arn/nah/ikt/quy coverage — this is a lookup, not an experiment. Whatever the result, report chrF (character-only, sacrebleu default word_order, distinct from chrF++) as a metric that doesn't depend on COMET-23's backbone coverage at all, alongside chrF++ and COMET-23, for every language.
5. **chrF++ differential sensitivity across typology → add the decomposition.** Report character-only chrF and word-only chrF components separately (not just the blended chrF++) in an appendix table, so a reader can see whether the headline effect is driven by the character component (which would suggest it behaves similarly across polysynthetic and analytic languages) or diverges structurally.
6. **Phylogenetic pseudo-replication → fixed by design swap, not just scoped language, resolved 2026-09.** Since the grid wasn't locked in yet, swapped Czech → Icelandic directly rather than deferring the fix to Phase 2/TACL and living with a "Slavic fusional" scoping caveat in the meantime. Fusional category is now Icelandic + Russian, different families (Germanic vs. Slavic), same resource tier (both High), so no new resource-level confound introduced. Georgian was considered and rejected: our own WALS-based classification tags it agglutinative, not fusional, so it would have been internally inconsistent to use as a "non-Slavic fusional" substitute. Vietnamese/Mandarin areal concern is softer than originally stated since Yoruba (Niger-Congo, unrelated family/area) is a third analytic-control language, not a pair.
7. **Morfessor degenerate-segmentation risk → pre-flight check done for Mapudungun (2026-08-19), still open for the other 11 languages.** Loaded the existing trained model (`/nobackup/archive/usr/it238/mapudungun_morfessor_model.bin`, from the AmericasNLP pipeline) and ran `viterbi_segment` on 40 real word forms sampled from the raw AVENUE transcription corpus (`projects/mapudungun/mapudungun-corpus`, not synthetic test strings). Result: **1/40 (2.5%) came back near-character-level** (`fep` → `fe+p`, a 3-letter word — expected noise on very short forms, not a systematic failure); the other 39 produced linguistically plausible-looking morpheme splits (`amuleayin` → `amule+ay+in`, `witranalwue` → `witran+al+wue`, `ngillaentukelaymi` → `ngilla+en+tukelaymi`). **Verdict: Mapudungun's existing Morfessor model is not degenerate** — Phase 0 can proceed on the current segmentation without re-tuning. One artifact found and worth fixing regardless: one sample word (`nmlchnfmcm2x0155nfmcm00SPAno`) was a leaked transcription-ID token, not real text — the corpus-cleaning step needs a tighter filter for the AVENUE `.trl` transcript-ID format before this corpus is reused as-is. The other 11 languages (Nahuatl, Inuktitut, Quechua, Turkish, Finnish, Swahili, Icelandic, Russian, Vietnamese, Mandarin, Yoruba) haven't been checked yet — none have a trained Morfessor model to inspect until their data pipelines exist, so this stays open until each language's corpus lands. **Update 2026-09-18: extended to every language with a real corpus now (10 of 12)** -- Finnish/Hungarian/Russian/Czech, Turkish/Swahili/Icelandic/Vietnamese/Yoruba (see Language Selection section), and now Nahuatl (0/40 near-character-level, n=23,597 words) and Quechua (0/40, n=143,717 words), both checked against the real corpora built 2026-09-18 (see Data pipelines note above). Only Mandarin (needs a dedicated CJK segmenter, different check entirely) and Inuktitut (blocked on the NRC source, see above -- no corpus to check yet regardless) remain.
   - Housekeeping note: the actual training-ready corpus files under `/nobackup/autodelete/usr/it238/mapudungun/data-processed/` (the `morfessor`, `cleaned`, `morfessor_vc` subdirs referenced by this plan) have **already been auto-deleted from `nobackup/autodelete` scratch** (BYU's autodelete policy) — only trained model checkpoints and predictions survive in `nobackup/archive`. The raw AVENUE transcripts in the git-tracked `mapudungun-corpus` repo are safe, but the *processed/cleaned/tokenized* intermediate files need regenerating from scratch before Phase 0 training can actually run. Worth confirming the corpus-prep scripts that produced them still exist somewhere reproducible (not just their output).
   - **Extended 2026-09 to the 4 gold-validation languages (Finnish, Hungarian, Russian, Czech)** using real MorphyNet word forms (20K-word samples, 40-word random inspection each). Result: **0/40 near-character-level in every language** (threshold: avg piece length <=1.5 chars, word length >3) -- segmentations look linguistically plausible across all four (e.g. Finnish `kuukausitaseetta` -> `kuukausi+tas+eetta`, Russian `обстановке` -> `об+станов+ке`, Czech `manhattanský` -> `man+hattan+ský`). None of these four show the degenerate-segmentation risk. Still open: Turkish, Swahili, Icelandic, Vietnamese, Mandarin, Yoruba, Nahuatl, Inuktitut, Quechua -- none have a checked word list/trained Morfessor model yet.
   - **Extended 2026-09-16 to Turkish, Swahili, Icelandic, Vietnamese, and Yoruba** using real FLORES+ dev/devtest word forms (`openlanguagedata/flores_plus` on HuggingFace, not gated, 7K-16K unique words per language from ~2,000 sentences -- much thinner than the 100K-word MorphyNet samples above, but sufficient for this 40-word spot-check). Result: **0-2/40 near-character-level in every language** (Turkish 0/40, Icelandic 0/40, Swahili 1/40, Vietnamese 1/40, Yoruba 2/40) -- all within normal noise, none flagged as degenerate. **Mandarin (cmn) skipped for this check**: FLORES text has no whitespace between words (Chinese doesn't use spaces), so naive `.split()` produced clause-length fragments instead of real words -- would need a dedicated CJK word segmenter (e.g. jieba) first, and as an analytic/isolating language the word-boundary story is different in kind anyway (morpheme ≈ character), so this whitespace-based check doesn't straightforwardly apply. Still fully open: Nahuatl, Inuktitut, Quechua -- their corpora (Axolotl, Nunavut Hansard, existing Quechua resources) haven't been pulled into this pipeline yet.
8. **WECHSEL category-error concern → clarify in the paper, not the design.** WECHSEL changes embedding initialization, not tokenization/segmentation — keep it in the baseline table but add one sentence in Methods explicitly stating it tests a different axis (representation transfer, not vocabulary alignment), so a WECHSEL win doesn't get mistakenly read as evidence against or for the tokenization hypothesis.
9. **Undertraining at ~5 GPU-hours/run → add a pilot.** Before committing the full 720 GPU-hours, run a 3-config pilot (Standard BPE, Morfessor-VC, Vocab Expansion) at 3x the planned budget (~15 GPU-hours each, ~45 GPU-hours total) on the Phase 0 language, and check whether method rankings match the 5-hour budget's rankings. If they don't, increase the Phase 1 per-run budget before launching — this is the one red-team fix that costs meaningful extra compute, but it's cheap insurance against a systematic bias toward smaller-vocabulary methods.
   - **Launched 2026-09-16 (SLURM jobs 13723727/28/29, `experiments/slurm/phase0_pilot_*.sh`, es->arn direction, NLLB-200 600M, 16h wall-clock cap).** Data prep reused the original AmericasNLP paper's own validated scripts against the regenerated pilot data (`experiments/data_pipeline/mapudungun/pilot_data/`): `segment_morfessor.py` (99,426 unique train word types, ~61% get a >1-piece segmentation) then `segment_morfessor_vc.py` (cascade merge against the real NLLB-200-600M vocab, 3.7-9.3% @@-marker reduction across splits). The Vocab Expansion condition had no precedent to reuse -- built `experiments/scripts/prep_vocab_expansion_data.py` (using the toolkit's `MorfessorVE` class) and `experiments/scripts/finetune_nllb_pilot.py` (adds real embedding-expansion support: new OOV tokens get their embedding initialized as the mean of the *old* tokenizer's own sub-piece encoding of that surface string, then the model is resized). Found 13,024 new OOV morpheme tokens needed (~5% growth over NLLB's 256K vocab) from the 106,571 unique training words.
   - Partition/QOS note for future SLURM submissions on this account: `--partition=dw` (used by the original paper's own training jobs) requires `--qos=dw87`, not `--qos=gpu` -- neither exists for this account's actual allocation. Working combination found by checking a real prior successful GPU job's recorded QOS: `--partition=cs --qos=cs --gres=gpu:a100:1`.
   - **Completed 2026-09-16, results in (`experiments/models/phase0_pilot/*/test_results.json`):**

     | Condition | Test chrF | Test BLEU | Wall-clock |
     |---|---|---|---|
     | Standard BPE | 44.79 | 9.95 | ~1h56m |
     | Morfessor-VC | 44.67 | 9.90 | ~2h02m |
     | Vocab Expansion (Morfessor-VE) | 43.41 | 8.80 | ~2h40m |

     **Reading these results:** es->arn is the target-side direction where the AmericasNLP paper found Morfessor-VC's source-side advantage "shrinks and reverses at scale." This pilot is consistent with that, not contradicting it: Morfessor-VC is essentially tied with Standard BPE here (-0.12 chrF, not a meaningful difference on a single run without significance testing), and Vocab Expansion actually underperforms both (-1.38 chrF vs. Standard) despite adding 13,024 new embedding parameters. Read as directional evidence for the target-side asymmetry (Hypothesis 2) holding up in a real run with the regenerated data and the newly-implemented methods, not as a discouraging result -- a clean win for the new methods on the *hard* direction this early would actually be more surprising than this.
     - **Undertraining check (red-team item 9): CORRECTED TWICE, 2026-09-24 -- the 10-epoch Phase 0 numbers were undertrained for every condition.** First error: the text here said the runs "converged via early stopping." Early stopping never triggered; every 10-epoch run used the full cap. Second error, in my first correction: I said Standard and Morfessor-VC had plateaued (last three dev chrF within ~0.4), so undertraining was unlikely for them. That inference was invalid: the linear learning-rate schedule decays to zero at the last epoch, which flattens the dev curve regardless of convergence. The 25-epoch es->arn runs show it (jobs 13878734/35; lr schedule stretched to 25 epochs, early stopping patience 3): **Standard 46.66 chrF / 11.01 BLEU (10-epoch run: 44.79 / 9.95), Vocab Expansion 45.81 / 10.44 (43.41 / 8.80).** Standard's dev chrF kept rising from ~44 (epoch 10) to 45.3 (epoch 21) before stopping at epoch 24; Vocab Expansion reached 44.8 dev and held for the last five epochs. So both were undertrained at 10 epochs, by about 1.9 (Standard) and 2.4 (Vocab Expansion) test chrF, and **the Standard-minus-Vocab-Expansion gap shrank from 1.38 to 0.85 chrF but did not close** (dev gap 0.48). This is the direction the original red-team concern predicted (a method with extra parameters converges more slowly and looks worse under a short budget), but the effect is modest, and this is one seed and one direction -- not yet bootstrapped. **Consequences:** every 10-epoch comparison in this section (both directions, the frequency-cap ablation, the seed-43 runs, the batch-128 runs, the retrieval-baseline comparison) is at an under-trained budget and should be treated as provisional; rankings could move with longer training, in particular Morfessor-VC vs. Standard, which has not been run at 25 epochs in either direction. The pre-registered 15h/45h check was never run. Launched the missing matched runs at 25 epochs: es->arn Morfessor-VC and arn->es Standard / Morfessor-VC / Vocab Expansion (jobs 13885150-53).
     - **Not yet done:** the actual 5h-vs-15h ranking comparison the pre-registration called for, since no separate 5h-budget run exists to compare against (early stopping made the distinction irrelevant for this run). Statistical significance testing (bootstrap, per the Evaluation section) not yet run on these three single-seed results -- worth doing before reading the Standard-vs-Morfessor-VC gap as anything more than directional.

     - **Bootstrap significance test run 2026-09-18** (`scripts/phase0_predict_and_bootstrap.py`, SLURM job 13756446, paired bootstrap n=1000 on all 9,367 test sentences' per-sentence chrF, `results/phase0_pilot_bootstrap.json`):

       | Comparison | Mean diff | 95% CI | p (approx) | Significant? |
       |---|---|---|---|---|
       | Morfessor-VC vs. Standard | -0.19 | [-0.37, +0.01] | 0.06 | **No** |
       | Vocab Expansion vs. Standard | -1.68 | [-1.96, -1.41] | <0.001 | **Yes** |
       | Vocab Expansion vs. Morfessor-VC | -1.49 | [-1.77, -1.23] | <0.001 | **Yes** |

       **Real finding, stated plainly rather than softened: Morfessor-VC is genuinely indistinguishable from Standard BPE on this pilot (not significant, consistent with the target-side-asymmetry reading above), but Vocab Expansion is reliably, significantly worse than both** -- not noise, a real effect at n=9,367. This is notable because Vocab Expansion (Morfessor-VE) is specifically the method thesis_proposal.md's Hypothesis 2 predicts should help most on the target side (it can generate morpheme tokens directly, unlike VC which only constrains segmentation). One pilot on one language doesn't overturn the hypothesis, but it's a real, disclosed data point against it, not for it.

       **Diagnosed 2026-09-18/20** (`scripts/phase0_diagnose_vocab_expansion.py`, SLURM job 13758474, `results/phase0_vocab_expansion_diagnosis.json`): checked whether the 13,024 new embedding tokens actually get used at inference time, or sit unused while the model falls back to standard fragments. **They get used: 99.04% of generated test sentences contain at least one new token, and new tokens make up 30.9% of all generated tokens -- close to (not far below) the 38.3% rate they appear at in the gold reference text.** This rules out the simpler "embeddings never became useful" explanation -- the model is actively choosing to use them most of the time, not avoiding them. The remaining, more likely explanation: with 13,024 new tokens spread across only ~56K training sentences, each individual new token gets very sparse training signal, so the model may learn *that* it should use a new token in a given context without reliably learning *which* one is correct among thousands of rarely-seen options -- consistent with a real chrF drop despite a usage rate close to the reference ceiling (the model picks *a* plausible new token often, just not always the *right* one). Not yet directly confirmed (would need per-token accuracy broken out by training frequency, not yet run) but the strongest-supported explanation given what's been checked. **Implication for Phase 1: this argues for larger per-language training data before trusting Vocab Expansion's results, or a vocabulary-growth cap relative to corpus size (e.g. only add new tokens above some frequency threshold, shrinking the added vocabulary to fit the data available) rather than abandoning the method outright** -- the mechanism (add new tokens, initialize by averaging) isn't obviously broken, the token-to-data ratio in this specific pilot plausibly is.

       **Frequency-cap ablation run 2026-09-21/22 to test this directly -- result: the sparse-signal explanation does NOT hold up** (jobs 13835015/16/17, `experiments/models/phase0_pilot/nllb-600M-es-arn-vocab_expansion_min{5,20,100}/`):

       | Condition | New tokens | Test chrF |
       |---|---|---|
       | Standard BPE | 0 | 44.79 |
       | Morfessor-VC | 0 | 44.67 |
       | Vocab Expansion (original) | 13,024 | 43.41 |
       | Vocab Expansion (min count >=5) | 6,920 | 43.53 |
       | Vocab Expansion (min count >=20) | 3,006 | 43.51 |
       | Vocab Expansion (min count >=100) | 787 | 43.77 |

       Cutting the added vocabulary by 94% (13,024 -> 787, keeping only the highest-frequency new tokens) moved chrF by only +0.36 -- nowhere near closing the ~1.3-1.4 point gap to Morfessor-VC/Standard. **This rules out sparse per-token training signal as the primary cause.** The underperformance is not a vocabulary-size problem; it persists essentially unchanged whether 13,024 or 787 tokens are added. Two explanations survive: (1) the averaging-based embedding initialization is a poor starting point regardless of how many tokens use it, or (2) forcing the model to generate *any* brand-new output tokens is intrinsically harder than staying within its pretrained vocabulary, independent of scale. Neither yet distinguished -- would need a different init strategy (e.g. nearest-neighbor instead of averaging, per the Phase 5 ablation already planned) or a non-generation diagnostic (e.g. teacher-forced accuracy on new-token positions specifically) to tell apart. **Revised implication for Phase 1: a vocabulary-growth cap does NOT fix Vocab Expansion's underperformance in this pilot** -- this is a more fundamental open question about the method than a data-scaling artifact, worth resolving before committing Vocab Expansion to the full grid at its current design.

       **Possible third explanation, from Abagyan et al. (2026), added 2026-09-23:** their vocabulary-coverage gains depended on the new-language tokens being in the tokenizer *from the start of pretraining*, with gradient exposure across the whole run. Vocab Expansion adds tokens *after* pretraining is finished, with only average-embedding init and a short fine-tune. If meaningful use of new tokens needs pretraining-scale exposure, then no post-hoc init trick would close the gap -- which would fit the flat frequency-cap results better than either explanation above. Not tested; it is an interpretation drawn from a different setting (LLM pretraining, not NLLB fine-tuning), so treat it as a hypothesis, not a finding. The nearest-neighbor-init ablation already planned in Phase 5 would help separate it from the init explanation.

     - **25-epoch results, all six conditions, 2026-09-24/28 (jobs 13878734/35, 13885150-53):**

       | Direction | Standard | Morfessor-VC | Vocab Expansion |
       |---|---|---|---|
       | es->arn, 10ep | 44.79 | 44.67 (-0.12) | 43.41 (-1.38) |
       | es->arn, 25ep | 46.66 | **46.67 (+0.01)** | 45.81 (-0.85) |
       | arn->es, 10ep | 47.67 | 46.48 (-1.19) | 45.67 (-2.00) |
       | arn->es, 25ep | 48.85 | 47.94 (-0.91) | 47.81 (-1.04) |

       Longer training narrows every gap, in some cases substantially (Vocab Expansion's arn->es deficit roughly halved), confirming the two-correction undertraining finding above applies across the board, not just to Vocab Expansion. **es->arn: Morfessor-VC and Standard are now statistically indistinguishable by eye** (46.67 vs 46.66, closer than at 10 epochs) -- consistent with the target-side tie being real, not an undertraining artifact. BLEU disagrees slightly here (Standard 11.01 vs VC 10.83), a metric-disagreement flag noted but not resolved. **arn->es: Standard still leads over both new methods at 25 epochs.** This is the direction that bears on the AmericasNLP paper's source-side claim -- the gap shrank (VC: -1.19 -> -0.91) but did not close, so on our pipeline Morfessor-VC still does not beat Standard here even after fixing the undertraining problem. This does not yet resolve the checkpoint-audit question below (the paper's own arn->es Standard checkpoint has not been retrained with the fixed script); it only confirms the pilot's own 10-epoch ranking on this direction was not an undertraining artifact. **Significance, first pass (jobs 13906422/23) -- tested the wrong statistic, now being redone.** That bootstrap (like the earlier Phase 0 one) resampled *mean sentence-level chrF*, which weights every sentence equally, while every table here reports *corpus* chrF, which weights by length. They disagree for arn->es: mean sentence-chrF gives Standard 48.17 vs Morfessor-VC 48.03 (diff -0.14, 95% CI [-0.35, +0.05], n.s.), while the corpus gap is -0.91. es->arn VC vs Standard: -0.16 [-0.36, +0.05], n.s.; Vocab Expansion significantly worse in both directions (-1.21 es->arn, -0.87 arn->es, both p<0.001). The mismatch matters: on the saved 10-epoch arn->es predictions, a corpus-level paired bootstrap (Koehn 2004; resample sentences, recompute corpus chrF from summed sufficient statistics -- verified to reproduce sacrebleu's corpus scores exactly) gives Standard minus VC = 1.17, 95% CI [0.99, 1.35], clearly significant, where the sentence-mean test was not. So the arn->es VC deficit is concentrated in longer sentences, and which statistic is used changes the conclusion. Script fixed to save predictions and report both; 25-epoch comparisons rerunning (jobs 13907698/99). **Going forward, significance should be tested on the same statistic that is reported (corpus chrF); the pre-registered "bootstrap resampling" should be read that way.** The earlier es->arn 10-epoch results in this section used the sentence-mean statistic too.

     - **arn->es direction and retrieval baseline, 2026-09-23** (single seed each; no bootstrap yet -- `phase0_predict_and_bootstrap.py` is hardcoded to es->arn):

       | Direction | Standard BPE | Morfessor-VC | Vocab Expansion |
       |---|---|---|---|
       | es->arn (target side) | 44.79 chrF / 9.95 BLEU | 44.67 / 9.90 | 43.41 / 8.80 |
       | arn->es (source side) | 47.67 / 18.98 | 46.48 / 18.40 | 45.67 / 17.93 |

       **Standard BPE is best in both directions.** On the source side (arn->es), where the AmericasNLP paper reported Morfessor-VC helping, it is 1.19 chrF *below* Standard here, and Vocab Expansion is 2.00 below. So this pilot does not reproduce the source-side advantage. Caveats before reading much into that: one seed, no significance test yet, and this pilot is not a like-for-like rerun of the paper (regenerated data, 56,197 vs. the paper's 55,452 training pairs; hyperparameters and model size should be checked against the paper's 600M numbers, which exist in `mapudungun-mt` and have not been compared yet). On Hypothesis 2: no method produces a benefit in either direction, so its premise (that there is a benefit to be larger on one side) isn't met here; the gaps happen to be smaller on the target side (-0.12) than the source side (-1.19), which is the opposite of the paper's story about where VC helps.

       **Cross-check against the AmericasNLP paper's own checkpoints, 2026-09-23** (`scripts/phase0_crosscheck_paper.py`, job 13874830, `results/phase0_crosscheck_paper.json`; all four systems arn->es on the same regenerated 9,367-sentence test set, same metric, chrF++ = `word_order=2`, which is what the paper's table uses; the training script's own logged chrF is `word_order=0` and runs ~2.3 points higher, so the earlier 47.67 was not comparable to the paper's table):

       | System | chrF++ | BLEU | Paper's reported chrF++ |
       |---|---|---|---|
       | Paper's Standard BPE 600M (its own checkpoint) | 36.21 | 11.31 | 34.90 |
       | Paper's Morfessor-VC 600M (its own checkpoint, input segmented with the paper's original Morfessor model + cascade) | 43.22 | 17.51 | 43.16 |
       | Ours, Standard BPE 600M | 45.36 | 18.95 | |
       | Ours, Morfessor-VC 600M | 44.24 | 18.38 | |

       **What this establishes:** the evaluation pipeline and test set are sound (the paper's checkpoints reproduce its table: VC 43.22 vs. 43.16; Standard 36.21 vs. 34.90, a 1.3-point difference unexplained). The disagreement is about the *Standard* baseline, not the method: Morfessor-VC scores similarly in both setups (43.2 paper, 44.2 ours), while Standard differs by 9.2 points on identical test data and metric. Our 600M Standard (45.36) is above the paper's 3.3B Standard (42.86) and about level with the paper's 3.3B cascade (45.84).

       **What it does not establish -- the cause.** Candidates: (a) the paper's Standard arn->es run was under-trained or otherwise anomalous; (b) something in this pilot's pipeline (regenerated training data, ~1% different size) favors Standard. Inference, not a finding: because VC agrees across setups and only Standard moves, a training-data difference (which should shift both) seems less likely than a problem specific to the paper's Standard run. Context from the paper's own `notes/research_log.md`: those runs had a documented wrong-language-label bug, stale single-file checkpoints, and retrains, and the 600M Standard arn->es number went 35.71 -> 34.90 through corrections; the final 600M Standard checkpoint dir has no `run_config.json`, so its epochs/learning rate can't be checked. The paper's own table also shows the 600M Standard row as an outlier: its es->arn is 39.87 (in line with the other conditions) but its arn->es (34.90) sits 7-8 points below every other 600M condition (42.2-43.2).

       **Why it matters -- this bears on published claims, so stated plainly.** The source-side advantage of Morfessor-VC in the AmericasNLP paper (+8.3 chrF++ at 600M, +4.7 at 1.3B, +3.0 at 3.3B) is measured against Standard baselines that, if this cross-check generalizes, may be weak; and the paper's "600M cascade beats 3.3B standard" claim depends on the 3.3B Standard number. thesis_proposal.md's Problem section currently states Morfessor-VC "improved Mapudungun->Spanish" and that its advantage "shrinks... at scale" -- both rest on those baselines. Not yet checked for 1.3B/3.3B, and the "shrinks at scale" pattern could partly be baselines becoming less under-trained as models grow. This is also the failure mode red-team item 9 anticipated (unequal effective training across conditions), showing up in the prior paper rather than the new grid. **Open, not resolved:** (1) does the paper's es->arn Standard (39.87) show a similar gap against ours (needs es->arn chrF++ for ours, not yet computed; script currently arn->es only); (2) does the gap hold at 1.3B/3.3B under the current pipeline; (3) what the paper's Standard run actually used. No thesis_proposal.md or paper edits proposed until the user decides how to handle it.

     **Retrieval-proxy baseline** (`scripts/phase0_retrieval_baseline.py`, after Chen et al. 2026; nearest training neighbor by TF-IDF character-n-gram similarity on the source side, not their BLEU/chrF similarity): es->arn corpus chrF 19.01 / BLEU 1.14; arn->es 19.41 / 2.15. Every fine-tuned condition beats it by roughly 24-28 chrF, so the pilot scores reflect real translation, not train-test memorization. The test split is file-level, which is consistent with the low retrieval score.

       **Second seed (43), partial:** Morfessor-VC es->arn came in at 44.78 chrF / 9.92 BLEU vs. 44.67 / 9.90 at seed 42, i.e. seed-to-seed spread of about 0.1 chrF for this condition (one pair only). The seed-43 Standard and Vocab Expansion runs failed with CUDA out-of-memory on `cs-1-2` (another process held ~75 GB of the shared GPU) and were resubmitted excluding that node (jobs 13874212/13874213); three-seed picture below.

     - **Seed 43 complete (reruns succeeded), es->arn test chrF / BLEU:** Standard 44.82 / 10.02 (seed 42: 44.79 / 9.95), Morfessor-VC 44.78 / 9.92 (44.67 / 9.90), Vocab Expansion 43.30 / 8.87 (43.41 / 8.80). Seed-to-seed spread is at most 0.11 chrF in every condition, and the ranking is identical across seeds. So the Vocab Expansion deficit (~1.4-1.5 chrF) is roughly 13x the seed noise, and the Standard-vs-Morfessor-VC tie is robust too. Two seeds only, so this bounds the noise rather than characterizing it.

     - **One concrete difference in training recipe found while chasing the cause (2026-09-23).** The paper's log and the surviving `training_args.bin` of its Morfessor-VC and es->arn checkpoints all give batch 32 x grad-accum 4 = effective batch 128, 10 epochs, lr 3e-5, warmup 1000; the pilot used batch 16 x 4 = 64 with the rest identical, i.e. twice the optimizer updates for the same 10 epochs. The paper's Standard arn->es checkpoint itself has no `training_args.bin` (only the published copy survives), so its own setup can't be confirmed. A weaker baseline from fewer updates is one candidate; note it doesn't obviously predict why the paper's es->arn Standard (39.87) is not weak relative to its Morfessor-VC (39.89). Open, not a conclusion.

     - **Code/history audit of the paper's runs (2026-09-23), what it does and doesn't show.** Established from files and git: the paper's arn->es *Standard* checkpoints (600M/1.3B/3.3B) are dated 2026-03-01/02, before commit 4ab3b3a (2026-03-09, "set tokenizer.tgt_lang in preprocess fn") and ff7bff3 (same day, es->arn direction fix); the 600M one's config records transformers 5.2.0. Its arn->es *Morfessor-VC* checkpoints (2026-03-16) and its retrained es->arn Standard runs (2026-03-09/10) record transformers 4.57.6 and used the fixed script. So the paper's arn->es Standard and Morfessor-VC runs were trained under different library versions and script versions -- a confound in the comparison, independent of whether it affected the numbers. This also explains why the paper's es->arn Standard is not weak (retrained after the fix), removing the puzzle noted above. **Hypothesis tested and refuted:** that the pre-fix script left the training labels with a wrong target-language token, so the Standard model expects a different first token than the `spa_Latn` forced at test time (`scripts/probe_forced_bos.py`, job 13877794, first 300 test sentences, chrF++). The paper's Standard checkpoint generates `spa_Latn` first on 300/300 sentences even with nothing forced (35.08, identical to forcing `spa_Latn`), and forcing `eng_Latn`/`arn_Latn` only lowers it (24.83 / 31.42); the post-fix controls behave the same (paper VC 39.95, ours 41.61 with `spa_Latn`). So there is no language-token mismatch. **CORRECTION 2026-09-28: that "refutation" was invalid.** The probe's "none" condition passed no `forced_bos_token_id`, but each checkpoint's saved `generation_config` already forces `spa_Latn` (the training script sets it before saving), so "none" still forced `spa_Latn` -- the 300/300 result says nothing about what the model learned. **Direct replay of the training preprocessing (2026-09-28) finds the actual mechanism:** running the pre-fix script's `build_preprocess_fn` (commit f1f06dc, no `tokenizer.tgt_lang`) under transformers 5.2.0 (scratch venv `/tmp/tf52`; the version those three checkpoints record, and the one an unpinned install resolved to on 2026-03-01 -- `uv.lock` pinned 4.57.6 only on 2026-03-06) gives, for every example, **input = `<text> </s> <unk>` with no `arn_Latn` tag, and labels = `<text> </s> <unk>` with no `spa_Latn` tag**; 5.2.0 loads NLLB as a generic `TokenizersBackend` that ignores `src_lang`. The fixed script under 4.57.6 gives the correct `arn_Latn <text> </s>` / `spa_Latn <text> </s>`, and the pre-fix script under 4.57.6 crashes (matching commit 4ab3b3a's message). So the paper's three arn->es Standard baselines were trained on untagged, `<unk>`-suffixed data and then evaluated with tagged input and a forced `spa_Latn` first token -- a train/eval format mismatch that the other 31 checkpoints don't have. **Decisive test running** (job 13908121, `scripts/paper_baseline_format_test.py`): score each of the three checkpoints in the published format and in the training format.

       **256-token rerun result (2026-10-02), Phase 0 pilot, 600M, 25 epochs, single seed; corpus chrF vs full references, corpus-level paired bootstrap (jobs 13912818-21 trained, 13952090/91 scored; `results/phase0_bootstrap_*_e25_len256.json`):** es->arn Standard 46.33 vs Morfessor-VC 46.62, **VC +0.29 [+0.07, +0.48], significant** (at 128 tokens: VC -0.64 [-0.85, -0.45]). arn->es Standard 48.93 vs VC 48.44, VC -0.49 [-0.66, -0.32] (at 128: -0.91). Raising the limit moves Standard by ~0.1 but VC by +1.0 (es->arn) and +0.5 (arn->es), confirming the 128-token cap biased against the segmented condition. **With the bias removed, Morfessor-VC is slightly but significantly better than Standard on the target side (es->arn) and slightly worse on the source side -- the pattern Hypothesis 2 predicts, and the reverse of the AmericasNLP paper's story.** Small effect, one seed. **BLEU disagrees on es->arn** (same predictions): Standard 10.93 vs VC 10.77, while chrF++ gives VC +0.16 (42.46 vs 42.30) and chrF +0.29; arn->es all metrics agree Standard is better (BLEU 19.62 vs 19.04). So the target-side VC edge is metric-dependent and should be read as "VC no longer loses on the target side once truncation is fixed," not as a clear win. Second seed queued (jobs 13952224-27, `..._len256_s43`). Implication: all Phase 0/1 comparisons involving segmented conditions should train and decode at 256 tokens (or longer), and the earlier statement that VC "doesn't beat Standard in either direction" holds only at 128 tokens.

       **METRIC CORRECTION (2026-10-06) -- the 2026-09-29 result below overstates the reversal.** The v2 Phase A audit (`mapudungun-mt/scripts/analysis/v2_phaseA_audit.py`) found, and I verified, that the paper's tables report the *mean of sentence-level chrF++*, not corpus chrF++ (600M arn->es Standard: table 34.90 = sentence-mean 34.90; corpus 35.71). The paper's significance test bootstraps the same mean, so the paper is internally consistent; my earlier claim that its tests "tested a different statistic than reported" was wrong and is retracted. Re-checking the corrected comparison on the paper's own metric (sentence-mean chrF++, same saved predictions, sentence-mean paired bootstrap n=1000): Standard (training format) vs the paper's Morfessor-VC is **600M 43.72 vs 43.76 (VC +0.05 [-0.16, +0.26], tie), 1.3B 45.74 vs 45.79 (+0.05 [-0.17, +0.28], tie), 3.3B 46.43 vs 46.09 (-0.33 [-0.55, -0.07], Standard better)**. 600M VC vs 3.3B Standard: -2.66 [-2.94, -2.37]. So the accurate statement is: **the paper's arn->es advantage for Morfessor-VC (+8.3/+4.7/+3.0) disappears -- tie at 600M and 1.3B, Standard slightly ahead at 3.3B on the paper's metric, and Standard ahead at all three sizes on corpus chrF++ -- and the "5x parameters" claim fails on both metrics.** Which metric v2 uses is the user's call; the two can disagree in sign on small gaps.

       **RESULT (2026-09-29): confirmed -- the paper's arn->es Standard baselines were mis-evaluated, and correcting it reverses the paper's arn->es conclusions.** All numbers: our regenerated 9,367-sentence test set, chrF++ (corpus), beam 4, max length 128 (the paper's own script used greedy and 256, so absolute values differ slightly from its table; both formats of each checkpoint use identical settings). Paper VC checkpoints scored the same way (`scripts/paper_vc_same_eval.py`, job 13908451), significance by corpus-level paired bootstrap on saved predictions (`results/paper_corrected_significance.json`).

       | Size | Standard, as published (tagged input, forced spa_Latn) | Standard, in its training format | Paper's Morfessor-VC | VC minus Standard (training format) |
       |---|---|---|---|---|
       | 600M | 36.18 | **43.95** | 43.22 | -0.73 [-0.92, -0.54] |
       | 1.3B | 42.74 | **46.01** | 45.19 | -0.82 [-0.99, -0.64] |
       | 3.3B | 44.42 | **46.64** | 45.62 | -1.01 [-1.21, -0.83] |

       Scored in the format they were trained in, the paper's own Standard baselines beat its own Morfessor-VC at every size, significantly (p<0.001 each). The paper reports the opposite (+8.3 / +4.7 / +3.0). The abstract's headline "morphology-aware tokenization can substitute for a 5x increase in model parameters" (600M VC matches 3.3B Standard, +0.31, p=0.039) does not hold: 600M VC is 3.42 chrF++ *below* the correctly-scored 3.3B Standard (CI [-3.67, -3.19]). Affected in the paper: abstract, contribution bullet, arn->es results paragraphs and table row, the significance-test claims against Standard, the Standard rows of the BLEU and COMET appendix tables, and the arn->es half of the human evaluation (its Standard outputs came from the mis-evaluated 3.3B checkpoint, and its sentence sample was stratified on that model's scores). Not affected: es->arn (Standard retrained post-fix; paper already found no method beats Standard there), and the comparisons among morphology-aware methods (all post-fix). Root cause: the pre-fix training script under the unpinned transformers 5.2.0 (2026-03-01) produced untagged, `<unk>`-suffixed training examples; the fix and the 4.57.6 pin came later, and the three arn->es Standard runs were never retrained because only es->arn was known to be broken. Consistent with this thesis's own pilot, which found Morfessor-VC does not beat a correctly trained Standard in either direction. Earlier text follows, kept for the record: **Still open:** whether the transformers-5.2.0-vs-4.57.6 difference (e.g. how the source side was tokenized) or the effective-batch difference or simply run-to-run variation made that Standard run weak; the batch-128 rerun (13877589/90) tests the batch part only.

     - **Checkpoint audit of the whole paper release, 2026-09-24 (`/nobackup/archive/usr/it238/mapudungun_final_models`, 34 checkpoints, training date from file mtime, library from each `config.json`).** Exactly three checkpoints predate the 2026-03-09 script fixes and record transformers 5.2.0: **the arn->es Standard BPE runs at 600M, 1.3B and 3.3B** (2026-03-01/01/02). Every other checkpoint -- all es->arn runs and every other arn->es method -- is post-fix and records 4.57.6. So the paper's arn->es Standard baseline, the row its source-side claims are measured against, was trained in a different environment from every method compared to it. In the paper's table that row is the lowest arn->es score at every size (34.9 / 40.6 / 42.9 vs. ~42-46 for all other methods), the 600M value is an outlier within its own table, and in es->arn (Standard retrained post-fix) no method beats Standard (39.87 vs. 38-40). The "600M cascade > 3.3B standard" claim (+0.31, p=0.037 in the paper's log) uses one of the three as its baseline. **Status: a documented confound, cause not established, and the conclusion not yet shown to change.** The pilot's own comparison (arn->es Standard ~1 chrF ahead of Morfessor-VC, at both batch sizes) is at 10 epochs, now known to be undertrained; the matched 25-epoch arn->es runs (13885151-53) are the fair test. Not checked: the paper's text/claims beyond the results table, human evaluation, COMET, the code-switching analysis, or 1.3B/3.3B re-scoring. Suggested next step: retrain the three arn->es Standard baselines with the fixed script, then decide on disclosure (advisor and co-authors first).
     - **Batch-128 rerun result (2026-09-24): batch size does NOT explain the weak paper baseline.** Same pipeline, paper's exact batch config (32 x 4 = 128), arn->es test chrF / BLEU: Standard 46.70 / 18.19 (vs. 47.67 / 18.98 at batch 64), Morfessor-VC 45.49 / 17.58 (vs. 46.48 / 18.40). Both drop ~1 point, Standard stays ~1.2 ahead of VC, and neither approaches the paper's Standard (~36 chrF++, roughly 38 in plain chrF on this test set). Note these are plain chrF; on this test set chrF++ runs about 2.3 lower (measured on the pilot's own Standard: 47.62 vs. 45.36), so Standard at batch 128 is roughly 44.4 chrF++ (an estimate from that offset, not a direct measurement). With the recipe difference and the wrong-language-token idea both eliminated, what remains for the paper's weak Standard arn->es run is the library/script version it was trained under (transformers 5.2.0, pre-fix script), something else specific to that run, or a dataset difference at the time -- none testable without that run's records. Dev curves for the batch-128 runs were still creeping up at epoch 10 (Standard 48.2 / 48.4 / 48.5), i.e. slightly less converged, consistent with fewer updates.

     - **Decisive test launched 2026-09-23:** retrain Standard and Morfessor-VC (arn->es) on this pipeline with the paper's exact recipe, effective batch 128 (jobs 13877589 / 13877590, `experiments/models/phase0_pilot/nllb-600M-arn-es-{standard,morfessor_vc}_bs128`). If Standard falls to ~36 and VC stays ~43, the gap is a training-budget effect; if Standard stays ~45, the cause is elsewhere (data pipeline or something in the paper's Standard run).
10. **Vocabulary-size confound (Morfessor vs. BPE at "the same" merge count) → report vocab size as a covariate.** For every trained tokenizer (all methods, all languages), log final vocabulary size alongside the results table, so a reviewer can check whether performance differences track vocabulary size rather than the segmentation strategy itself. No new runs needed, just an additional logged statistic per existing run.

Items 1 and 9 are the only two that cost additional compute; everything else is design/reporting discipline. Recommend resolving 3, 4 (the lookup), 7, and 8 during Phase 0 since they're free and fast; decide on 1 and 9's added compute once the Phase 0 language's actual corpus size and per-run wall-clock time are known.

**Hyperparameter grid — pre-registered 2026-09-15, before any Phase 0/1 runs launch (closes the "item 12" open decision from the initial-probes list).** No prior Optuna config exists to inherit from the Mapudungun paper's own code (checked — the "Optuna BPE" baseline referenced there has no surviving script), so these are set fresh rather than copied:
- **Vocab-biased BPE's λ:** Optuna TPE sampler, log-uniform search over **[0.1, 10.0]**, **20 trials per language**. This step is CPU-only (BPE merge scoring, no GPU), so 20 trials is cheap regardless of Phase 0's GPU-hour findings — no reason to under-search it just because the GPU budget is still unsettled.
- **Per-run GPU-hour budget (resolves red-team item 9, line ~435):** pre-register the undertraining check at exactly **15 GPU-hours/run (3× the original 5h assumption)**, 3 conditions (Standard BPE, Morfessor-VC, Vocab Expansion) on the Phase 0 language — **~45 GPU-hours total**, as the red-team originally scoped it, rather than leaving the multiplier open. Decision rule pre-committed now: if method rankings at 5h and 15h agree, keep 5h/run for the full Phase 1 grid; if they disagree, 15h/run becomes the standard for all of Phase 1 (recompute the Compute Estimate table accordingly before launching).
- **Morfessor's own hyperparameters:** fixed at the library default corpus-weight (α = 1.0) uniformly across every language, not tuned per-language. This matches how Morfessor-VC was already run in the Mapudungun paper, and — more importantly — keeps it a controlled variable across the 12-language typology grid rather than an uncontrolled source of per-language variation that could masquerade as a typological effect. Per-language Morfessor tuning is explicitly out of scope for the primary grid; if pursued at all, it's a Phase 5 ablation (line ~208), reported separately, never folded into the main results table.

---

## Open Questions / Decisions Needed

- [ ] **Final language list (Nahuatl/Inuktitut):** recommendation given above under "Polysynthetic notes" — keep both, evaluate on project-specific test sets (Axolotl/OPUS, Nunavut Hansard) since neither is in FLORES-200, disclose as a Limitations item. Still open: actually auditing corpus size/quality for each before Phase 1 commits compute.
- [ ] **Toolkit name — recommendation: `morphotok`.** Short, descriptive without being cryptic, reads naturally as a pip package name (`pip install morphotok`, `from morphotok import MorphoTokenizer` — matches the API sketch already in this doc). Check PyPI/GitHub namespace availability before committing; `vamt` and `morphalign` are the fallback candidates already listed above if it's taken.
- [x] Whether to include mT5 or Aya as a third model — **revised 2026-09: both are Tier 3 stretch goals**, superseding the 2026-08-19 "both, committed for ARR" decision. See Model Scope above for reasoning and the reduced-subset scoping if pursued.
- [ ] **Human evaluation languages and annotators (revised 2026-09):** minimal eval needs one confirmed annotator per typological category (Mapudungun confirmed for polysynthetic; agglutinative/fusional/analytic slots still need an actual audit, not just the assumption that "Turkish and Finnish are easy to staff" — verify against BYU's actual language population before committing).
  - **2026-09-16: PathSay identified as a real annotator pool for 2 of the 4 slots.** Sam Bird's current PathSay language list (updated daily, ~100 languages, Church/scripture-domain parallel data) includes Swahili (67,926 rows, agglutinative slot) and Yoruba (35,210 rows, analytic slot) -- both already in the 12-language grid, no language swap needed for those two. Fusional (Icelandic/Russian) and the Turkish/Finnish agglutinative option remain uncovered by PathSay entirely. Two things pending before this closes: (1) confirming these are actual available annotators for preference-ranking/MQM-lite tasks, not just evidence that translated corpora exist -- getting an account (via Joseph, per Sam's suggestion) to check; (2) checking whether PathSay annotators can meaningfully judge general-domain (FLORES-200-style) text given their work is Church/scripture-domain, a domain-mismatch caveat in the same category as Nunavut Hansard's legislative-domain Inuktitut data (see Language Selection section).
  - **If confirmed, recommendation: promote Extensive human eval (currently Tier 3) to Tier 2, scoped specifically to Swahili and Yoruba** (real multi-annotator IAA, broader sentence sample) rather than opening it to all 12 languages -- bounded to where the annotator pool actually exists, doesn't silently expand scope elsewhere. Icelandic/Russian/Turkish/Finnish stay at Minimal.
  - **Also open: whether good PathSay-represented substitutes exist for any of the 12 core grid languages.** User is getting a PathSay account to see the live/updated language list before deciding; if a strong typologically-equivalent substitute turns up (matching morphological type, resource level, and NLLB-200/FLORES-200 coverage the way the current 12 were each individually verified), a swap is worth considering -- but per the earlier discussion, a swap needs to clear the same bar the current languages already cleared (WALS classification, FLORES-200 coverage, resource-level check), not just annotator convenience alone.
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
| High (>5M) | Turkish, Finnish, Icelandic, Russian |

Transfer pairs to test: Quechua → Mapudungun, Turkish → Uzbek (if in scope), Icelandic → Faroese, Finnish → Estonian.

### Key Claim

Typological similarity is a better predictor of useful transfer than geographic proximity or script similarity — and this holds even controlling for resource level. If confirmed, this gives practitioners a principled way to choose a transfer source language without exhaustive search.

### Venue Target

ARR cycle following Paper A submission. Pipeline and language data already built; incremental compute cost only for augmentation and transfer experiments.
