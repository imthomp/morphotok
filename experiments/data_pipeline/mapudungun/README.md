# Mapudungun training data (regenerated 2026-09)

The original processed/cleaned/tokenized Mapudungun training data was
auto-deleted from BYU's `nobackup/autodelete` scratch storage (see plan.md's
GLM-5.2 red-team disposition, item 7). This directory is the regenerated
replacement, reproduced from the raw AVENUE corpus using the lab's own
pipeline, not copied from anywhere.

**Reproduction steps:**
1. Extract parallel Mapudungun-Spanish text from `mapudungun-corpus/translation-clean/*.txt`
   (M:/C: transcript-translation pairs) using `mapudungun-mt/scripts/data/extract_to_parallel.py`.
2. Split by file according to `mapudungun-corpus/dataset_splits/mt/{training,dev,test}_files.txt`
   (the same official split the original paper used; matched 333/343 files by
   filename, ~97%, the remainder didn't have an exact match, not yet
   investigated).
3. Clean each split with the BYU MT lab's shared data-cleaning-pipeline
   (`grp_mtlab/projects/data-cleaning/data-cleaning-pipeline`), using the
   existing `arn-CL.yaml` config already deployed there (permissive
   character set, `min_num_words: 1`, no long-word filtering, both choices
   specific to polysynthetic Mapudungun where a single word can be a full
   clause).

**Result vs. the original paper's published numbers:**

| Split | Regenerated here | Paper (Thompson et al., 2026) |
|---|---|---|
| Train | 56,197 | 55,452 |
| Dev | 1,586 | 1,581 |
| Test | 9,367 | 9,382 |

Within ~1% on every split. The small gap is most likely the ~3% of files
that didn't match during the split step (10 of 343), not a pipeline
difference; not yet root-caused.

**Files:** `cleaned/{train,dev,test}.{arn,es}`, plain text, one sentence per
line, aligned by line number between the `.arn` and `.es` files.
