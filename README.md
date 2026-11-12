# morphotok

Vocabulary-aligned morphological tokenization for multilingual MT.

Code and research planning for the thesis *When Does Morpheme-Aware
Tokenization Help? A Predictive Framework Across Language Types*
(Isaac Thompson). Full research design lives in `notes/thesis_proposal.md`
(the shareable proposal) and `notes/plan.md` (the fuller working plan,
including the GLM-5.2 red-team disposition and open decisions).

**Status: paused (November 2026).** Phase 0 pilot experiments (NLLB-200
fine-tuning on Mapudungun<->Spanish and Inuktitut<->English) are complete and
their outputs are in `experiments/results/`. These are preliminary research
artifacts: not peer reviewed, not yet written up. The `morphotok` package API
below is a prototype, and `method="auto"` is not implemented.

## Install

```bash
pip install -e .
```

## Usage

```python
from morphotok import MorphoTokenizer

tok = MorphoTokenizer(
    lang="tur",
    model="facebook/nllb-200-distilled-600M",
    method="auto",  # or "morfessor-vc", "morfessor-ve", "vocab-biased-bpe"
)
tokenized = tok.tokenize(corpus)
```

`method="auto"` recommends a tokenization strategy from vocabulary density
alone, no training required, once the underlying predictive framework
(plan.md Phase 4) is validated. Falls back to `morfessor-vc` otherwise.

## Layout

- `src/morphotok/` — the installable package (Morfessor-VC, Morfessor-VE, Morfessor-VM, vocab-biased BPE)
- `tests/` — unit tests for the package
- `experiments/` — research code, not part of the package:
  - `scripts/` — data prep, NLLB-200 fine-tuning (`finetune_nllb_pilot.py`), scoring and paired bootstrap (`phase0_bootstrap_general.py`), Inuktitut syllabics transliteration and coverage conditions, density/fertility analyses
  - `slurm/` — the SLURM job scripts used to run them (BYU cluster paths)
  - `data/` — small raw inputs (language lists, NLLB resource levels)
  - `results/` — JSON outputs written by the scripts
  - Gitignored, local only: `models/` (checkpoints), `data_pipeline/` (corpora), `logs/`
- `notes/` — research planning docs

## Reproducing the experiments

- Fine-tuning needs `transformers` 4.x (tested 4.57.6). transformers 5.x loads the
  NLLB tokenizer as a generic backend that ignores `src_lang`, producing training
  examples without language tags. `finetune_nllb_pilot.py` asserts the tags are present.
- Corpora are not redistributed: the Mapudungun data comes from the AmericasNLP pipeline, and
  Inuktitut from the Nunavut Hansard 3.0 release (NRC, DOI 10.4224/40001819).
- The SLURM scripts hard-code cluster paths and need editing elsewhere.

## Methods

- **Morfessor-VC** (Vocabulary Constrained) — constrains Morfessor segmentation to the pretrained model's existing vocabulary
- **Morfessor-VE** (Vocabulary Expanded) — adds OOV morphemes as new tokens with average-embedding initialization
- **Morfessor-VM** (Vocabulary Maximized) — merges out-of-vocabulary Morfessor pieces with their neighbors until every piece is in the existing vocabulary; adds no new tokens
- **Vocab-biased BPE** — upweights BPE merges that produce in-vocab tokens

## License

MIT, see `LICENSE`.
