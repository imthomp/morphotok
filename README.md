# morphotok

Vocabulary-aligned morphological tokenization for multilingual MT.

Companion code for the thesis *When Does Morpheme-Aware Tokenization Help?
A Predictive Framework Across Language Types* (Isaac Thompson). Full research
design lives in `notes/thesis_proposal.md` and `notes/plan.md` in the parent
[morphology-mt](https://github.com/imthomp) project directory, not in this repo.

**Status: early scaffolding.** Method implementations are stubs pending
Phase 0 results; API surface below is stable, behavior is not yet.

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

- `morphotok/` — the installable package (this is what `pip install morphotok` ships)
- `experiments/` — paper-reproduction scripts (data pipeline, training configs, eval), not part of the package
- `tests/` — unit tests for `morphotok/`

## Methods

- **Morfessor-VC** (Vocabulary Constrained) — constrains Morfessor segmentation to the pretrained model's existing vocabulary
- **Morfessor-VE** (Vocabulary Expanded) — adds OOV morphemes as new tokens with average-embedding initialization
- **Vocab-biased BPE** — upweights BPE merges that produce in-vocab tokens

## License

MIT, see `LICENSE`.
