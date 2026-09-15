"""Vocab-biased BPE.

Modified BPE training that upweights merges producing tokens already present
in the pretrained model's vocabulary:

    score(merge) = freq(merge) * (1 + lambda * 1[result in pretrained_vocab])

lambda is a hyperparameter tuned per language. No Morfessor dependency;
purely BPE-based, so it retains BPE's universal coverage guarantee.

See notes/plan.md, "New Methods".
"""

from __future__ import annotations


class VocabBiasedBPE:
    def __init__(self, model_vocab: set[str], lam: float = 1.0) -> None:
        self.model_vocab = model_vocab
        self.lam = lam

    def fit(self, monolingual_corpus):
        raise NotImplementedError("Train a BPE model with the vocab-biased merge score.")

    def segment(self, word: str) -> list[str]:
        raise NotImplementedError
