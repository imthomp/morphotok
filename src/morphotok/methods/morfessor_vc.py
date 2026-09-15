"""Morfessor-VC (Vocabulary Constrained).

Run Morfessor unsupervised morpheme segmentation, then greedily merge any
resulting piece not already in the pretrained model's vocabulary with its
neighbor, until every piece is in-vocabulary. Constrains segmentation to the
existing vocab; established in the Mapudungun paper to help source-side only.

See notes/plan.md, "Methods" and "Existing (from Mapudungun paper)".
"""

from __future__ import annotations


class MorfessorVC:
    def __init__(self, model_vocab: set[str]) -> None:
        self.model_vocab = model_vocab

    def fit(self, monolingual_corpus):
        raise NotImplementedError("Train/load a Morfessor model, then apply vocab-constrained merging.")

    def segment(self, word: str) -> list[str]:
        raise NotImplementedError
