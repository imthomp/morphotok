"""Morfessor-VE (Vocabulary Expanded).

Run Morfessor; for morphemes not in the pretrained vocabulary, add them as
new tokens with embeddings initialized by averaging their constituent
subword-token embeddings. Addresses both source (better input segmentation)
and target (model can generate morpheme tokens directly) sides, unlike
Morfessor-VC.

See notes/plan.md, "New Methods".
"""

from __future__ import annotations


class MorfessorVE:
    def __init__(self, model_vocab: set[str]) -> None:
        self.model_vocab = model_vocab

    def fit(self, monolingual_corpus):
        raise NotImplementedError("Train/load a Morfessor model, then identify OOV morphemes to add.")

    def new_tokens(self) -> list[str]:
        """Morphemes to add to the model's vocabulary, with average-embedding init."""
        raise NotImplementedError

    def segment(self, word: str) -> list[str]:
        raise NotImplementedError
