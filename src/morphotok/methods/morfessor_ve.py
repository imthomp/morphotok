"""Morfessor-VE (Vocabulary Expanded).

Run Morfessor; for morphemes not in the pretrained vocabulary, add them as
new tokens with embeddings initialized by averaging their constituent
subword-token embeddings. Addresses both source (better input segmentation)
and target (model can generate morpheme tokens directly), unlike
Morfessor-VC which only constrains segmentation to the existing vocab.

See notes/plan.md, "New Methods".

Embedding initialization (averaging constituent subword-token embeddings for
each new morpheme) is a model-loading-time step, not a tokenizer-segmentation
step, and belongs in the training pipeline that consumes this class's
new_tokens() output -- this class only determines segmentation and which new
tokens are needed, matching where methods/vocab_biased_bpe.py and
morfessor_vc.py draw the same line (models don't own embeddings here).
"""

from __future__ import annotations

import morfessor


class MorfessorVE:
    def __init__(self, model_vocab: set[str]) -> None:
        self.model_vocab = model_vocab
        self._model: morfessor.BaselineModel | None = None
        self._new_tokens: set[str] | None = None

    def fit(self, monolingual_corpus) -> None:
        words = [w for w in monolingual_corpus if w]
        model = morfessor.BaselineModel()
        model.load_data([(1, w) for w in words])
        model.train_batch()
        self._model = model

        oov: set[str] = set()
        for w in words:
            for i, piece in enumerate(model.viterbi_segment(w)[0]):
                marked = f"▁{piece}" if i == 0 else piece
                if marked not in self.model_vocab and piece not in self.model_vocab:
                    oov.add(marked)
        self._new_tokens = oov

    def new_tokens(self) -> list[str]:
        """Morphemes not in the pretrained vocabulary, to be added as new
        tokens with average-embedding init. Word-initial occurrences carry
        the '▁' marker (matching SentencePiece convention), so the same
        surface morpheme can appear twice if it's attested both word-initially
        and word-internally as OOV.
        """
        if self._new_tokens is None:
            raise RuntimeError("Call fit() before new_tokens().")
        return sorted(self._new_tokens)

    def segment(self, word: str) -> list[str]:
        if self._model is None:
            raise RuntimeError("Call fit() before segment().")
        # After vocabulary expansion every Morfessor-predicted piece is
        # in-vocabulary by construction (OOV pieces were added via
        # new_tokens()), so segmentation is just Morfessor's own output --
        # no vocab-constrained merging needed, unlike Morfessor-VC.
        return list(self._model.viterbi_segment(word)[0])
