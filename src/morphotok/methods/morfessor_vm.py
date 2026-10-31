"""Morfessor-VM (Vocabulary-Maximized).

Run Morfessor unsupervised morpheme segmentation, then greedily merge any
resulting piece not already in the pretrained model's vocabulary with its
neighbor, until every piece is in-vocabulary (or the whole word collapses to
a single fallback token). Unlike Morfessor-VC's opportunistic merging (which
only merges when a merge happens to land in-vocab, and can leave pieces OOV),
Morfessor-VM forces complete in-vocab coverage of its output using only the
existing vocabulary -- no new tokens are added (contrast Morfessor-VE, which
solves the OOV problem by expanding the vocabulary instead).

This was `MorfessorVC`'s implementation until 2026-09-16, when it was found
to not match the Mapudungun paper's actual published Morfessor-VC algorithm
(see morfessor_vc.py's docstring). Rather than discard it, it was split out
as its own method under a name that describes what it actually does -- see
notes/plan.md, "New Methods" (Morfessor-VM, added 2026-09-16, Tier 2).
"""

from __future__ import annotations

import morfessor


def _in_vocab(piece: str, is_word_initial: bool, vocab: set[str]) -> bool:
    if is_word_initial:
        return f"▁{piece}" in vocab or piece in vocab
    return piece in vocab


class MorfessorVM:
    def __init__(self, model_vocab: set[str]) -> None:
        self.model_vocab = model_vocab
        self._model: morfessor.BaselineModel | None = None

    def fit(self, monolingual_corpus) -> None:
        words = [w for w in monolingual_corpus if w]
        model = morfessor.BaselineModel()
        model.load_data([(1, w) for w in words])
        model.train_batch()
        self._model = model

    def segment(self, word: str) -> list[str]:
        if self._model is None:
            raise RuntimeError("Call fit() before segment().")
        segs = list(self._model.viterbi_segment(word)[0])
        if not segs:
            return segs

        while len(segs) > 1:
            oov_idx = None
            for i, piece in enumerate(segs):
                if not _in_vocab(piece, is_word_initial=(i == 0), vocab=self.model_vocab):
                    oov_idx = i
                    break
            if oov_idx is None:
                break
            if oov_idx < len(segs) - 1:
                segs[oov_idx : oov_idx + 2] = [segs[oov_idx] + segs[oov_idx + 1]]
            else:
                segs[oov_idx - 1 : oov_idx + 1] = [segs[oov_idx - 1] + segs[oov_idx]]
        return segs
