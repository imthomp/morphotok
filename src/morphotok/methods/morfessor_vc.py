"""Morfessor-VC (Vocabulary Constrained).

Run Morfessor unsupervised morpheme segmentation, then greedily merge
adjacent morpheme pairs (left-to-right within each word) whenever their
concatenation is itself already a single token in the pretrained model's
vocabulary -- regardless of whether the two original pieces were themselves
in- or out-of-vocabulary. This is the exact algorithm the Mapudungun paper's
own `segment_morfessor_vc.py` (Kevin Duh, JHU, 2026) validated and published
results from, not a re-derivation from the "Vocabulary Constrained" name.

**Important:** despite the name, this does NOT guarantee every output piece
ends up in-vocabulary -- a piece can remain OOV if no adjacent merge happens
to land on an in-vocab concatenation. See the "OOV coverage note" in
notes/plan.md's Methods/Baselines section: Morfessor-VC (like plain
Morfessor) needs a defined fallback for segments it can't cover, unlike
Standard BPE or Vocab-biased BPE.

(An earlier version of this class implemented a different, stricter
algorithm -- merge only OOV pieces with a neighbor, repeating until full
coverage or one piece remains -- which is NOT what the paper's code does,
and produced spurious whole-word-fallback cascades on rare/long words that
weren't a real property of Morfessor-VC, just an artifact of that wrong
algorithm. Corrected 2026-09-16.)

See notes/plan.md, "Methods" and "Existing (from Mapudungun paper)".
"""

from __future__ import annotations

import morfessor


class MorfessorVC:
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
        if len(segs) <= 1:
            return segs

        result = list(segs)
        i = 0
        while i < len(result) - 1:
            m1, m2 = result[i], result[i + 1]
            candidate = m1 + m2
            is_word_start = i == 0
            in_vocab = (f"▁{candidate}" in self.model_vocab) if is_word_start else False
            in_vocab = in_vocab or (candidate in self.model_vocab)
            if in_vocab:
                result[i] = candidate
                del result[i + 1]
                # don't advance i -- try merging the new combined token with the next
            else:
                i += 1
        return result
