"""Vocabulary-aligned morphological tokenization.

See notes/thesis_proposal.md and notes/plan.md (morphology-mt project root)
for the research design this package implements.
"""

from __future__ import annotations

import json
from pathlib import Path

from morphotok.methods import MorfessorVC, MorfessorVE, MorfessorVM, VocabBiasedBPE

SUPPORTED_METHODS = (
    "auto",
    "morfessor-vc",
    "morfessor-ve",
    "morfessor-vm",
    "vocab-biased-bpe",
)


def _load_vocab(model: str) -> set[str]:
    """Load the pretrained model's vocab as a set of surface tokens.

    `model` is either a local path to a HF `tokenizer.json`, or a directory
    containing one. HF Hub repo ids are not fetched here -- pass an
    already-downloaded local path (this project trains offline via
    HF_HUB_OFFLINE, see notes/plan.md's Compute Estimate/SLURM strategy).
    """
    path = Path(model)
    if path.is_dir():
        path = path / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"No tokenizer.json found for model={model!r}. Pass a local path to a "
            "downloaded HF tokenizer.json (see notes/plan.md SLURM strategy: "
            "pre-download weights, run offline)."
        )
    d = json.loads(path.read_text(encoding="utf-8"))
    return set(d["model"]["vocab"].keys())


class MorphoTokenizer:
    """Recommend and apply a vocabulary-aligned tokenization method for a language.

    Parameters
    ----------
    lang:
        Language code (e.g. "tur", "arn").
    model:
        Local path to a pretrained model's tokenizer.json (or a directory
        containing one) whose vocabulary tokenization should align to.
    method:
        One of SUPPORTED_METHODS. "auto" uses the vocabulary-density-based
        predictive framework (see plan.md Phase 4) to recommend a method
        without training; not yet available (Phase 4 hasn't run), so "auto"
        currently always raises -- pass an explicit method.
    """

    def __init__(self, lang: str, model: str, method: str = "auto") -> None:
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Unknown method {method!r}; expected one of {SUPPORTED_METHODS}")
        self.lang = lang
        self.model = model
        self.method = method
        self._resolved_method: str | None = None
        self._vocab: set[str] | None = None

    def resolve_method(self) -> str:
        """Return the concrete method to use, resolving "auto" via vocabulary density."""
        if self.method != "auto":
            return self.method
        raise NotImplementedError(
            "Auto-mode recommendation depends on the Phase 4 predictive framework "
            "(vocab density -> recommended method), not yet trained/validated. "
            "Pass an explicit method for now."
        )

    def _get_vocab(self) -> set[str]:
        if self._vocab is None:
            self._vocab = _load_vocab(self.model)
        return self._vocab

    def tokenize(self, corpus, **method_kwargs) -> list[list[list[str]]]:
        """Tokenize `corpus` (iterable of sentence strings).

        Returns one list of word-piece-lists per sentence, i.e.
        result[i][j] is the piece sequence for the j-th whitespace-delimited
        word of sentence i. The chosen method is fit once on the whole
        corpus's vocabulary of unique words, then applied per-word --
        matches how Morfessor/BPE are trained in the rest of this repo's
        scripts (experiments/scripts/*.py), not per-sentence.
        """
        resolved = self.resolve_method()
        vocab = self._get_vocab()
        sentences = [s for s in corpus]
        words = sorted({w for s in sentences for w in s.split()})

        if resolved == "morfessor-vc":
            method = MorfessorVC(vocab)
            method.fit(words)
        elif resolved == "morfessor-ve":
            method = MorfessorVE(vocab)
            method.fit(words)
        elif resolved == "morfessor-vm":
            method = MorfessorVM(vocab)
            method.fit(words)
        elif resolved == "vocab-biased-bpe":
            method = VocabBiasedBPE(vocab, lam=method_kwargs.get("lam", 1.0))
            method.fit(words, num_merges=method_kwargs.get("num_merges", 10000))
        else:
            raise NotImplementedError(f"Method {resolved!r} not yet implemented.")

        return [[method.segment(w) for w in s.split()] for s in sentences]
