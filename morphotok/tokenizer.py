"""Vocabulary-aligned morphological tokenization.

See notes/thesis_proposal.md and notes/plan.md (morphology-mt project root)
for the research design this package implements. Skeleton only for now;
methods are stubbed pending Phase 0 results.
"""

from __future__ import annotations

SUPPORTED_METHODS = (
    "auto",
    "morfessor-vc",
    "morfessor-ve",
    "vocab-biased-bpe",
)


class MorphoTokenizer:
    """Recommend and apply a vocabulary-aligned tokenization method for a language.

    Parameters
    ----------
    lang:
        Language code (e.g. "tur", "arn").
    model:
        Pretrained model name or local path whose vocabulary tokenization
        should align to (e.g. "facebook/nllb-200-distilled-600M").
    method:
        One of SUPPORTED_METHODS. "auto" uses the vocabulary-density-based
        predictive framework (see plan.md Phase 4) to recommend a method
        without training; falls back to "morfessor-vc" if the predictor is
        unavailable or underconfident.
    """

    def __init__(self, lang: str, model: str, method: str = "auto") -> None:
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Unknown method {method!r}; expected one of {SUPPORTED_METHODS}")
        self.lang = lang
        self.model = model
        self.method = method
        self._resolved_method = None

    def resolve_method(self) -> str:
        """Return the concrete method to use, resolving "auto" via vocabulary density."""
        if self.method != "auto":
            return self.method
        raise NotImplementedError(
            "Auto-mode recommendation depends on the Phase 4 predictive framework "
            "(vocab density -> recommended method), not yet trained/validated. "
            "Pass an explicit method for now."
        )

    def tokenize(self, corpus):
        """Tokenize `corpus` (iterable of strings) using the resolved method."""
        resolved = self.resolve_method()
        raise NotImplementedError(f"Method {resolved!r} not yet implemented.")
