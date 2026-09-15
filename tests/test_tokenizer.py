import pytest

from morphotok import MorphoTokenizer


def test_rejects_unknown_method():
    with pytest.raises(ValueError):
        MorphoTokenizer(lang="tur", model="facebook/nllb-200-distilled-600M", method="not-a-method")


def test_auto_resolve_not_yet_implemented():
    tok = MorphoTokenizer(lang="tur", model="facebook/nllb-200-distilled-600M", method="auto")
    with pytest.raises(NotImplementedError):
        tok.resolve_method()
