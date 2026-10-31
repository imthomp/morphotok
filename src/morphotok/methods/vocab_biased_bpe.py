"""Vocab-biased BPE.

Standard BPE training, modified so the merge selection score upweights any
merge whose resulting token is already present in the pretrained model's
vocabulary:

    score(merge) = freq(merge) * (1 + lambda * 1[result in pretrained_vocab])

lambda is a hyperparameter tuned per language (see notes/plan.md's
pre-registered Optuna search: log-uniform [0.1, 10.0], 20 trials/language).
No Morfessor dependency; purely BPE-based, so it retains BPE's universal
coverage guarantee (falls back to individual characters for anything never
merged).

No existing library exposes a bias term in merge selection, so this is a
from-scratch (word-frequency-weighted, character-pair) BPE trainer, not a
wrapper around `tokenizers`/sentencepiece. It follows the standard BPE
training loop (Sennrich et al., 2016a) with one change: the pair-selection
score at each step.

Training uses the standard incremental-update trick (track pair frequencies
and which words contain each pair; on each merge, only re-derive pair counts
for the words actually containing the merged pair) rather than rescanning
the whole corpus every merge -- an earlier full-rescan version was O(words *
num_merges) and became the bottleneck once num_merges reaches NLLB-scale
vocab sizes (tens of thousands), not just on small sanity-check runs.
"""

from __future__ import annotations

from collections import Counter, defaultdict

WORD_BOUNDARY = "▁"  # SentencePiece word-initial marker


def _word_to_symbols(word: str) -> tuple[str, ...]:
    if not word:
        return ()
    return (WORD_BOUNDARY + word[0],) + tuple(word[1:])


def _pairs(syms: list[str]):
    return zip(syms, syms[1:])


class VocabBiasedBPE:
    def __init__(self, model_vocab: set[str], lam: float = 1.0) -> None:
        self.model_vocab = model_vocab
        self.lam = lam
        self._merges: list[tuple[str, str]] = []
        self._merge_rank: dict[tuple[str, str], int] = {}

    def fit(self, monolingual_corpus, num_merges: int = 10000) -> None:
        word_freq = Counter(w for w in monolingual_corpus if w)
        # symbols[word] = current list of symbols for that word type, mutated in place as merges are learned
        symbols: dict[str, list[str]] = {w: list(_word_to_symbols(w)) for w in word_freq}

        pair_freq: Counter[tuple[str, str]] = Counter()
        pair_words: dict[tuple[str, str], set[str]] = defaultdict(set)
        for w, syms in symbols.items():
            freq = word_freq[w]
            for pair in _pairs(syms):
                pair_freq[pair] += freq
                pair_words[pair].add(w)

        def score(item: tuple[tuple[str, str], int]) -> float:
            pair, freq = item
            merged = pair[0] + pair[1]
            bias = 1.0 + self.lam if merged in self.model_vocab else 1.0
            return freq * bias

        merges: list[tuple[str, str]] = []
        for _ in range(num_merges):
            if not pair_freq:
                break
            best_pair = max(pair_freq.items(), key=score)[0]
            merged_symbol = best_pair[0] + best_pair[1]
            merges.append(best_pair)

            for w in list(pair_words.get(best_pair, ())):
                freq = word_freq[w]
                syms = symbols[w]

                for pair in _pairs(syms):
                    pair_freq[pair] -= freq
                    if pair_freq[pair] <= 0:
                        del pair_freq[pair]
                    pair_words[pair].discard(w)

                new_syms: list[str] = []
                i = 0
                while i < len(syms):
                    if i < len(syms) - 1 and syms[i] == best_pair[0] and syms[i + 1] == best_pair[1]:
                        new_syms.append(merged_symbol)
                        i += 2
                    else:
                        new_syms.append(syms[i])
                        i += 1
                symbols[w] = new_syms

                for pair in _pairs(new_syms):
                    pair_freq[pair] += freq
                    pair_words[pair].add(w)

            pair_freq.pop(best_pair, None)
            pair_words.pop(best_pair, None)

        self._merges = merges
        self._merge_rank = {pair: i for i, pair in enumerate(merges)}

    def segment(self, word: str) -> list[str]:
        if not self._merges:
            raise RuntimeError("Call fit() before segment().")
        syms = list(_word_to_symbols(word))
        while len(syms) > 1:
            ranked = [
                (self._merge_rank[(a, b)], i)
                for i, (a, b) in enumerate(zip(syms, syms[1:]))
                if (a, b) in self._merge_rank
            ]
            if not ranked:
                break
            _, merge_at = min(ranked)
            syms[merge_at : merge_at + 2] = [syms[merge_at] + syms[merge_at + 1]]
        return syms
