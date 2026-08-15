"""Predictability features (Feature Group 7).

The usual approach is GPT-2 perplexity. This module instead fits an
interpolated trigram language model on the *human training essays only* and
measures each sentence's cross-entropy under it.

Decision
    A back-off n-gram language model, trained in-repo, as the primary
    predictability signal.

Why
    Three reasons, in order of importance. (1) It is auditable: when the
    evidence panel says a sentence is highly predictable, that claim traces to
    counts over a corpus we built and can inspect, not to a 500 MB binary whose
    training data is unknown. (2) It runs in milliseconds on CPU, so analysis
    stays responsive without a GPU. (3) It fits the 1.4 GB of disk this machine
    has left, where GPT-2 does not.

Alternative
    GPT-2 or TinyLlama negative log-likelihood. Almost certainly a stronger raw
    signal, since a neural model captures long-range structure a trigram cannot.

Trade-off
    We accept weaker separation for transparency and speed. There is also a
    subtler cost, stated plainly because it constrains interpretation: fit on
    our own corpus, this model partly measures *domain fit* rather than pure
    fluency. A sentence about an unusual topic scores as "unpredictable" even
    if a machine wrote it. That is one reason predictability is only one input
    among many, and never the verdict.

Verification
    `scripts/evaluate.py` reports model performance with and without this
    feature group. If it does not improve validation AUC over the other groups,
    it should be removed rather than kept for appearances.

Leakage
    The model is fit strictly on the training split. Fitting on all human text
    would let test-set sentences inform their own predictability score, which
    would inflate every number downstream.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = ["NgramLanguageModel", "SENTENCE_BOUNDARY", "UNKNOWN_TOKEN"]

SENTENCE_BOUNDARY = "<s>"
UNKNOWN_TOKEN = "<unk>"

# Interpolation weights for trigram, bigram and unigram estimates. Fixed in
# advance rather than tuned, so they cannot quietly overfit the validation set;
# the standard heuristic weighting is adequate for a feature that feeds a
# downstream classifier.
_TRIGRAM_WEIGHT = 0.5
_BIGRAM_WEIGHT = 0.3
_UNIGRAM_WEIGHT = 0.2

# Add-k smoothing constant for the unigram floor, which guarantees every token
# receives non-zero probability and therefore finite cross-entropy.
_ADDITIVE_SMOOTHING = 0.4


@dataclass
class _ModelCounts:
    unigrams: Counter[str]
    bigrams: Counter[tuple[str, str]]
    trigrams: Counter[tuple[str, str, str]]
    bigram_contexts: Counter[tuple[str, str]]
    unigram_contexts: Counter[str]


class NgramLanguageModel:
    """Interpolated trigram model over lowercased word forms.

    Memory is controlled by two prunings, both necessary at corpus scale: the
    vocabulary is capped, and singleton trigrams are discarded. Singletons are
    the bulk of the table and contribute almost nothing after interpolation.
    """

    def __init__(
        self,
        maximum_vocabulary: int = 50_000,
        minimum_trigram_count: int = 2,
    ) -> None:
        self.maximum_vocabulary = maximum_vocabulary
        self.minimum_trigram_count = minimum_trigram_count
        self.vocabulary: set[str] = set()
        self._counts: _ModelCounts | None = None
        self._total_tokens: int = 0

    @property
    def is_fitted(self) -> bool:
        return self._counts is not None

    def _map_token(self, token: str) -> str:
        return token if token in self.vocabulary else UNKNOWN_TOKEN

    def _padded(self, tokens: Sequence[str]) -> list[str]:
        """Prepend two boundary symbols so the first real token has a context."""
        return [SENTENCE_BOUNDARY, SENTENCE_BOUNDARY, *tokens]

    def fit(self, tokenized_sentences: Iterable[Sequence[str]]) -> NgramLanguageModel:
        """Fit on tokenized sentences from the training split only."""
        sentences = [list(sentence) for sentence in tokenized_sentences]

        raw_frequencies: Counter[str] = Counter()
        for sentence in sentences:
            raw_frequencies.update(sentence)

        self.vocabulary = {
            token
            for token, _ in raw_frequencies.most_common(self.maximum_vocabulary)
        }

        unigrams: Counter[str] = Counter()
        bigrams: Counter[tuple[str, str]] = Counter()
        trigrams: Counter[tuple[str, str, str]] = Counter()
        bigram_contexts: Counter[tuple[str, str]] = Counter()
        unigram_contexts: Counter[str] = Counter()

        for sentence in sentences:
            mapped = self._padded([self._map_token(token) for token in sentence])
            for position in range(2, len(mapped)):
                first, second, third = mapped[position - 2 : position + 1]
                unigrams[third] += 1
                bigrams[(second, third)] += 1
                trigrams[(first, second, third)] += 1
                bigram_contexts[(first, second)] += 1
                unigram_contexts[second] += 1

        pruned_trigrams = Counter(
            {
                ngram: count
                for ngram, count in trigrams.items()
                if count >= self.minimum_trigram_count
            }
        )

        self._counts = _ModelCounts(
            unigrams=unigrams,
            bigrams=bigrams,
            trigrams=pruned_trigrams,
            bigram_contexts=bigram_contexts,
            unigram_contexts=unigram_contexts,
        )
        self._total_tokens = sum(unigrams.values())
        return self

    def _token_probability(self, first: str, second: str, third: str) -> float:
        """Interpolated P(third | first, second). Always strictly positive."""
        assert self._counts is not None
        counts = self._counts

        vocabulary_size = max(len(self.vocabulary) + 1, 1)  # +1 for <unk>
        unigram_probability = (counts.unigrams[third] + _ADDITIVE_SMOOTHING) / (
            self._total_tokens + _ADDITIVE_SMOOTHING * vocabulary_size
        )

        bigram_context_count = counts.unigram_contexts[second]
        bigram_probability = (
            counts.bigrams[(second, third)] / bigram_context_count
            if bigram_context_count > 0
            else 0.0
        )

        trigram_context_count = counts.bigram_contexts[(first, second)]
        trigram_probability = (
            counts.trigrams[(first, second, third)] / trigram_context_count
            if trigram_context_count > 0
            else 0.0
        )

        return (
            _TRIGRAM_WEIGHT * trigram_probability
            + _BIGRAM_WEIGHT * bigram_probability
            + _UNIGRAM_WEIGHT * unigram_probability
        )

    def cross_entropy(self, tokens: Sequence[str]) -> float | None:
        """Mean negative log2 probability per token. Lower means more predictable.

        Returns None for empty input or an unfitted model, rather than a
        sentinel number that would be mistaken for a measurement.
        """
        if not self.is_fitted or not tokens:
            return None

        mapped = self._padded([self._map_token(token) for token in tokens])
        total_log_probability = 0.0
        counted = 0

        for position in range(2, len(mapped)):
            first, second, third = mapped[position - 2 : position + 1]
            probability = self._token_probability(first, second, third)
            if probability <= 0.0:
                # Cannot occur given the smoothed unigram floor, but guarding
                # here means a future change to the weights degrades loudly in
                # tests rather than producing a silent -inf.
                return None
            total_log_probability += math.log2(probability)
            counted += 1

        if counted == 0:
            return None
        return -total_log_probability / counted

    def perplexity(self, tokens: Sequence[str]) -> float | None:
        """2 ** cross_entropy, in the conventional units."""
        entropy = self.cross_entropy(tokens)
        return None if entropy is None else 2.0**entropy

    def save(self, path: Path) -> None:
        """Serialize to JSON.

        JSON rather than pickle: model artifacts are loaded by the API at
        startup, and an unpickle is arbitrary code execution. A JSON file
        cannot execute anything, which matters more here than the file size.
        """
        if self._counts is None:
            raise ValueError("Cannot save an unfitted model.")

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "maximum_vocabulary": self.maximum_vocabulary,
            "minimum_trigram_count": self.minimum_trigram_count,
            "total_tokens": self._total_tokens,
            "vocabulary": sorted(self.vocabulary),
            "unigrams": dict(self._counts.unigrams),
            "bigrams": {
                "\t".join(key): value for key, value in self._counts.bigrams.items()
            },
            "trigrams": {
                "\t".join(key): value for key, value in self._counts.trigrams.items()
            },
            "bigram_contexts": {
                "\t".join(key): value
                for key, value in self._counts.bigram_contexts.items()
            },
            "unigram_contexts": dict(self._counts.unigram_contexts),
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> NgramLanguageModel:
        """Load a model previously written by `save`."""
        payload = json.loads(path.read_text(encoding="utf-8"))

        model = cls(
            maximum_vocabulary=payload["maximum_vocabulary"],
            minimum_trigram_count=payload["minimum_trigram_count"],
        )
        model.vocabulary = set(payload["vocabulary"])
        model._total_tokens = payload["total_tokens"]
        model._counts = _ModelCounts(
            unigrams=Counter(payload["unigrams"]),
            bigrams=Counter(
                {tuple(key.split("\t")): value for key, value in payload["bigrams"].items()}
            ),
            trigrams=Counter(
                {tuple(key.split("\t")): value for key, value in payload["trigrams"].items()}
            ),
            bigram_contexts=Counter(
                {
                    tuple(key.split("\t")): value
                    for key, value in payload["bigram_contexts"].items()
                }
            ),
            unigram_contexts=Counter(payload["unigram_contexts"]),
        )
        return model
