"""Shared contract for feature extractors.

Every extractor is a pure function from a parsed sentence (or a list of them)
to a `FeatureDict`. Purity matters for two reasons: the same essay must always
produce the same evidence, and a feature that can be computed in isolation can
be tested in isolation.

Missing values
--------------
A feature returns `None` when it is *undefined* for the input, not when it is
zero. Type-token ratio on an empty sentence is undefined; comma density on a
comma-free sentence is genuinely 0.0. Collapsing those two cases would teach
the model that short sentences look comma-free, which is false.

`None` is later converted to the training-set median plus a companion
`<name>__missing` indicator, so the model can learn that missingness is itself
informative rather than silently imputing a lie.
"""

from __future__ import annotations

from enum import Enum

from spacy.tokens import Doc, Span

__all__ = [
    "FeatureDict",
    "FeatureScope",
    "safe_divide",
    "prefix_features",
    "word_forms",
    "MISSING_SUFFIX",
]

# Feature name -> value, where None means "undefined for this input".
FeatureDict = dict[str, float | None]

MISSING_SUFFIX = "__missing"


class FeatureScope(str, Enum):
    """What a feature is computed over.

    The distinction drives the two-model architecture: a sentence-level model
    may only consume SENTENCE and CONTEXTUAL features, because WINDOW and
    DOCUMENT features are undefined for a single sentence in isolation.
    Feeding a document-scoped variance feature to a sentence classifier would
    produce a number, but it would be a meaningless one, and the evidence panel
    would then be explaining noise.
    """

    SENTENCE = "sentence"
    """Computable from one sentence alone."""

    CONTEXTUAL = "contextual"
    """This sentence expressed relative to its document (e.g. a z-score)."""

    WINDOW = "window"
    """Computed over a sliding window of consecutive sentences."""

    DOCUMENT = "document"
    """Computed over the whole essay."""


def safe_divide(numerator: float, denominator: float) -> float | None:
    """Divide, returning None when the ratio is undefined.

    Used pervasively: nearly every density and ratio feature has a denominator
    that can legitimately be zero for short or unusual input.
    """
    if denominator == 0:
        return None
    return numerator / denominator


def word_forms(sentence: Doc | Span) -> list[str]:
    """Lowercased word forms, excluding punctuation and whitespace.

    The single definition of "what counts as a word" in this codebase. Lexical
    diversity, repetition and the language model must agree on it: if they
    disagreed, a token could be counted for one feature and not another, and
    the resulting feature values would not be mutually comparable.
    """
    return [
        token.text.lower()
        for token in sentence
        if not token.is_punct and not token.is_space and token.text.strip()
    ]


def prefix_features(features: FeatureDict, prefix: str) -> FeatureDict:
    """Namespace a feature group, e.g. 'ttr' -> 'lex_ttr'.

    Keeps names unique once groups are merged, and makes the origin of any
    feature obvious in the evidence panel and in model coefficient dumps.
    """
    return {f"{prefix}_{name}": value for name, value in features.items()}
