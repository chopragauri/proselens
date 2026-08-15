"""Repetition features (Feature Group 3).

Rationale: the brief's observation that machine text "repeats a narrower set of
constructions than people do" is measurable at two levels, and both are
implemented here:

* *Within* a sentence — immediate n-gram and word repetition.
* *Across* a document — how much this sentence's vocabulary, n-grams and
  syntactic skeleton echo the rest of the essay.

The cross-sentence measures are `FeatureScope.CONTEXTUAL`: they describe a
sentence relative to its document, and are undefined for a sentence analysed
alone. That is the point. A sentence that would look unremarkable in isolation
can be obviously formulaic once you see it is the fourth in the essay built on
the same template.

Structural repetition is measured over part-of-speech sequences rather than
words, so it catches "Throughout my academic journey, I have always..." and
"Across my extracurricular pursuits, I have consistently..." as the same
construction, which word-level matching would miss entirely.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from spacy.tokens import Doc, Span

from app.nlp.features.base import (
    FeatureDict,
    prefix_features,
    safe_divide,
    word_forms,
)

__all__ = [
    "extract_repetition_features",
    "extract_contextual_repetition_features",
    "REPETITION_FEATURE_NAMES",
    "CONTEXTUAL_REPETITION_FEATURE_NAMES",
]

# Length of the part-of-speech prefix used as a sentence's "opening template".
# Four tags is enough to distinguish "ADP DET ADJ NOUN" (a prepositional
# scene-setter) from "PRON VERB DET NOUN" (a plain declarative), while staying
# short enough that many sentences legitimately share one.
_OPENING_TEMPLATE_LENGTH = 4


def _ngrams(items: Sequence[str], size: int) -> list[tuple[str, ...]]:
    if len(items) < size:
        return []
    return [tuple(items[index : index + size]) for index in range(len(items) - size + 1)]


def _repeated_ngram_ratio(items: Sequence[str], size: int) -> float | None:
    """Share of n-grams that occur more than once within `items`."""
    ngrams = _ngrams(items, size)
    if not ngrams:
        return None
    counts = Counter(ngrams)
    repeated = sum(count for count in counts.values() if count > 1)
    return safe_divide(repeated, len(ngrams))


def extract_repetition_features(sentence: Doc | Span) -> FeatureDict:
    """Repetition measured within a single sentence."""
    words = word_forms(sentence)

    features: FeatureDict = {
        "repeated_bigram_ratio": _repeated_ngram_ratio(words, 2),
        "repeated_trigram_ratio": _repeated_ngram_ratio(words, 3),
    }

    # Immediate adjacent duplication ("very very", "the the") is a distinct
    # phenomenon from distant repetition and is more often a human typo or a
    # degenerate-decoding artefact.
    adjacent_duplicates = sum(
        1 for first, second in zip(words, words[1:]) if first == second
    )
    features["adjacent_duplicate_ratio"] = safe_divide(adjacent_duplicates, len(words))

    if words:
        most_common_count = Counter(words).most_common(1)[0][1]
        features["max_word_frequency"] = safe_divide(most_common_count, len(words))
    else:
        features["max_word_frequency"] = None

    return prefix_features(features, "rep")


def _opening_template(sentence: Doc | Span) -> tuple[str, ...]:
    """Part-of-speech skeleton of a sentence's opening."""
    tags = [
        token.pos_
        for token in sentence
        if not token.is_space and not token.is_punct
    ]
    return tuple(tags[:_OPENING_TEMPLATE_LENGTH])


def extract_contextual_repetition_features(
    sentence: Doc | Span, all_sentences: Sequence[Doc | Span]
) -> FeatureDict:
    """How strongly this sentence echoes the rest of its document.

    Args:
        sentence: The sentence being scored.
        all_sentences: Every parsed sentence in the document, including
            `sentence` itself. Self-matches are excluded internally.
    """
    if len(all_sentences) < 2:
        # With nothing to compare against, these features are undefined rather
        # than zero: a one-sentence essay has not demonstrated low repetition.
        return prefix_features(
            {
                "vocabulary_echo": None,
                "bigram_echo": None,
                "template_echo": None,
            },
            "repctx",
        )

    own_words = set(word_forms(sentence))
    own_bigrams = set(_ngrams(word_forms(sentence), 2))
    own_template = _opening_template(sentence)

    other_words: set[str] = set()
    other_bigrams: set[tuple[str, ...]] = set()
    matching_templates = 0
    other_count = 0

    for candidate in all_sentences:
        if candidate is sentence:
            continue
        other_count += 1
        candidate_words = word_forms(candidate)
        other_words.update(candidate_words)
        other_bigrams.update(_ngrams(candidate_words, 2))
        if own_template and _opening_template(candidate) == own_template:
            matching_templates += 1

    features: FeatureDict = {
        # Share of this sentence's vocabulary that also appears elsewhere.
        "vocabulary_echo": safe_divide(
            len(own_words & other_words), len(own_words)
        ),
        # Share of its bigrams reused elsewhere: a stronger signal than single
        # words, which repeat for ordinary reasons of topic.
        "bigram_echo": safe_divide(
            len(own_bigrams & other_bigrams), len(own_bigrams)
        ),
        # Share of other sentences that open with the same POS skeleton.
        "template_echo": safe_divide(matching_templates, other_count),
    }

    return prefix_features(features, "repctx")


REPETITION_FEATURE_NAMES: list[str] = [
    "rep_repeated_bigram_ratio",
    "rep_repeated_trigram_ratio",
    "rep_adjacent_duplicate_ratio",
    "rep_max_word_frequency",
]

CONTEXTUAL_REPETITION_FEATURE_NAMES: list[str] = [
    "repctx_vocabulary_echo",
    "repctx_bigram_echo",
    "repctx_template_echo",
]
