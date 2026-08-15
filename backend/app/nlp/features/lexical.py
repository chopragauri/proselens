"""Lexical diversity features (Feature Group 2).

Rationale: machine prose draws on a narrower vocabulary and reuses it more
evenly than human prose, but naive diversity measures cannot see this because
they are confounded by length.

The short-text instability problem
----------------------------------
Type-token ratio (unique words / total words) falls mechanically as text gets
longer: a 10-word sentence almost always scores near 1.0, a 40-word sentence
almost never does. A classifier fed raw TTR therefore learns *sentence length*
wearing a disguise, and the evidence panel would tell the user "low lexical
diversity" when it means "this sentence is long".

Two mitigations, both implemented here:

* `mtld` — Measure of Textual Lexical Diversity. Counts how many words it takes
  for a running TTR to fall through a threshold, then averages those run
  lengths. It is length-invariant by construction, which is exactly the
  property TTR lacks.
* `root_ttr` — Guiraud's index, `types / sqrt(tokens)`, a cheap partial
  correction retained for comparison.

Raw `ttr` is still emitted, because on short spans MTLD is itself unstable and
the model benefits from seeing both alongside the token count.
"""

from __future__ import annotations

from spacy.tokens import Doc, Span

from app.nlp.features.base import (
    FeatureDict,
    prefix_features,
    safe_divide,
    word_forms,
)

__all__ = ["extract_lexical_features", "mtld", "LEXICAL_FEATURE_NAMES"]

# The conventional MTLD threshold from McCarthy & Jarvis (2010). A factor ends
# when running TTR drops below this value.
_MTLD_TTR_THRESHOLD = 0.72

# Below this many tokens MTLD cannot complete even one factor, so it would
# report an artefact of the fallback rather than a measurement.
_MTLD_MINIMUM_TOKENS = 10


def _mtld_one_direction(words: list[str]) -> float | None:
    """Mean factor length scanning `words` once, left to right."""
    factor_count = 0.0
    types: set[str] = set()
    token_count = 0

    for word in words:
        types.add(word)
        token_count += 1
        running_ttr = len(types) / token_count
        if running_ttr <= _MTLD_TTR_THRESHOLD:
            factor_count += 1
            types = set()
            token_count = 0

    # The trailing incomplete factor is credited proportionally: a run that got
    # most of the way to the threshold is evidence of diversity and discarding
    # it would bias short texts.
    if token_count > 0:
        running_ttr = len(types) / token_count
        remaining = (1.0 - running_ttr) / (1.0 - _MTLD_TTR_THRESHOLD)
        factor_count += remaining

    # Text whose running TTR never falls through the threshold completes no
    # factor, leaving factor_count at zero. That is not an undefined
    # measurement — it is the maximum one: diversity never decayed across the
    # whole span. Flooring at one factor reports MTLD = token count, the
    # standard reading of "at least this diverse".
    #
    # Returning None here instead, as an earlier version did, was actively
    # harmful: every sentence with no repeated word scored as missing, and
    # sentences with no repeated word are disproportionately human. The
    # missingness indicator would then have carried label information that the
    # feature itself was supposed to carry.
    factor_count = max(factor_count, 1.0)
    return len(words) / factor_count


def mtld(words: list[str]) -> float | None:
    """Measure of Textual Lexical Diversity, averaged over both directions.

    Returns None when the text is too short for the measure to mean anything.
    """
    if len(words) < _MTLD_MINIMUM_TOKENS:
        return None

    forward = _mtld_one_direction(words)
    backward = _mtld_one_direction(list(reversed(words)))
    if forward is None or backward is None:
        return None
    return (forward + backward) / 2.0


def extract_lexical_features(sentence: Doc | Span) -> FeatureDict:
    """Lexical diversity and vocabulary features for one sentence."""
    words = word_forms(sentence)
    total = len(words)
    unique = len(set(words))

    features: FeatureDict = {
        "token_count": float(total),
        "unique_count": float(unique),
        "ttr": safe_divide(unique, total),
        "root_ttr": safe_divide(unique, total**0.5) if total > 0 else None,
        "mtld": mtld(words),
        # Words occurring exactly once. A high share indicates the writer is
        # not circling the same vocabulary.
        "hapax_ratio": safe_divide(
            sum(1 for word in set(words) if words.count(word) == 1), total
        ),
        "repeated_word_ratio": safe_divide(total - unique, total),
        "mean_word_length": safe_divide(sum(len(word) for word in words), total),
    }

    # Stopword share proxies how much of the sentence is function words. Machine
    # prose often has a flatter, more uniform function-word skeleton.
    stopword_count = sum(
        1 for token in sentence if token.is_stop and not token.is_punct
    )
    non_space_count = sum(1 for token in sentence if not token.is_space)
    features["stopword_ratio"] = safe_divide(stopword_count, non_space_count)

    content_count = sum(
        1
        for token in sentence
        if not token.is_stop and not token.is_punct and not token.is_space
    )
    features["content_word_ratio"] = safe_divide(content_count, non_space_count)

    # Long words are a rough register signal; formal machine prose leans on
    # them more consistently than human personal writing does.
    features["long_word_ratio"] = safe_divide(
        sum(1 for word in words if len(word) >= 7), total
    )

    return prefix_features(features, "lex")


LEXICAL_FEATURE_NAMES: list[str] = [
    "lex_token_count",
    "lex_unique_count",
    "lex_ttr",
    "lex_root_ttr",
    "lex_mtld",
    "lex_hapax_ratio",
    "lex_repeated_word_ratio",
    "lex_mean_word_length",
    "lex_stopword_ratio",
    "lex_content_word_ratio",
    "lex_long_word_ratio",
]
