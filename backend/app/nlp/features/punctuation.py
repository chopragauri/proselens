"""Punctuation features (Feature Group 4).

Rationale: punctuation choice is close to a stylistic fingerprint and is
largely untouched by topic. Machine prose tends toward a narrow, regular
repertoire — commas and periods — while human writers reach more often for
dashes, parentheses, ellipses, exclamation marks and semicolons, and do so
unevenly across a document.

All densities are per-token rather than per-character, so a long sentence and
a short one are comparable.
"""

from __future__ import annotations

from spacy.tokens import Doc, Span

from app.nlp.features.base import FeatureDict, prefix_features, safe_divide

__all__ = ["extract_punctuation_features", "PUNCTUATION_FEATURE_NAMES"]

# Each entry maps a feature name to the literal characters that count toward
# it. Kept explicit rather than using unicode categories so that the evidence
# panel can name precisely what was counted.
_PUNCTUATION_CLASSES: dict[str, str] = {
    "comma": ",",
    "semicolon": ";",
    "colon": ":",
    "period": ".",
    "question": "?",
    "exclamation": "!",
    "dash": "-",
    "quote": "\"'",
    "parenthesis": "()",
}


def extract_punctuation_features(sentence: Doc | Span) -> FeatureDict:
    """Punctuation densities for one sentence, normalized by token count."""
    token_count = sum(1 for token in sentence if not token.is_space)
    text = sentence.text

    features: FeatureDict = {}
    for class_name, characters in _PUNCTUATION_CLASSES.items():
        occurrences = sum(text.count(character) for character in characters)
        features[f"{class_name}_density"] = safe_divide(occurrences, token_count)

    total_punctuation = sum(1 for token in sentence if token.is_punct)
    features["total_density"] = safe_divide(total_punctuation, token_count)

    # An ellipsis is a distinct rhetorical move, not three periods. Counted
    # separately because the period density above would otherwise triple-count
    # it and read as unusually heavy sentence-final punctuation.
    features["ellipsis_count"] = float(text.count("...") + text.count("…"))

    return prefix_features(features, "punct")


PUNCTUATION_FEATURE_NAMES: list[str] = [
    f"punct_{name}_density" for name in _PUNCTUATION_CLASSES
] + ["punct_total_density", "punct_ellipsis_count"]
