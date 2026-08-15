"""Sentence-structure and burstiness features (Feature Groups 1 and 5).

Rationale: this is the brief's "its sentence rhythms are more even" claim,
made measurable. Human writers vary sentence length substantially and
unevenly — a long winding sentence followed by a short one. Decoding from a
language model produces a flatter rhythm.

Three distinct views of that rhythm, because they disagree in useful ways:

* `coefficient_of_variation` — spread relative to the mean. Scale-free, so a
  writer of long sentences and a writer of short ones are comparable.
* `burstiness` — `(σ − μ) / (σ + μ)`, the standard burstiness coefficient,
  bounded in [−1, 1]. Negative means more regular than a Poisson process.
* `mean_successive_difference` — average absolute gap between *consecutive*
  sentence lengths. This is the one that catches alternation: a document of
  lengths [10, 30, 10, 30] and one of [10, 10, 30, 30] have identical variance
  and identical burstiness, but completely different rhythm, and only this
  feature can tell them apart.

Document-scoped statistics are also reused to express each sentence as a
z-score against its own document, which is what makes a sentence's length
meaningful ("unusually regular *for this writer*") rather than absolute.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Sequence

from app.nlp.features.base import FeatureDict, prefix_features, safe_divide

__all__ = [
    "DocumentStatistics",
    "compute_document_statistics",
    "extract_structure_features",
    "extract_contextual_structure_features",
    "STRUCTURE_FEATURE_NAMES",
    "CONTEXTUAL_STRUCTURE_FEATURE_NAMES",
    "DOCUMENT_STRUCTURE_FEATURE_NAMES",
]

# Variance-based rhythm features need several sentences before they describe a
# rhythm rather than an accident. Below this, they are reported as undefined.
_MINIMUM_SENTENCES_FOR_RHYTHM = 4


@dataclass(frozen=True)
class DocumentStatistics:
    """Document-level length statistics, reused for z-scoring sentences."""

    sentence_count: int
    mean_length: float | None
    standard_deviation: float | None


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _standard_deviation(values: Sequence[float]) -> float | None:
    """Population standard deviation; None when undefined."""
    if len(values) < 2:
        return None
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def compute_document_statistics(token_counts: Sequence[int]) -> DocumentStatistics:
    """Summarize sentence lengths for one document."""
    lengths = [float(count) for count in token_counts]
    return DocumentStatistics(
        sentence_count=len(lengths),
        mean_length=_mean(lengths),
        standard_deviation=_standard_deviation(lengths),
    )


def extract_structure_features(token_count: int, character_count: int) -> FeatureDict:
    """Absolute size features for one sentence."""
    features: FeatureDict = {
        "token_count": float(token_count),
        "character_count": float(character_count),
        "mean_character_per_token": safe_divide(character_count, token_count),
    }
    return prefix_features(features, "struct")


def extract_contextual_structure_features(
    token_count: int, statistics: DocumentStatistics
) -> FeatureDict:
    """This sentence's length relative to the document it appears in."""
    if statistics.mean_length is None:
        return prefix_features(
            {"length_z_score": None, "length_ratio": None}, "structctx"
        )

    # A zero standard deviation means every sentence is the same length. The
    # z-score is undefined, but that uniformity is itself a strong signal, so
    # it is captured by the document-level features rather than faked here.
    z_score = (
        None
        if not statistics.standard_deviation
        else (token_count - statistics.mean_length) / statistics.standard_deviation
    )

    features: FeatureDict = {
        "length_z_score": z_score,
        "length_ratio": safe_divide(token_count, statistics.mean_length),
    }
    return prefix_features(features, "structctx")


def extract_document_rhythm_features(token_counts: Sequence[int]) -> FeatureDict:
    """Rhythm of sentence lengths across a document or window."""
    lengths = [float(count) for count in token_counts]

    if len(lengths) < _MINIMUM_SENTENCES_FOR_RHYTHM:
        return prefix_features(
            {
                "mean_length": _mean(lengths),
                "length_std": None,
                "coefficient_of_variation": None,
                "burstiness": None,
                "mean_successive_difference": None,
                "normalized_successive_difference": None,
            },
            "rhythm",
        )

    mean_length = _mean(lengths)
    standard_deviation = _standard_deviation(lengths)

    features: FeatureDict = {
        "mean_length": mean_length,
        "length_std": standard_deviation,
        "coefficient_of_variation": safe_divide(standard_deviation, mean_length),
    }

    # Burstiness coefficient (Goh & Barabasi). Bounded in [-1, 1]; more
    # negative means more regular than random.
    if standard_deviation is not None and mean_length is not None:
        denominator = standard_deviation + mean_length
        features["burstiness"] = safe_divide(
            standard_deviation - mean_length, denominator
        )
    else:
        features["burstiness"] = None

    successive_differences = [
        abs(second - first) for first, second in zip(lengths, lengths[1:])
    ]
    mean_successive = _mean(successive_differences)
    features["mean_successive_difference"] = mean_successive
    # Normalized so it is comparable between writers of long and short
    # sentences, in the same spirit as the coefficient of variation.
    features["normalized_successive_difference"] = (
        safe_divide(mean_successive, mean_length) if mean_successive is not None else None
    )

    return prefix_features(features, "rhythm")


STRUCTURE_FEATURE_NAMES: list[str] = [
    "struct_token_count",
    "struct_character_count",
    "struct_mean_character_per_token",
]

CONTEXTUAL_STRUCTURE_FEATURE_NAMES: list[str] = [
    "structctx_length_z_score",
    "structctx_length_ratio",
]

DOCUMENT_STRUCTURE_FEATURE_NAMES: list[str] = [
    "rhythm_mean_length",
    "rhythm_length_std",
    "rhythm_coefficient_of_variation",
    "rhythm_burstiness",
    "rhythm_mean_successive_difference",
    "rhythm_normalized_successive_difference",
]
