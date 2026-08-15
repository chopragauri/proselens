"""Composition of every feature group into per-sentence and per-document vectors.

This module owns the scope discipline described in `base.FeatureScope`. It
produces two things from one essay:

* A feature vector per sentence, containing only SENTENCE and CONTEXTUAL
  features — everything a sentence-level classifier is allowed to see.
* A feature vector for the document, containing rhythm statistics and
  distributional summaries of the sentence features.

The separation is deliberate and is the core of the two-model architecture: a
document-scoped feature such as `rhythm_burstiness` cannot be computed from one
sentence, so letting a sentence classifier consume it would mean feeding it a
number that does not describe the thing being scored.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from spacy.language import Language
from spacy.tokens import Doc

from app.nlp.features.base import FeatureDict, word_forms
from app.nlp.features.formulaic import extract_formulaic_features
from app.nlp.features.lexical import extract_lexical_features, mtld
from app.nlp.features.predictability import NgramLanguageModel
from app.nlp.features.punctuation import extract_punctuation_features
from app.nlp.features.repetition import (
    extract_contextual_repetition_features,
    extract_repetition_features,
)
from app.nlp.features.structure import (
    DocumentStatistics,
    compute_document_statistics,
    extract_contextual_structure_features,
    extract_document_rhythm_features,
    extract_structure_features,
)
from app.nlp.features.syntax import extract_syntax_features
from app.nlp.pipeline import get_nlp
from app.nlp.segmentation import SegmentedDocument, Sentence

__all__ = [
    "SentenceFeatures",
    "DocumentFeatures",
    "extract_features",
    "extract_window_features",
    "PASSAGE_WINDOW_SIZE",
]

# Sliding-window size for passage-level scoring. Four sentences is inside the
# 3-5 range the brief specifies: long enough for variance features to mean
# something, short enough to localize a rewritten paragraph.
PASSAGE_WINDOW_SIZE = 4

# Sentence features summarized into document-level distribution statistics.
# Restricted to the features with a clear document-level reading, rather than
# summarizing all of them, which would triple the document feature count for
# little gain.
_SUMMARIZED_SENTENCE_FEATURES = (
    "lex_ttr",
    "lex_mtld",
    "lex_stopword_ratio",
    "syn_mean_dependency_depth",
    "syn_first_person_ratio",
    "syn_clauses_per_token",
    "punct_total_density",
    "punct_comma_density",
    "form_hedge_density",
    "form_nominalization_density",
    "form_specificity_density",
    "pred_cross_entropy",
)


@dataclass(frozen=True)
class SentenceFeatures:
    """One sentence and its feature vector."""

    sentence: Sentence
    values: FeatureDict


@dataclass(frozen=True)
class DocumentFeatures:
    """Feature vectors for an entire essay."""

    sentence_features: list[SentenceFeatures]
    document_values: FeatureDict
    statistics: DocumentStatistics


def _predictability_features(
    parsed: Doc, language_model: NgramLanguageModel | None
) -> FeatureDict:
    """Cross-entropy under the reference language model.

    Returns None when no model is supplied, so the feature pipeline stays
    usable before the model has been trained — which is what lets Phase 3 be
    tested before Phase 5 exists.
    """
    if language_model is None or not language_model.is_fitted:
        return {"pred_cross_entropy": None, "pred_perplexity": None}

    tokens = word_forms(parsed)
    return {
        "pred_cross_entropy": language_model.cross_entropy(tokens),
        "pred_perplexity": language_model.perplexity(tokens),
    }


def _summarize(values: Sequence[float]) -> tuple[float | None, float | None]:
    """Mean and standard deviation of the defined values in a series."""
    if not values:
        return None, None
    mean_value = statistics.fmean(values)
    deviation = statistics.pstdev(values) if len(values) > 1 else None
    return mean_value, deviation


def extract_features(
    document: SegmentedDocument,
    language_model: NgramLanguageModel | None = None,
    nlp: Language | None = None,
) -> DocumentFeatures:
    """Extract every feature group for one segmented essay.

    Args:
        document: Output of `segmentation.segment`.
        language_model: Optional fitted reference model for predictability.
        nlp: Optional pipeline override, for tests.
    """
    pipeline = nlp if nlp is not None else get_nlp()
    sentences = document.sentences

    if not sentences:
        return DocumentFeatures(
            sentence_features=[],
            document_values={},
            statistics=compute_document_statistics([]),
        )

    # One batched pass over sentence texts. Parsing each sentence separately
    # would multiply pipeline overhead by the sentence count.
    parsed_sentences = list(pipeline.pipe([sentence.text for sentence in sentences]))

    token_counts = [sentence.token_count for sentence in sentences]
    document_statistics = compute_document_statistics(token_counts)

    sentence_features: list[SentenceFeatures] = []
    for sentence, parsed in zip(sentences, parsed_sentences):
        values: FeatureDict = {}
        # SENTENCE scope: computable from this sentence alone.
        values.update(
            extract_structure_features(
                token_count=sentence.token_count,
                character_count=len(sentence.text),
            )
        )
        values.update(extract_lexical_features(parsed))
        values.update(extract_syntax_features(parsed))
        values.update(extract_punctuation_features(parsed))
        values.update(extract_repetition_features(parsed))
        values.update(extract_formulaic_features(parsed))
        values.update(_predictability_features(parsed, language_model))
        # CONTEXTUAL scope: this sentence relative to its document.
        values.update(
            extract_contextual_structure_features(
                sentence.token_count, document_statistics
            )
        )
        values.update(
            extract_contextual_repetition_features(parsed, parsed_sentences)
        )
        sentence_features.append(
            SentenceFeatures(sentence=sentence, values=values)
        )

    document_values = _build_document_features(
        parsed_sentences=parsed_sentences,
        sentence_features=sentence_features,
        token_counts=token_counts,
    )

    return DocumentFeatures(
        sentence_features=sentence_features,
        document_values=document_values,
        statistics=document_statistics,
    )


def _build_document_features(
    parsed_sentences: Sequence[Doc],
    sentence_features: Sequence[SentenceFeatures],
    token_counts: Sequence[int],
) -> FeatureDict:
    """Document-scoped features: rhythm, whole-document lexis, and summaries."""
    document_values: FeatureDict = {}
    document_values.update(extract_document_rhythm_features(token_counts))

    # Lexical diversity over the concatenated document, which is a different
    # quantity from the mean of per-sentence diversity: it can see a writer
    # reusing the same vocabulary across sentences.
    all_words = [word for parsed in parsed_sentences for word in word_forms(parsed)]
    unique_words = len(set(all_words))
    document_values["doc_token_count"] = float(len(all_words))
    document_values["doc_unique_count"] = float(unique_words)
    document_values["doc_ttr"] = (
        unique_words / len(all_words) if all_words else None
    )
    document_values["doc_mtld"] = mtld(all_words)
    document_values["doc_sentence_count"] = float(len(token_counts))

    # Distributional summaries of the per-sentence features. The standard
    # deviations matter as much as the means: uniformly average sentences are
    # the signature the brief describes as "smoother than it should be".
    for feature_name in _SUMMARIZED_SENTENCE_FEATURES:
        defined_values = [
            features.values[feature_name]
            for features in sentence_features
            if features.values.get(feature_name) is not None
        ]
        mean_value, deviation = _summarize(defined_values)  # type: ignore[arg-type]
        document_values[f"docmean_{feature_name}"] = mean_value
        document_values[f"docstd_{feature_name}"] = deviation

    return document_values


def extract_window_features(
    sentence_features: Sequence[SentenceFeatures],
    window_size: int = PASSAGE_WINDOW_SIZE,
) -> list[FeatureDict]:
    """Rhythm features for each sliding window of consecutive sentences.

    Returns one FeatureDict per window position. Windows are what passage-level
    scoring consumes: they smooth the noise in individual sentence scores while
    still localizing which part of the essay is anomalous.
    """
    if len(sentence_features) < window_size:
        return []

    windows: list[FeatureDict] = []
    for start in range(len(sentence_features) - window_size + 1):
        window = sentence_features[start : start + window_size]
        token_counts = [features.sentence.token_count for features in window]
        window_values = extract_document_rhythm_features(token_counts)
        window_values["window_start_index"] = float(window[0].sentence.index)
        window_values["window_end_index"] = float(window[-1].sentence.index)
        windows.append(window_values)
    return windows
