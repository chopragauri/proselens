"""The analysis service: essay text in, structured assessment out.

This is where the pipeline described in `docs/architecture.md` is actually
executed — preprocessing, segmentation, feature extraction, sentence scoring,
passage aggregation, document assessment, evidence generation.

Loading
-------
Models are loaded once, at import of the singleton, never per request. The
spaCy pipeline alone takes ~0.3 s to construct and the reference language model
is 12 MB of JSON; paying that per request would dominate latency and multiply
memory under concurrency.

Document score
--------------
The document risk is the trained document model's output, not an average of
sentence scores. It consumes document-scoped style features together with
summary statistics of the sentence-score *distribution* — see
`app.ml.detector` for why the shape of that distribution matters more than its
mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.ml.detector import DetectorArtifacts, confidence_from_evidence
from app.ml.evaluation import build_document_frame
from app.nlp.features.aggregate import (
    PASSAGE_WINDOW_SIZE,
    extract_features,
)
from app.nlp.features.predictability import NgramLanguageModel
from app.nlp.pipeline import SPACY_MODEL_NAME, get_nlp
from app.nlp.preprocessing import normalize
from app.nlp.segmentation import segment
from app.schemas.analysis import (
    Assessment,
    AnalyzeResponse,
    DocumentSummary,
    PassageResult,
    RiskBand,
    SentenceResult,
    SignalEvidence,
)
from app.services.evidence import (
    ReferenceStatistics,
    build_explanation,
    summarize_signals,
)

__all__ = ["AnalysisService", "get_analysis_service"]

# Risk-band cut points on the 0-100 scale. Chosen to match the operating
# threshold (50) used throughout evaluation, with a medium band around it
# acknowledging that scores near the boundary are genuinely uncertain.
LOW_RISK_CEILING = 35
HIGH_RISK_FLOOR = 65

# Confidence bands for display.
LOW_CONFIDENCE_CEILING = 0.34
HIGH_CONFIDENCE_FLOOR = 0.67

# Length gates, set from measurement rather than intuition. See
# `scripts/measure_length_sensitivity.py` and data/models/baseline/
# length_sensitivity.json for the numbers behind them.
#
# Truncating held-out test essays and rescoring shows the failure is entirely
# one-sided. Recall stays at or above 0.95 at every length, so short machine
# text is still caught; what collapses is the false-positive rate on human
# writing:
#
#     sentences:   3      5      8     12     16     25
#     FPR:      0.695  0.475  0.220  0.075  0.025  0.000
#
# Below five sentences the detector flags roughly half of all human writing,
# which is worse than useless — it is actively misleading. No verdict is
# offered there at all.
MINIMUM_SCORABLE_SENTENCES = 5

# Below twelve sentences the false-positive rate is still above 20%, so a
# "likely machine" verdict is withheld and the assessment is capped at mixed
# signals. The score and the highlighting are still shown, because the user
# may well want to see them; what is withheld is the accusation.
RELIABLE_SENTENCE_COUNT = 12


def risk_band_for(score: int | None) -> RiskBand:
    if score is None:
        return RiskBand.NOT_SCORED
    if score <= LOW_RISK_CEILING:
        return RiskBand.LOW
    if score >= HIGH_RISK_FLOOR:
        return RiskBand.HIGH
    return RiskBand.MEDIUM


def confidence_band_for(confidence: float) -> RiskBand:
    if confidence <= LOW_CONFIDENCE_CEILING:
        return RiskBand.LOW
    if confidence >= HIGH_CONFIDENCE_FLOOR:
        return RiskBand.HIGH
    return RiskBand.MEDIUM


def _to_score(probability: float) -> int:
    return int(round(probability * 100))


@dataclass
class AnalysisService:
    """Holds the loaded models and performs analysis."""

    artifacts: DetectorArtifacts
    language_model: NgramLanguageModel
    reference: ReferenceStatistics
    model_version: str

    def analyze(self, text: str) -> AnalyzeResponse:
        """Analyze one essay."""
        normalized = normalize(text)
        document = segment(normalized, nlp=get_nlp())
        features = extract_features(
            document, language_model=self.language_model, nlp=get_nlp()
        )

        sentences = document.sentences
        if not sentences:
            return self._empty_response(normalized, document.paragraphs)

        sentence_frame = pd.DataFrame(
            [
                {
                    "essay_id": "request",
                    "is_scorable": features_row.sentence.is_scorable,
                    **features_row.values,
                }
                for features_row in features.sentence_features
            ]
        )

        scorable_mask = sentence_frame["is_scorable"].to_numpy(dtype=bool)
        sentence_scores = np.full(len(sentence_frame), np.nan)
        contributions: list[dict[str, float]] = [{} for _ in range(len(sentence_frame))]

        if scorable_mask.any():
            scorable_frame = sentence_frame[scorable_mask]
            scores = self.artifacts.score_sentences(scorable_frame)
            sentence_scores[scorable_mask] = scores
            scorable_contributions = self.artifacts.sentence_contributions(
                scorable_frame
            )
            for position, index in enumerate(np.flatnonzero(scorable_mask)):
                contributions[index] = scorable_contributions[position]

        document_score, document_contributions = self._score_document(
            features, sentence_frame, sentence_scores
        )

        scorable_scores = sentence_scores[scorable_mask]
        confidence = confidence_from_evidence(
            scorable_scores, int(scorable_mask.sum()), len(sentences)
        )

        sentence_results = self._build_sentence_results(
            features, sentence_scores, contributions
        )
        passages = self._build_passages(features, sentence_scores)

        document_signals = summarize_signals(
            document_contributions,
            {key: value for key, value in features.document_values.items()},
            self.reference,
        )

        scorable_count = int(scorable_mask.sum())
        assessment = self._assess(document_score, scorable_count)
        summary_text = self._build_summary_text(
            assessment, document_signals, document_score, scorable_count
        )

        return AnalyzeResponse(
            risk_score=_to_score(document_score),
            confidence=round(confidence, 4),
            confidence_band=confidence_band_for(confidence),
            assessment=assessment,
            summary_text=summary_text,
            signals=[_to_signal_schema(signal) for signal in document_signals],
            sentences=sentence_results,
            passages=passages,
            summary=self._build_summary(
                normalized, document, features, sentence_scores, scorable_mask
            ),
            model_version=self.model_version,
        )

    def _build_summary_text(
        self,
        assessment: Assessment,
        signals: list,
        probability: float,
        scorable_count: int,
    ) -> str:
        """Prose summary, with an explicit caveat when the text is short."""
        if assessment is Assessment.INSUFFICIENT_TEXT:
            return (
                f"This text is too short to assess. Only {scorable_count} "
                "sentences carried enough content to measure, and below "
                f"{MINIMUM_SCORABLE_SENTENCES} the detector flags roughly half "
                "of all human writing. No verdict is offered."
            )

        explanation = build_explanation(signals, probability)

        if scorable_count < RELIABLE_SENTENCE_COUNT and probability >= 0.5:
            return (
                f"{explanation} This assessment is capped at mixed signals "
                f"because only {scorable_count} sentences were scorable. On "
                f"text this short the detector's false-positive rate on human "
                "writing exceeds 20%, so a stronger conclusion is not "
                "supportable."
            )

        return explanation

    def _score_document(
        self,
        features,
        sentence_frame: pd.DataFrame,
        sentence_scores: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        """Run the document model and decompose its output."""
        document_row = pd.DataFrame([{"essay_id": "request", **features.document_values}])
        scored_sentences = sentence_frame.assign(
            essay_id="request", _score=np.nan_to_num(sentence_scores)
        )
        frame = build_document_frame(
            document_row, scored_sentences, np.nan_to_num(sentence_scores)
        )

        probability = float(self.artifacts.score_document(frame)[0])

        matrix = self.artifacts.document_builder.transform(frame)
        coefficients = self.artifacts.document_model.coef_[0]
        names = self.artifacts.document_builder.column_names
        decomposition = {
            name: float(value * coefficient)
            for name, value, coefficient in zip(names, matrix[0], coefficients)
        }
        return probability, decomposition

    def _build_sentence_results(
        self,
        features,
        sentence_scores: np.ndarray,
        contributions: list[dict[str, float]],
    ) -> list[SentenceResult]:
        results: list[SentenceResult] = []
        for position, sentence_features in enumerate(features.sentence_features):
            sentence = sentence_features.sentence
            raw_score = sentence_scores[position]
            scored = not np.isnan(raw_score)

            signals = (
                summarize_signals(
                    contributions[position], sentence_features.values, self.reference
                )
                if scored
                else []
            )
            score = _to_score(float(raw_score)) if scored else None

            results.append(
                SentenceResult(
                    index=sentence.index,
                    text=sentence.text,
                    start=sentence.start,
                    end=sentence.end,
                    paragraph_index=sentence.paragraph_index,
                    token_count=sentence.token_count,
                    is_scorable=sentence.is_scorable,
                    risk_score=score,
                    risk_band=risk_band_for(score),
                    signals=[_to_signal_schema(signal) for signal in signals],
                    explanation=(
                        build_explanation(signals, float(raw_score)) if scored else None
                    ),
                )
            )
        return results

    def _build_passages(
        self, features, sentence_scores: np.ndarray
    ) -> list[PassageResult]:
        """Sliding windows, scored by the mean of their scorable sentences."""
        sentence_features = features.sentence_features
        if len(sentence_features) < PASSAGE_WINDOW_SIZE:
            return []

        passages: list[PassageResult] = []
        for start in range(len(sentence_features) - PASSAGE_WINDOW_SIZE + 1):
            window = sentence_features[start : start + PASSAGE_WINDOW_SIZE]
            window_scores = sentence_scores[start : start + PASSAGE_WINDOW_SIZE]
            defined = window_scores[~np.isnan(window_scores)]
            if defined.size == 0:
                continue

            score = _to_score(float(defined.mean()))
            passages.append(
                PassageResult(
                    start_sentence_index=window[0].sentence.index,
                    end_sentence_index=window[-1].sentence.index,
                    start=window[0].sentence.start,
                    end=window[-1].sentence.end,
                    risk_score=score,
                    risk_band=risk_band_for(score),
                )
            )
        return passages

    def _build_summary(
        self,
        normalized: str,
        document,
        features,
        sentence_scores: np.ndarray,
        scorable_mask: np.ndarray,
    ) -> DocumentSummary:
        scored = sentence_scores[~np.isnan(sentence_scores)]
        bands = [risk_band_for(_to_score(float(score))) for score in scored]

        return DocumentSummary(
            sentence_count=len(document.sentences),
            scorable_sentence_count=int(scorable_mask.sum()),
            paragraph_count=len(document.paragraphs),
            word_count=len(normalized.split()),
            character_count=len(normalized),
            high_risk_sentences=bands.count(RiskBand.HIGH),
            medium_risk_sentences=bands.count(RiskBand.MEDIUM),
            low_risk_sentences=bands.count(RiskBand.LOW),
            unscored_sentences=int((~scorable_mask).sum()),
            mean_sentence_risk=float(scored.mean()) if scored.size else None,
            lexical_diversity=features.document_values.get("doc_mtld"),
            sentence_length_variation=features.document_values.get(
                "rhythm_coefficient_of_variation"
            ),
            mean_predictability=features.document_values.get(
                "docmean_pred_cross_entropy"
            ),
        )

    def _assess(self, probability: float, scorable_count: int) -> Assessment:
        if scorable_count < MINIMUM_SCORABLE_SENTENCES:
            return Assessment.INSUFFICIENT_TEXT

        score = _to_score(probability)
        if score <= LOW_RISK_CEILING:
            return Assessment.LIKELY_HUMAN

        # On short text the model's high scores are unreliable in one specific
        # direction: it over-flags humans while still catching machines. So a
        # low score stays trustworthy and is reported as-is, but a high one is
        # downgraded rather than presented as a finding.
        if scorable_count < RELIABLE_SENTENCE_COUNT:
            return Assessment.MIXED_SIGNALS

        if score >= HIGH_RISK_FLOOR:
            return Assessment.LIKELY_MACHINE
        return Assessment.MIXED_SIGNALS

    def _empty_response(self, normalized: str, paragraphs) -> AnalyzeResponse:
        return AnalyzeResponse(
            risk_score=0,
            confidence=0.0,
            confidence_band=RiskBand.LOW,
            assessment=Assessment.INSUFFICIENT_TEXT,
            summary_text="No sentences could be identified in the submitted text.",
            signals=[],
            sentences=[],
            passages=[],
            summary=DocumentSummary(
                sentence_count=0,
                scorable_sentence_count=0,
                paragraph_count=len(paragraphs),
                word_count=len(normalized.split()),
                character_count=len(normalized),
                high_risk_sentences=0,
                medium_risk_sentences=0,
                low_risk_sentences=0,
                unscored_sentences=0,
                mean_sentence_risk=None,
            ),
            model_version=self.model_version,
        )


def _to_signal_schema(signal) -> SignalEvidence:
    return SignalEvidence(
        family=signal.family,
        direction=signal.direction,
        contribution=round(signal.contribution, 4),
        descriptor=signal.descriptor,
        measured_feature=signal.strongest_feature,
        measured_value=(
            round(signal.strongest_value, 6)
            if signal.strongest_value is not None
            else None
        ),
        z_score=(
            round(signal.strongest_z_score, 3)
            if signal.strongest_z_score is not None
            else None
        ),
    )


@lru_cache(maxsize=1)
def get_analysis_service() -> AnalysisService:
    """Load models once per process and return the shared service."""
    settings = get_settings()
    model_directory = Path(settings.model_directory)

    artifacts = DetectorArtifacts.load(model_directory)
    language_model = NgramLanguageModel.load(Path(settings.language_model_path))
    reference = ReferenceStatistics.load(model_directory / "reference_statistics.json")

    # Warm the spaCy pipeline so the first request does not pay for it.
    get_nlp()

    return AnalysisService(
        artifacts=artifacts,
        language_model=language_model,
        reference=reference,
        model_version=str(artifacts.metadata.get("version", "baseline-1")),
    )


def spacy_model_name() -> str:
    return SPACY_MODEL_NAME
