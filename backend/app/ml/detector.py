"""The fitted two-model detector, and how a document score is derived.

Architecture
------------
Two logistic regressions, not one:

* **Sentence model** — consumes only SENTENCE and CONTEXTUAL features, so every
  input it sees is genuinely a property of the sentence being scored. Trained on
  sentences inheriting their document's label.
* **Document model** — consumes document-scoped features (rhythm, whole-essay
  lexis) *plus* summary statistics of the sentence scores. It therefore sees
  both the shape of the essay and the distribution of local evidence.

Why the document model does not simply average sentence scores
--------------------------------------------------------------
Averaging throws away the shape of the distribution, which is exactly what
distinguishes the interesting cases. An essay with every sentence at 0.5 and an
essay with half its sentences at 0.1 and half at 0.9 have the same mean and
completely different stories — the second looks like partial machine editing,
which is the realistic threat. The summary features below (mean, spread,
quantiles, high-risk mass) preserve that shape, and the model learns what to do
with it rather than us guessing.

Risk versus confidence
----------------------
These are separate outputs and must stay separate. Risk is the model's estimate
that the text is machine-generated. Confidence is how much evidence supports
that estimate — driven by how much scorable text there was and how consistent
the local signals are. A short essay of unusual sentences can legitimately
produce high risk with low confidence, and saying so is more honest than
reporting a single number that conflates them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from app.ml.feature_matrix import FeatureMatrixBuilder

__all__ = [
    "SENTENCE_SCORE_SUMMARY_FEATURES",
    "summarize_sentence_scores",
    "DetectorArtifacts",
    "confidence_from_evidence",
]

# Quantiles of the per-sentence score distribution used as document features.
_SCORE_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)

# A sentence at or above this score counts toward the "high risk mass" feature.
HIGH_RISK_THRESHOLD = 0.60

# Below this many scorable sentences, document-level evidence is thin and
# confidence is penalized accordingly.
SUFFICIENT_SENTENCE_COUNT = 12

SENTENCE_SCORE_SUMMARY_FEATURES: list[str] = (
    ["sentscore_mean", "sentscore_std", "sentscore_min", "sentscore_max"]
    + [f"sentscore_q{int(quantile * 100):02d}" for quantile in _SCORE_QUANTILES]
    + ["sentscore_high_risk_fraction", "sentscore_scorable_count",
       "sentscore_range", "sentscore_top_quartile_mean"]
)


def summarize_sentence_scores(scores: np.ndarray) -> dict[str, float]:
    """Reduce a document's sentence scores to distribution-shape features."""
    if scores.size == 0:
        return {name: 0.0 for name in SENTENCE_SCORE_SUMMARY_FEATURES}

    quantile_values = np.quantile(scores, _SCORE_QUANTILES)
    top_quartile_threshold = np.quantile(scores, 0.75)
    top_quartile = scores[scores >= top_quartile_threshold]

    summary = {
        "sentscore_mean": float(scores.mean()),
        "sentscore_std": float(scores.std()) if scores.size > 1 else 0.0,
        "sentscore_min": float(scores.min()),
        "sentscore_max": float(scores.max()),
        "sentscore_high_risk_fraction": float((scores >= HIGH_RISK_THRESHOLD).mean()),
        "sentscore_scorable_count": float(scores.size),
        "sentscore_range": float(scores.max() - scores.min()),
        "sentscore_top_quartile_mean": float(top_quartile.mean()),
    }
    for quantile, value in zip(_SCORE_QUANTILES, quantile_values):
        summary[f"sentscore_q{int(quantile * 100):02d}"] = float(value)
    return summary


def confidence_from_evidence(
    scores: np.ndarray, scorable_count: int, total_count: int
) -> float:
    """Confidence in [0, 1], computed from evidence quantity and consistency.

    Three multiplicative components, each in [0, 1]:

    * **Quantity** — how much scorable text there was, saturating at
      SUFFICIENT_SENTENCE_COUNT sentences. Two sentences cannot support a
      confident verdict regardless of how extreme they look.
    * **Coverage** — the share of sentences that were scorable at all. An essay
      of mostly fragments yields little to measure.
    * **Consistency** — how much the sentence scores agree. Scores scattered
      across the whole range mean the local evidence is contradictory.

    This is a defined function of measurable quantities, not a second model
    output, so it can be explained exactly. It is deliberately *not* calibrated
    as a probability, and the UI presents it as a qualitative band.
    """
    if scorable_count == 0 or scores.size == 0:
        return 0.0

    quantity = min(1.0, scorable_count / SUFFICIENT_SENTENCE_COUNT)
    coverage = scorable_count / total_count if total_count else 0.0

    # Standard deviation of scores in [0,1] is at most 0.5, so doubling maps
    # maximal disagreement to 1.0 and perfect agreement to 0.0.
    disagreement = min(1.0, float(scores.std()) * 2.0) if scores.size > 1 else 0.0
    consistency = 1.0 - disagreement

    return float(quantity * coverage * consistency)


@dataclass
class DetectorArtifacts:
    """Everything the API needs to score an essay, loadable without pickle."""

    sentence_builder: FeatureMatrixBuilder
    sentence_model: LogisticRegression
    document_builder: FeatureMatrixBuilder
    document_model: LogisticRegression
    metadata: dict[str, object]

    def score_sentences(self, sentence_frame: pd.DataFrame) -> np.ndarray:
        """Per-sentence probability of being machine-generated."""
        if sentence_frame.empty:
            return np.empty(0)
        matrix = self.sentence_builder.transform(sentence_frame)
        return self.sentence_model.predict_proba(matrix)[:, 1]

    def score_document(self, document_frame: pd.DataFrame) -> np.ndarray:
        """Per-document probability of being machine-generated."""
        matrix = self.document_builder.transform(document_frame)
        return self.document_model.predict_proba(matrix)[:, 1]

    def sentence_contributions(
        self, sentence_frame: pd.DataFrame
    ) -> list[dict[str, float]]:
        """Signed per-feature contributions to each sentence's log-odds.

        This is what the evidence panel renders. Because inputs are standardized
        and the model is linear, `coefficient * standardized_value` is directly
        the number of log-odds that feature contributed for this sentence —
        an actual decomposition of the score, not a plausible-sounding story
        generated after the fact.
        """
        matrix = self.sentence_builder.transform(sentence_frame)
        coefficients = self.sentence_model.coef_[0]
        names = self.sentence_builder.column_names

        contributions: list[dict[str, float]] = []
        for row in matrix:
            contributions.append(
                {name: float(value * coefficient)
                 for name, value, coefficient in zip(names, row, coefficients)}
            )
        return contributions

    def save(self, directory: Path) -> None:
        """Persist artifacts as JSON, avoiding pickle entirely."""
        directory.mkdir(parents=True, exist_ok=True)
        self.sentence_builder.save(directory / "sentence_builder.json")
        self.document_builder.save(directory / "document_builder.json")

        for name, model in (
            ("sentence_model", self.sentence_model),
            ("document_model", self.document_model),
        ):
            (directory / f"{name}.json").write_text(
                json.dumps(
                    {
                        "coefficients": model.coef_[0].tolist(),
                        "intercept": float(model.intercept_[0]),
                        "classes": model.classes_.tolist(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        (directory / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> DetectorArtifacts:
        """Rebuild the detector from JSON artifacts."""
        def _restore(name: str) -> LogisticRegression:
            payload = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
            model = LogisticRegression()
            model.coef_ = np.array([payload["coefficients"]])
            model.intercept_ = np.array([payload["intercept"]])
            model.classes_ = np.array(payload["classes"])
            # scikit-learn requires this attribute for predict_proba on a model
            # that was not fitted in this process.
            model.n_features_in_ = len(payload["coefficients"])
            return model

        return cls(
            sentence_builder=FeatureMatrixBuilder.load(directory / "sentence_builder.json"),
            sentence_model=_restore("sentence_model"),
            document_builder=FeatureMatrixBuilder.load(directory / "document_builder.json"),
            document_model=_restore("document_model"),
            metadata=json.loads((directory / "metadata.json").read_text(encoding="utf-8")),
        )
