"""Metrics and document-frame assembly, shared by training and evaluation.

Kept in one place so that a number reported during training and the same number
reported at test time are produced by identical code. Two separate
implementations that drift apart is a quiet way to publish a metric nobody can
reproduce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.ml.detector import SENTENCE_SCORE_SUMMARY_FEATURES, summarize_sentence_scores

__all__ = [
    "classification_metrics",
    "reliability_table",
    "expected_calibration_error",
    "build_document_frame",
    "wilson_interval",
]


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Standard metrics plus the confusion counts they derive from.

    The raw counts are included so that any rate in the report can be checked
    by hand, and so that a rate computed on a handful of essays is visibly
    computed on a handful of essays.
    """
    predictions = (probabilities >= threshold).astype(int)
    confusion = {
        "true_negative": int(((labels == 0) & (predictions == 0)).sum()),
        "false_positive": int(((labels == 0) & (predictions == 1)).sum()),
        "false_negative": int(((labels == 1) & (predictions == 0)).sum()),
        "true_positive": int(((labels == 1) & (predictions == 1)).sum()),
    }
    negatives = confusion["true_negative"] + confusion["false_positive"]
    positives = confusion["true_positive"] + confusion["false_negative"]

    # AUC is undefined when only one class is present, which happens on the
    # ELL slice (all human). Reported as NaN rather than crashing or, worse,
    # returning a misleading 0.5.
    both_classes_present = len(np.unique(labels)) > 1

    return {
        "count": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities))
        if both_classes_present
        else float("nan"),
        "brier": float(brier_score_loss(labels, probabilities))
        if both_classes_present
        else float("nan"),
        "false_positive_rate": confusion["false_positive"] / negatives
        if negatives
        else float("nan"),
        "false_negative_rate": confusion["false_negative"] / positives
        if positives
        else float("nan"),
        **confusion,
    }


def wilson_interval(
    successes: int, trials: int, z_score: float = 1.96
) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion.

    Used for false-positive rates on small slices. The normal approximation
    misbehaves badly near 0 and 1 and on small samples — exactly the regime the
    ELL slice sits in — and would produce intervals that overstate certainty.
    """
    if trials == 0:
        return (float("nan"), float("nan"))

    proportion = successes / trials
    denominator = 1.0 + z_score**2 / trials
    centre = proportion + z_score**2 / (2 * trials)
    spread = z_score * np.sqrt(
        proportion * (1 - proportion) / trials + z_score**2 / (4 * trials**2)
    )
    return (
        float(max(0.0, (centre - spread) / denominator)),
        float(min(1.0, (centre + spread) / denominator)),
    )


def reliability_table(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> list[dict[str, float]]:
    """Observed frequency against predicted probability, per bin.

    This decides the wording in the UI. If observed frequency tracks the
    prediction, the output is a probability and may be described as one; if it
    does not, it is an ordering and must be presented as a risk score.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float]] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= lower) & (
            probabilities < upper if upper < 1.0 else probabilities <= upper
        )
        if not mask.any():
            continue
        rows.append(
            {
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "count": int(mask.sum()),
                "mean_predicted": float(probabilities[mask].mean()),
                "observed_frequency": float(labels[mask].mean()),
            }
        )
    return rows


def expected_calibration_error(rows: list[dict[str, float]], total: int) -> float:
    """Weighted mean gap between predicted probability and observed frequency."""
    if not rows or total == 0:
        return float("nan")
    return float(
        sum(
            row["count"] * abs(row["mean_predicted"] - row["observed_frequency"])
            for row in rows
        )
        / total
    )


def build_document_frame(
    documents: pd.DataFrame, sentences: pd.DataFrame, sentence_scores: np.ndarray
) -> pd.DataFrame:
    """Attach sentence-score distribution summaries to each document row.

    Only scorable sentences contribute. Including fragments would let a run of
    two-word lines drag a document's summary toward whatever the model happens
    to output for text it was never meant to judge.
    """
    scored = sentences.assign(_score=sentence_scores)
    summaries: list[dict[str, object]] = []

    for essay_id, group in scored.groupby("essay_id", sort=False):
        scorable = group[group["is_scorable"]]
        summaries.append(
            {
                "essay_id": essay_id,
                **summarize_sentence_scores(scorable["_score"].to_numpy()),
                "total_sentence_count": float(len(group)),
            }
        )

    merged = documents.merge(pd.DataFrame(summaries), on="essay_id", how="inner")

    missing = [name for name in SENTENCE_SCORE_SUMMARY_FEATURES if name not in merged]
    if missing:
        raise ValueError(f"Sentence-score summary features missing: {missing}")
    return merged
