"""Tests for the feature matrix builder and the detector container.

The recurring theme: anything fitted on training data must be applied
*unchanged* afterwards. A builder that recomputes its medians at transform time
would leak test statistics into test predictions and would not fail loudly, so
it is pinned by tests here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from app.ml.detector import (
    SENTENCE_SCORE_SUMMARY_FEATURES,
    DetectorArtifacts,
    confidence_from_evidence,
    summarize_sentence_scores,
)
from app.ml.feature_matrix import FeatureMatrixBuilder


def _training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "essay_id": ["a", "b", "c", "d"],
            "label": [0, 1, 0, 1],
            "split": ["train"] * 4,
            "alpha": [1.0, 2.0, 3.0, 4.0],
            "beta": [10.0, np.nan, 30.0, 40.0],
            "constant": [5.0, 5.0, 5.0, 5.0],
            "doc_token_count": [100.0, 200.0, 300.0, 400.0],
        }
    )


# --------------------------------------------------------------------------
# FeatureMatrixBuilder
# --------------------------------------------------------------------------


def test_metadata_columns_are_never_treated_as_features() -> None:
    builder = FeatureMatrixBuilder().fit(_training_frame())
    for column in ("essay_id", "label", "split"):
        assert column not in builder.feature_names


def test_constant_features_are_dropped() -> None:
    """A constant column contributes nothing but dilutes coefficient ranking."""
    builder = FeatureMatrixBuilder().fit(_training_frame())
    assert "constant" not in builder.feature_names


def test_length_artefacts_excluded_by_default_and_kept_on_request() -> None:
    excluded = FeatureMatrixBuilder(exclude_length_artefacts=True).fit(_training_frame())
    included = FeatureMatrixBuilder(exclude_length_artefacts=False).fit(_training_frame())
    assert "doc_token_count" not in excluded.feature_names
    assert "doc_token_count" in included.feature_names


def test_missingness_indicator_created_only_for_columns_with_gaps() -> None:
    builder = FeatureMatrixBuilder().fit(_training_frame())
    assert "beta__missing" in builder.indicator_names
    assert "alpha__missing" not in builder.indicator_names


def test_transform_output_shape_matches_column_names() -> None:
    frame = _training_frame()
    builder = FeatureMatrixBuilder().fit(frame)
    matrix = builder.transform(frame)
    assert matrix.shape == (len(frame), len(builder.column_names))


def test_standardization_produces_zero_mean_unit_variance_on_train() -> None:
    frame = _training_frame()
    builder = FeatureMatrixBuilder().fit(frame)
    matrix = builder.transform(frame)
    alpha_column = matrix[:, builder.feature_names.index("alpha")]
    assert alpha_column.mean() == pytest.approx(0.0, abs=1e-9)
    assert alpha_column.std() == pytest.approx(1.0, abs=1e-9)


def test_imputation_uses_the_training_median_not_the_incoming_batch() -> None:
    """The leak this guards against would be invisible in ordinary use."""
    train = _training_frame()
    builder = FeatureMatrixBuilder().fit(train)
    training_median = builder.medians["beta"]

    # A later batch with a wildly different distribution and a missing value.
    later = pd.DataFrame(
        {
            "alpha": [2.5, 2.5],
            "beta": [np.nan, 9000.0],
            "constant": [5.0, 5.0],
            "doc_token_count": [1.0, 1.0],
        }
    )
    matrix = builder.transform(later)
    beta_index = builder.feature_names.index("beta")
    imputed_standardized = matrix[0, beta_index]
    expected = (training_median - builder.means["beta"]) / builder.standard_deviations["beta"]
    assert imputed_standardized == pytest.approx(expected)


def test_missing_indicator_is_set_for_absent_values() -> None:
    train = _training_frame()
    builder = FeatureMatrixBuilder().fit(train)
    later = pd.DataFrame(
        {"alpha": [1.0], "beta": [np.nan], "constant": [5.0], "doc_token_count": [1.0]}
    )
    matrix = builder.transform(later)
    indicator_index = len(builder.feature_names) + builder.indicator_names.index(
        "beta__missing"
    )
    assert matrix[0, indicator_index] == 1.0


def test_column_order_is_stable_across_transforms() -> None:
    frame = _training_frame()
    builder = FeatureMatrixBuilder().fit(frame)
    first = builder.column_names
    builder.transform(frame)
    assert builder.column_names == first


def test_transform_before_fit_raises() -> None:
    with pytest.raises(ValueError, match="must be fitted"):
        FeatureMatrixBuilder().transform(_training_frame())


def test_builder_round_trips_through_disk(tmp_path) -> None:
    frame = _training_frame()
    builder = FeatureMatrixBuilder().fit(frame)
    expected = builder.transform(frame)

    path = tmp_path / "builder.json"
    builder.save(path)
    reloaded = FeatureMatrixBuilder.load(path)

    assert reloaded.column_names == builder.column_names
    np.testing.assert_allclose(reloaded.transform(frame), expected)


def test_unfitted_builder_refuses_to_save(tmp_path) -> None:
    with pytest.raises(ValueError):
        FeatureMatrixBuilder().save(tmp_path / "nothing.json")


# --------------------------------------------------------------------------
# sentence-score summaries
# --------------------------------------------------------------------------


def test_summary_returns_every_declared_feature() -> None:
    summary = summarize_sentence_scores(np.array([0.1, 0.5, 0.9]))
    assert set(summary) == set(SENTENCE_SCORE_SUMMARY_FEATURES)


def test_summary_of_empty_input_is_all_zero() -> None:
    summary = summarize_sentence_scores(np.array([]))
    assert set(summary) == set(SENTENCE_SCORE_SUMMARY_FEATURES)
    assert all(value == 0.0 for value in summary.values())


def test_summary_distinguishes_distributions_that_share_a_mean() -> None:
    """The reason the document model does not simply average sentence scores.

    A uniformly middling essay and a half-rewritten one have the same mean and
    entirely different stories. Averaging cannot tell them apart; the shape
    features must.
    """
    uniform = summarize_sentence_scores(np.full(10, 0.5))
    polarized = summarize_sentence_scores(np.array([0.1] * 5 + [0.9] * 5))

    assert uniform["sentscore_mean"] == pytest.approx(polarized["sentscore_mean"])
    assert polarized["sentscore_std"] > uniform["sentscore_std"]
    assert polarized["sentscore_range"] > uniform["sentscore_range"]
    assert polarized["sentscore_high_risk_fraction"] > uniform["sentscore_high_risk_fraction"]


def test_high_risk_fraction_counts_sentences_above_the_threshold() -> None:
    summary = summarize_sentence_scores(np.array([0.2, 0.4, 0.7, 0.95]))
    assert summary["sentscore_high_risk_fraction"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# confidence
# --------------------------------------------------------------------------


def test_confidence_is_zero_without_scorable_sentences() -> None:
    assert confidence_from_evidence(np.array([]), 0, 10) == 0.0


def test_confidence_rises_with_more_evidence() -> None:
    consistent = np.full(4, 0.8)
    few = confidence_from_evidence(consistent, 4, 4)
    many = confidence_from_evidence(np.full(20, 0.8), 20, 20)
    assert many > few


def test_confidence_falls_when_sentences_disagree() -> None:
    agreeing = confidence_from_evidence(np.full(20, 0.8), 20, 20)
    disagreeing = confidence_from_evidence(
        np.array([0.05] * 10 + [0.95] * 10), 20, 20
    )
    assert disagreeing < agreeing


def test_confidence_falls_when_most_sentences_are_unscorable() -> None:
    scores = np.full(6, 0.8)
    full_coverage = confidence_from_evidence(scores, 6, 6)
    poor_coverage = confidence_from_evidence(scores, 6, 30)
    assert poor_coverage < full_coverage


def test_confidence_stays_within_unit_range() -> None:
    for scores, scorable, total in (
        (np.full(50, 0.9), 50, 50),
        (np.array([0.0, 1.0]), 2, 2),
        (np.full(3, 0.5), 3, 100),
    ):
        value = confidence_from_evidence(scores, scorable, total)
        assert 0.0 <= value <= 1.0


def test_high_risk_with_low_confidence_is_representable() -> None:
    """The product claim in the brief: risk and confidence are independent."""
    scores = np.array([0.92, 0.88])
    risk = float(scores.mean())
    confidence = confidence_from_evidence(scores, 2, 20)
    assert risk > 0.85
    assert confidence < 0.2


# --------------------------------------------------------------------------
# DetectorArtifacts
# --------------------------------------------------------------------------


def _fitted_artifacts() -> tuple[DetectorArtifacts, pd.DataFrame]:
    frame = pd.DataFrame(
        {
            "essay_id": [f"e{index}" for index in range(20)],
            "label": [index % 2 for index in range(20)],
            "alpha": np.linspace(0.0, 1.0, 20),
            "beta": np.linspace(1.0, 0.0, 20),
        }
    )
    builder = FeatureMatrixBuilder().fit(frame)
    model = LogisticRegression(max_iter=1000).fit(
        builder.transform(frame), frame["label"].to_numpy()
    )
    artifacts = DetectorArtifacts(
        sentence_builder=builder,
        sentence_model=model,
        document_builder=builder,
        document_model=model,
        metadata={"note": "test"},
    )
    return artifacts, frame


def test_scores_are_probabilities() -> None:
    artifacts, frame = _fitted_artifacts()
    scores = artifacts.score_sentences(frame)
    assert scores.shape == (len(frame),)
    assert ((scores >= 0.0) & (scores <= 1.0)).all()


def test_scoring_an_empty_frame_returns_an_empty_array() -> None:
    artifacts, _ = _fitted_artifacts()
    assert artifacts.score_sentences(pd.DataFrame()).size == 0


def test_contributions_decompose_the_log_odds() -> None:
    """The evidence panel's honesty depends on this identity holding exactly.

    For a linear model on standardized inputs, the contributions must sum to
    the log-odds minus the intercept. If they did not, the panel would be
    showing a plausible story rather than the actual decomposition.
    """
    artifacts, frame = _fitted_artifacts()
    contributions = artifacts.sentence_contributions(frame)
    scores = artifacts.score_sentences(frame)

    for row_contributions, score in zip(contributions, scores):
        log_odds = np.log(score / (1.0 - score))
        total = sum(row_contributions.values()) + float(
            artifacts.sentence_model.intercept_[0]
        )
        assert total == pytest.approx(log_odds, abs=1e-6)


def test_artifacts_round_trip_through_disk(tmp_path) -> None:
    artifacts, frame = _fitted_artifacts()
    expected = artifacts.score_document(frame)

    artifacts.save(tmp_path / "model")
    reloaded = DetectorArtifacts.load(tmp_path / "model")

    np.testing.assert_allclose(reloaded.score_document(frame), expected)
    assert reloaded.metadata["note"] == "test"


def test_saved_artifacts_contain_no_pickle(tmp_path) -> None:
    """Model loading must not be able to execute code."""
    artifacts, _ = _fitted_artifacts()
    artifacts.save(tmp_path / "model")
    written = sorted(path.name for path in (tmp_path / "model").iterdir())
    assert written == [
        "document_builder.json",
        "document_model.json",
        "metadata.json",
        "sentence_builder.json",
        "sentence_model.json",
    ]
