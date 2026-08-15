"""Tests for evidence generation.

These guard the property the evidence panel depends on: what it says must
follow from the measured contributions, including when those contributions
disagree with each other.
"""

from __future__ import annotations

import pytest

from app.services.evidence import (
    ReferenceStatistics,
    SignalSummary,
    build_explanation,
    family_for,
    summarize_signals,
)


def test_aggregation_prefixes_are_stripped_before_family_lookup() -> None:
    """docmean_punct_* is punctuation, not a generic document feature."""
    assert family_for("docmean_punct_comma_density") == "Punctuation"
    assert family_for("docstd_lex_mtld") == "Lexical diversity"
    assert family_for("punct_comma_density") == "Punctuation"


def test_missing_suffix_does_not_change_family() -> None:
    assert family_for("docmean_lex_mtld__missing") == "Lexical diversity"


def test_families_are_not_collapsed_into_one_bucket() -> None:
    """Regression: every doc-prefixed feature once mapped to a single family."""
    names = [
        "docmean_punct_comma_density",
        "docmean_lex_ttr",
        "docmean_syn_mean_dependency_depth",
        "docmean_form_hedge_density",
    ]
    assert len({family_for(name) for name in names}) == 4


def test_z_score_is_none_without_reference_data() -> None:
    reference = ReferenceStatistics(means={}, standard_deviations={})
    assert reference.z_score("anything", 1.0) is None


def test_z_score_is_none_for_zero_variance_feature() -> None:
    reference = ReferenceStatistics(
        means={"flat": 1.0}, standard_deviations={"flat": 0.0}
    )
    assert reference.z_score("flat", 5.0) is None


def test_z_score_computed_against_the_human_reference() -> None:
    reference = ReferenceStatistics(
        means={"punct_comma_density": 0.03},
        standard_deviations={"punct_comma_density": 0.01},
    )
    assert reference.z_score("punct_comma_density", 0.05) == pytest.approx(2.0)


def test_negligible_contributions_are_not_reported() -> None:
    """Padding the panel with near-zero families would look like reasoning."""
    reference = ReferenceStatistics(means={}, standard_deviations={})
    summaries = summarize_signals(
        {"punct_comma_density": 0.001, "lex_ttr": 1.5}, {}, reference
    )
    assert [summary.family for summary in summaries] == ["Lexical diversity"]


def test_empty_summaries_produce_an_honest_message() -> None:
    explanation = build_explanation([], risk=0.5)
    assert "No individual signal" in explanation


def test_explanation_never_names_one_family_as_both_cause_and_counterevidence() -> None:
    """Regression: the leading family must point toward the verdict.

    Ranking by magnitude alone produced text that said a score was "driven
    mainly by lexical diversity" and then, in the same breath, that lexical
    diversity was "pushing the other way".
    """
    summaries = [
        SignalSummary("Lexical diversity", -1.69, "human", "lex_ttr", 0.9, -0.1, "close to the human average"),
        SignalSummary("Sentence length", 1.46, "machine", "struct_token_count", 5.4, -0.1, "close to the human average"),
        SignalSummary("Syntactic structure", 1.33, "machine", "syn_mean_dependency_depth", 4.2, 1.8, "noticeably above the human average"),
    ]
    explanation = build_explanation(summaries, risk=0.78)

    driver = explanation.split("driven mainly by ")[1].split(",")[0]
    counter = explanation.split("Pushing the other way: ")[1].split(",")[0]
    assert driver != counter
    assert driver == "sentence length"
    assert counter == "lexical diversity"


def test_explanation_handles_all_families_opposing_the_verdict() -> None:
    """No supporting family should not produce a contradictory sentence."""
    summaries = [
        SignalSummary("Punctuation", -0.8, "human", "punct_comma_density", 0.03, -1.2, "noticeably below the human average"),
    ]
    explanation = build_explanation(summaries, risk=0.62)
    assert "driven mainly by" not in explanation
    assert "many small effects" in explanation
    assert "Pushing the other way: punctuation" in explanation
