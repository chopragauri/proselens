"""Tests for feature extraction.

Two things are being checked throughout: that features are *correct* on inputs
whose answer we can work out by hand, and that they are *undefined rather than
misleading* on degenerate inputs. The second matters more for this project —
a feature that silently returns 0.0 for a case it cannot measure produces an
evidence panel that lies to the user.
"""

from __future__ import annotations

import pytest

from app.nlp.features.aggregate import (
    PASSAGE_WINDOW_SIZE,
    extract_features,
    extract_window_features,
)
from app.nlp.features.base import safe_divide
from app.nlp.features.lexical import extract_lexical_features, mtld
from app.nlp.features.predictability import NgramLanguageModel
from app.nlp.features.punctuation import extract_punctuation_features
from app.nlp.features.repetition import extract_repetition_features
from app.nlp.features.structure import (
    compute_document_statistics,
    extract_document_rhythm_features,
)
from app.nlp.features.formulaic import extract_formulaic_features
from app.nlp.pipeline import get_nlp
from app.nlp.preprocessing import normalize
from app.nlp.segmentation import segment

HUMAN_STYLE_ESSAY = (
    "I burned the dumplings. Every single one, blackened on the bottom while "
    "my grandmother pretended not to notice from across our narrow kitchen in "
    "Queens.\n"
    "\n"
    "She handed me another wrapper. I tried again, and again, and the "
    "fourteenth one held. That is the whole lesson, really."
)

MACHINE_STYLE_ESSAY = (
    "Throughout my academic journey, I have consistently demonstrated a "
    "commitment to excellence. Moreover, my involvement in extracurricular "
    "activities has significantly enhanced my leadership capabilities. "
    "Furthermore, these experiences have provided me with valuable insights "
    "into collaboration. Additionally, my dedication to community service "
    "reflects my genuine passion for helping others."
)


@pytest.fixture(scope="module")
def nlp():
    return get_nlp()


def _parse_one(nlp, text: str):
    return next(nlp.pipe([text]))


# --------------------------------------------------------------------------
# base
# --------------------------------------------------------------------------


def test_safe_divide_returns_none_on_zero_denominator() -> None:
    assert safe_divide(5, 0) is None
    assert safe_divide(0, 0) is None
    assert safe_divide(3, 4) == 0.75


# --------------------------------------------------------------------------
# lexical
# --------------------------------------------------------------------------


def test_type_token_ratio_is_computed_by_hand(nlp) -> None:
    parsed = _parse_one(nlp, "the cat sat on the mat")
    features = extract_lexical_features(parsed)
    # 6 tokens, 5 unique ("the" twice).
    assert features["lex_token_count"] == 6
    assert features["lex_unique_count"] == 5
    assert features["lex_ttr"] == pytest.approx(5 / 6)
    assert features["lex_repeated_word_ratio"] == pytest.approx(1 / 6)


def test_lexical_features_are_undefined_for_empty_input(nlp) -> None:
    features = extract_lexical_features(_parse_one(nlp, "..."))
    assert features["lex_ttr"] is None
    assert features["lex_mtld"] is None
    assert features["lex_mean_word_length"] is None


def test_mtld_is_undefined_for_short_text() -> None:
    assert mtld(["one", "two", "three"]) is None


def test_mtld_ranks_repetitive_text_below_diverse_text() -> None:
    """The property that makes MTLD worth having over raw TTR."""
    repetitive = ["the", "cat", "the", "cat"] * 10
    diverse = [f"word{index}" for index in range(40)]
    assert mtld(repetitive) is not None
    assert mtld(diverse) is not None
    assert mtld(repetitive) < mtld(diverse)


def test_mtld_of_fully_diverse_text_is_maximal_not_missing() -> None:
    """Regression: text with no repeated word completes no MTLD factor.

    An earlier version returned None here, treating maximal diversity as an
    undefined measurement. Because sentences with no repeated word are
    disproportionately human, that turned the missingness indicator into a
    label proxy and would have inflated every downstream metric.
    """
    all_unique = [f"word{index}" for index in range(25)]
    score = mtld(all_unique)
    assert score is not None
    assert score == pytest.approx(len(all_unique))


def test_mtld_is_length_invariant_where_ttr_is_not() -> None:
    """The core justification for including MTLD at all.

    Doubling a text's length leaves its lexical diversity unchanged in
    substance. TTR drops sharply; MTLD should not.
    """
    unit = [f"word{index}" for index in range(20)]
    short, long_text = unit, unit * 3

    short_ttr = len(set(short)) / len(short)
    long_ttr = len(set(long_text)) / len(long_text)
    assert long_ttr < short_ttr / 2  # TTR collapses with length

    short_mtld, long_mtld = mtld(short), mtld(long_text)
    assert short_mtld is not None and long_mtld is not None
    # MTLD stays the same order of magnitude rather than collapsing.
    assert long_mtld > short_mtld * 0.5


# --------------------------------------------------------------------------
# punctuation
# --------------------------------------------------------------------------


def test_punctuation_densities_counted_by_hand(nlp) -> None:
    parsed = _parse_one(nlp, "Yes, I agree; however, it is complex.")
    features = extract_punctuation_features(parsed)
    assert features["punct_comma_density"] is not None
    assert features["punct_semicolon_density"] is not None
    assert features["punct_comma_density"] > features["punct_semicolon_density"]


def test_absent_punctuation_is_zero_not_none(nlp) -> None:
    """Zero and undefined mean different things and must not be conflated."""
    features = extract_punctuation_features(_parse_one(nlp, "no punctuation here"))
    assert features["punct_semicolon_density"] == 0.0
    assert features["punct_comma_density"] == 0.0


def test_ellipsis_counted_separately_from_periods(nlp) -> None:
    features = extract_punctuation_features(_parse_one(nlp, "I waited... then left."))
    assert features["punct_ellipsis_count"] == 1.0


# --------------------------------------------------------------------------
# repetition
# --------------------------------------------------------------------------


def test_repeated_bigrams_detected(nlp) -> None:
    parsed = _parse_one(nlp, "we must act now we must act now")
    features = extract_repetition_features(parsed)
    assert features["rep_repeated_bigram_ratio"] > 0.5


def test_no_repetition_scores_zero(nlp) -> None:
    parsed = _parse_one(nlp, "each word here appears exactly once truly")
    features = extract_repetition_features(parsed)
    assert features["rep_repeated_bigram_ratio"] == 0.0


def test_adjacent_duplicates_detected(nlp) -> None:
    features = extract_repetition_features(_parse_one(nlp, "it was very very good"))
    assert features["rep_adjacent_duplicate_ratio"] > 0


def test_template_echo_detects_shared_openings(nlp) -> None:
    """Machine essays reuse sentence templates; this is what catches it."""
    document = segment(normalize(MACHINE_STYLE_ESSAY))
    features = extract_features(document)
    echoes = [
        sentence.values["repctx_template_echo"]
        for sentence in features.sentence_features
    ]
    assert any(echo is not None and echo > 0 for echo in echoes)


# --------------------------------------------------------------------------
# formulaic
# --------------------------------------------------------------------------


def test_connective_opening_detected_structurally(nlp) -> None:
    features = extract_formulaic_features(
        _parse_one(nlp, "Moreover, the results were encouraging.")
    )
    assert features["form_opens_with_connective"] == 1.0


def test_plain_opening_is_not_flagged(nlp) -> None:
    features = extract_formulaic_features(
        _parse_one(nlp, "I dropped the pan on the floor.")
    )
    assert features["form_opens_with_connective"] == 0.0


def test_fronted_adverbial_detected_without_a_phrase_list(nlp) -> None:
    """Fires on a construction that appears in no word list in this repo."""
    features = extract_formulaic_features(
        _parse_one(nlp, "Across my many varied pursuits, I have grown considerably.")
    )
    assert features["form_has_fronted_adverbial"] == 1.0


def test_nominalization_density_detects_abstract_register(nlp) -> None:
    abstract = extract_formulaic_features(
        _parse_one(nlp, "The implementation of the transformation required dedication.")
    )
    concrete = extract_formulaic_features(
        _parse_one(nlp, "I burned the rice and the kitchen filled with smoke.")
    )
    assert abstract["form_nominalization_density"] > concrete["form_nominalization_density"]


def test_specificity_density_rewards_concrete_detail(nlp) -> None:
    specific = extract_formulaic_features(
        _parse_one(nlp, "In 2019 my brother Arun moved to Chennai.")
    )
    generic = extract_formulaic_features(
        _parse_one(nlp, "A family member relocated to a different city.")
    )
    assert specific["form_specificity_density"] > generic["form_specificity_density"]


# --------------------------------------------------------------------------
# structure and rhythm
# --------------------------------------------------------------------------


def test_document_statistics_computed_by_hand() -> None:
    stats = compute_document_statistics([10, 20, 30])
    assert stats.sentence_count == 3
    assert stats.mean_length == pytest.approx(20.0)
    assert stats.standard_deviation == pytest.approx(8.16496, rel=1e-4)


def test_rhythm_undefined_for_too_few_sentences() -> None:
    features = extract_document_rhythm_features([12, 14])
    assert features["rhythm_burstiness"] is None
    assert features["rhythm_coefficient_of_variation"] is None


def test_uniform_lengths_produce_minimal_variation() -> None:
    features = extract_document_rhythm_features([20, 20, 20, 20, 20])
    assert features["rhythm_coefficient_of_variation"] == 0.0
    assert features["rhythm_mean_successive_difference"] == 0.0
    # Perfectly regular text sits at the floor of the burstiness range.
    assert features["rhythm_burstiness"] == pytest.approx(-1.0)


def test_varied_lengths_score_higher_than_uniform_ones() -> None:
    uniform = extract_document_rhythm_features([20, 20, 20, 20, 20, 20])
    varied = extract_document_rhythm_features([4, 35, 9, 28, 6, 41])
    assert varied["rhythm_coefficient_of_variation"] > uniform["rhythm_coefficient_of_variation"]
    assert varied["rhythm_burstiness"] > uniform["rhythm_burstiness"]


def test_successive_difference_distinguishes_what_variance_cannot() -> None:
    """The reason this feature exists alongside the variance-based ones.

    Both series have identical mean, identical variance and therefore identical
    burstiness. Only the alternation differs, and only this feature sees it.
    """
    alternating = extract_document_rhythm_features([10, 30, 10, 30, 10, 30])
    grouped = extract_document_rhythm_features([10, 10, 10, 30, 30, 30])

    assert alternating["rhythm_coefficient_of_variation"] == pytest.approx(
        grouped["rhythm_coefficient_of_variation"]
    )
    assert alternating["rhythm_burstiness"] == pytest.approx(grouped["rhythm_burstiness"])
    assert alternating["rhythm_mean_successive_difference"] > (
        grouped["rhythm_mean_successive_difference"]
    )


# --------------------------------------------------------------------------
# predictability
# --------------------------------------------------------------------------


def test_language_model_returns_none_before_fitting() -> None:
    model = NgramLanguageModel()
    assert model.is_fitted is False
    assert model.cross_entropy(["some", "words"]) is None


def test_language_model_scores_seen_text_as_more_predictable() -> None:
    training = [
        ["i", "walked", "to", "the", "store", "yesterday"],
        ["i", "walked", "to", "the", "park", "today"],
        ["i", "walked", "to", "the", "store", "again"],
    ] * 5
    model = NgramLanguageModel().fit(training)

    familiar = model.cross_entropy(["i", "walked", "to", "the", "store"])
    unfamiliar = model.cross_entropy(["quantum", "chromodynamics", "bewilders", "me"])

    assert familiar is not None and unfamiliar is not None
    assert familiar < unfamiliar


def test_language_model_cross_entropy_is_finite_for_unseen_tokens() -> None:
    """Smoothing must guarantee this; an -inf would poison the feature vector."""
    model = NgramLanguageModel().fit([["a", "b", "c"]] * 3)
    entropy = model.cross_entropy(["totally", "unseen", "vocabulary"])
    assert entropy is not None
    assert entropy > 0
    assert entropy != float("inf")


def test_language_model_is_deterministic() -> None:
    training = [["the", "quick", "brown", "fox"], ["the", "lazy", "brown", "dog"]] * 4
    first = NgramLanguageModel().fit(training).cross_entropy(["the", "brown", "fox"])
    second = NgramLanguageModel().fit(training).cross_entropy(["the", "brown", "fox"])
    assert first == second


def test_language_model_round_trips_through_disk(tmp_path) -> None:
    training = [["alpha", "beta", "gamma"], ["alpha", "beta", "delta"]] * 4
    model = NgramLanguageModel().fit(training)
    probe = ["alpha", "beta", "gamma"]
    expected = model.cross_entropy(probe)

    path = tmp_path / "lm.json"
    model.save(path)
    reloaded = NgramLanguageModel.load(path)

    assert reloaded.is_fitted
    assert reloaded.cross_entropy(probe) == pytest.approx(expected)


def test_unfitted_model_refuses_to_save(tmp_path) -> None:
    with pytest.raises(ValueError):
        NgramLanguageModel().save(tmp_path / "empty.json")


# --------------------------------------------------------------------------
# aggregate
# --------------------------------------------------------------------------


def test_extract_features_on_empty_document() -> None:
    features = extract_features(segment(normalize("")))
    assert features.sentence_features == []
    assert features.document_values == {}


def test_every_sentence_receives_a_feature_vector() -> None:
    document = segment(normalize(HUMAN_STYLE_ESSAY))
    features = extract_features(document)
    assert len(features.sentence_features) == len(document.sentences)
    for sentence_features in features.sentence_features:
        assert sentence_features.values


def test_sentence_vectors_share_identical_key_sets() -> None:
    """A ragged feature space would break the model's column alignment."""
    features = extract_features(segment(normalize(HUMAN_STYLE_ESSAY)))
    key_sets = [
        frozenset(sentence.values) for sentence in features.sentence_features
    ]
    assert len(set(key_sets)) == 1


def test_sentence_vectors_exclude_document_scoped_features() -> None:
    """Scope discipline: burstiness is meaningless for a single sentence."""
    features = extract_features(segment(normalize(HUMAN_STYLE_ESSAY)))
    for sentence in features.sentence_features:
        assert "rhythm_burstiness" not in sentence.values
        assert "doc_ttr" not in sentence.values


def test_document_features_include_rhythm_and_summaries() -> None:
    features = extract_features(segment(normalize(MACHINE_STYLE_ESSAY)))
    assert "rhythm_burstiness" in features.document_values
    assert "doc_mtld" in features.document_values
    assert "docmean_lex_ttr" in features.document_values
    assert "docstd_lex_ttr" in features.document_values


def test_extraction_is_deterministic() -> None:
    """Same essay in, same evidence out. Required for defensible explanations."""
    document = segment(normalize(HUMAN_STYLE_ESSAY))
    first = extract_features(document)
    second = extract_features(document)
    assert first.document_values == second.document_values
    for left, right in zip(first.sentence_features, second.sentence_features):
        assert left.values == right.values


def test_predictability_is_none_without_a_model() -> None:
    features = extract_features(segment(normalize(HUMAN_STYLE_ESSAY)))
    for sentence in features.sentence_features:
        assert sentence.values["pred_cross_entropy"] is None


def test_predictability_populated_when_a_model_is_supplied() -> None:
    model = NgramLanguageModel().fit(
        [["i", "burned", "the", "dumplings"], ["she", "handed", "me", "another"]] * 4
    )
    features = extract_features(segment(normalize(HUMAN_STYLE_ESSAY)), language_model=model)
    assert any(
        sentence.values["pred_cross_entropy"] is not None
        for sentence in features.sentence_features
    )


def test_window_features_cover_every_position() -> None:
    features = extract_features(segment(normalize(MACHINE_STYLE_ESSAY)))
    windows = extract_window_features(features.sentence_features)
    expected = max(0, len(features.sentence_features) - PASSAGE_WINDOW_SIZE + 1)
    assert len(windows) == expected


def test_window_features_empty_when_document_too_short() -> None:
    features = extract_features(segment(normalize("One short sentence here now.")))
    assert extract_window_features(features.sentence_features) == []


def test_short_essay_does_not_crash_and_marks_rhythm_undefined() -> None:
    features = extract_features(segment(normalize("I tried. I failed. I tried again.")))
    assert features.document_values["rhythm_burstiness"] is None
    assert len(features.sentence_features) == 3
