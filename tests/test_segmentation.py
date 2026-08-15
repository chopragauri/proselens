"""Tests for paragraph and sentence segmentation, centred on offset accuracy."""

from __future__ import annotations

import pytest

from app.nlp.preprocessing import normalize
from app.nlp.segmentation import MIN_TOKENS_FOR_SCORING, segment

MULTI_PARAGRAPH_ESSAY = (
    "When I was twelve, my grandmother taught me to fold dumplings.\n"
    "Her hands moved faster than I could follow.\n"
    "\n"
    "I burned the first batch. She laughed, and we started again.\n"
    "\n"
    "That kitchen is where I learned patience."
)


def _assert_offsets_are_exact(document) -> None:
    """Every span must slice back to its own text."""
    for paragraph in document.paragraphs:
        assert document.text[paragraph.start : paragraph.end] == paragraph.text
        for sentence in paragraph.sentences:
            assert document.text[sentence.start : sentence.end] == sentence.text


@pytest.mark.parametrize("raw", ["", "   ", "\n\n\n", "\t \n \t"])
def test_empty_and_whitespace_input_yields_no_paragraphs(raw: str) -> None:
    document = segment(normalize(raw))
    assert document.paragraphs == []
    assert document.sentences == []


def test_single_sentence() -> None:
    document = segment(normalize("This is one sentence."))
    assert len(document.sentences) == 1
    assert document.sentences[0].text == "This is one sentence."
    _assert_offsets_are_exact(document)


def test_paragraphs_split_on_blank_lines_not_soft_wraps() -> None:
    document = segment(normalize(MULTI_PARAGRAPH_ESSAY))
    # Three blank-line-separated blocks; the single newlines inside the first
    # block must not create extra paragraphs.
    assert len(document.paragraphs) == 3
    _assert_offsets_are_exact(document)


def test_sentence_offsets_are_exact_across_paragraphs() -> None:
    document = segment(normalize(MULTI_PARAGRAPH_ESSAY))
    _assert_offsets_are_exact(document)


def test_sentence_indices_are_sequential_across_the_document() -> None:
    document = segment(normalize(MULTI_PARAGRAPH_ESSAY))
    indices = [sentence.index for sentence in document.sentences]
    assert indices == list(range(len(indices)))


def test_paragraph_index_is_recorded_on_each_sentence() -> None:
    document = segment(normalize(MULTI_PARAGRAPH_ESSAY))
    for paragraph in document.paragraphs:
        for sentence in paragraph.sentences:
            assert sentence.paragraph_index == paragraph.index


def test_sentences_carry_no_leading_or_trailing_whitespace() -> None:
    document = segment(normalize(MULTI_PARAGRAPH_ESSAY))
    for sentence in document.sentences:
        assert sentence.text == sentence.text.strip()
        assert sentence.text != ""


def test_short_sentences_are_returned_but_marked_unscorable() -> None:
    """The UI must still render them; the model must not score them."""
    document = segment(normalize("Yes. I had never seen anything quite like it before."))
    short, long_sentence = document.sentences[0], document.sentences[1]
    assert short.text == "Yes."
    assert short.token_count < MIN_TOKENS_FOR_SCORING
    assert short.is_scorable is False
    assert long_sentence.is_scorable is True
    assert document.scorable_sentences == [long_sentence]


def test_punctuation_does_not_inflate_token_count() -> None:
    """"Wow!!!" is one content token, not four."""
    document = segment(normalize("Wow!!!"))
    assert document.sentences[0].token_count == 1
    assert document.sentences[0].is_scorable is False


def test_abbreviations_do_not_split_sentences() -> None:
    document = segment(normalize("I met Dr. Chen in St. Louis. He changed my mind."))
    assert len(document.sentences) == 2
    _assert_offsets_are_exact(document)


def test_quoted_dialogue_keeps_offsets_exact() -> None:
    raw = 'She said, “I will not go.” Then she left the room.'
    document = segment(normalize(raw))
    _assert_offsets_are_exact(document)
    # Offsets computed on normalized text still slice correctly on the original.
    for sentence in document.sentences:
        assert len(raw[sentence.start : sentence.end]) == len(sentence.text)


def test_trailing_whitespace_and_newlines_are_ignored() -> None:
    document = segment(normalize("A complete thought here.   \n\n\n   "))
    assert len(document.paragraphs) == 1
    assert len(document.sentences) == 1
    _assert_offsets_are_exact(document)


def test_windows_line_endings_do_not_split_paragraphs() -> None:
    """End-to-end regression for the CRLF bug caught in preprocessing."""
    document = segment(normalize("First line here.\r\nSecond line here.\r\nThird one."))
    assert len(document.paragraphs) == 1
