"""Paragraph and sentence segmentation with exact character offsets.

Every span produced here carries `(start, end)` offsets into the *normalized*
essay text, which `preprocessing.normalize` guarantees is index-aligned with
the text the user actually pasted. That lets the frontend highlight a sentence
by slicing the original string, with no reconstruction and no drift.

The invariant to remember: for any Sentence `s` over text `t`,
`t[s.start:s.end] == s.text`. It is asserted in tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from spacy.language import Language

from app.nlp.pipeline import get_nlp

__all__ = ["Sentence", "Paragraph", "SegmentedDocument", "segment"]

# A paragraph break is a newline followed by any amount of horizontal
# whitespace and at least one more newline. Single newlines are treated as soft
# wrapping, not paragraph breaks, because pasted essays frequently wrap mid
# paragraph.
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n\s*")

# Sentences shorter than this carry too little signal for stable feature
# estimates: type-token ratio on a four-word sentence is essentially always
# 1.0, and variance-based features are undefined. Such sentences are still
# returned and still rendered by the UI, but are marked unscorable so the model
# never produces a confident verdict from noise.
MIN_TOKENS_FOR_SCORING = 5


@dataclass(frozen=True)
class Sentence:
    """One sentence, located precisely within the document."""

    text: str
    start: int
    end: int
    index: int
    paragraph_index: int
    token_count: int
    is_scorable: bool


@dataclass(frozen=True)
class Paragraph:
    """One paragraph and the sentences it contains."""

    text: str
    start: int
    end: int
    index: int
    sentences: list[Sentence] = field(default_factory=list)


@dataclass(frozen=True)
class SegmentedDocument:
    """A normalized essay, split into paragraphs and sentences."""

    text: str
    paragraphs: list[Paragraph]

    @property
    def sentences(self) -> list[Sentence]:
        """All sentences in reading order."""
        return [
            sentence for paragraph in self.paragraphs for sentence in paragraph.sentences
        ]

    @property
    def scorable_sentences(self) -> list[Sentence]:
        """Sentences long enough to produce stable feature values."""
        return [sentence for sentence in self.sentences if sentence.is_scorable]


def _find_paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Return `(start, end)` spans of paragraphs, with outer whitespace trimmed.

    Empty paragraphs (runs of whitespace only) are omitted.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for break_match in _PARAGRAPH_BREAK.finditer(text):
        spans.append((cursor, break_match.start()))
        cursor = break_match.end()
    spans.append((cursor, len(text)))

    trimmed: list[tuple[int, int]] = []
    for start, end in spans:
        trimmed_start, trimmed_end = _trim_span(text, start, end)
        if trimmed_end > trimmed_start:
            trimmed.append((trimmed_start, trimmed_end))
    return trimmed


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink a span past leading and trailing whitespace, keeping offsets true."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _count_content_tokens(spacy_sentence) -> int:
    """Count tokens that carry linguistic content.

    Punctuation and whitespace are excluded so that "Yes!!!" counts as one
    token rather than four, which keeps the `is_scorable` threshold meaningful.
    """
    return sum(
        1 for token in spacy_sentence if not token.is_punct and not token.is_space
    )


def segment(text: str, nlp: Language | None = None) -> SegmentedDocument:
    """Split normalized essay `text` into paragraphs and sentences.

    Args:
        text: Normalized essay text, as returned by `preprocessing.normalize`.
        nlp: Optional pipeline override, primarily for tests. Defaults to the
            shared process-wide pipeline.

    Returns:
        A `SegmentedDocument` whose spans index into `text`.
    """
    pipeline = nlp if nlp is not None else get_nlp()

    paragraph_spans = _find_paragraph_spans(text)
    if not paragraph_spans:
        return SegmentedDocument(text=text, paragraphs=[])

    # Batch the paragraphs through spaCy in one call: pipe() amortizes model
    # overhead across paragraphs instead of paying it per paragraph.
    paragraph_texts = [text[start:end] for start, end in paragraph_spans]
    docs = pipeline.pipe(paragraph_texts)

    paragraphs: list[Paragraph] = []
    sentence_index = 0

    for paragraph_index, (doc, (paragraph_start, paragraph_end)) in enumerate(
        zip(docs, paragraph_spans)
    ):
        sentences: list[Sentence] = []
        for spacy_sentence in doc.sents:
            # spaCy sentence spans include trailing whitespace; trim so the
            # highlighted region in the UI does not bleed into the next line.
            absolute_start = paragraph_start + spacy_sentence.start_char
            absolute_end = paragraph_start + spacy_sentence.end_char
            absolute_start, absolute_end = _trim_span(text, absolute_start, absolute_end)
            if absolute_end <= absolute_start:
                continue

            token_count = _count_content_tokens(spacy_sentence)
            sentences.append(
                Sentence(
                    text=text[absolute_start:absolute_end],
                    start=absolute_start,
                    end=absolute_end,
                    index=sentence_index,
                    paragraph_index=paragraph_index,
                    token_count=token_count,
                    is_scorable=token_count >= MIN_TOKENS_FOR_SCORING,
                )
            )
            sentence_index += 1

        paragraphs.append(
            Paragraph(
                text=text[paragraph_start:paragraph_end],
                start=paragraph_start,
                end=paragraph_end,
                index=paragraph_index,
                sentences=sentences,
            )
        )

    return SegmentedDocument(text=text, paragraphs=paragraphs)
