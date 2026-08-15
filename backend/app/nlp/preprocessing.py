"""Text normalization for ProseLens.

Design decision: every transformation here is **length-preserving**, i.e. the
output string has exactly the same number of characters as the input, and the
character at index `i` of the output corresponds to the character at index `i`
of the input.

Why: the frontend highlights sentences by character offset. If normalization
could delete or insert characters, every offset we compute downstream would
need a mapping table back to the original text, and any bug in that table would
silently mis-highlight the essay the user pasted. Forbidding length changes
removes that entire class of bug, and `assert_length_preserved` makes the
invariant testable rather than aspirational.

Trade-off: we cannot collapse runs of whitespace or expand ligatures here.
Whitespace is instead handled at segmentation time by trimming sentence spans,
which keeps offsets exact. Ligatures ("ﬁ") are left alone; they are rare in
pasted essays and spaCy tokenizes them acceptably.
"""

from __future__ import annotations

import unicodedata

__all__ = ["normalize", "is_length_preserving", "strip_control_characters"]


# Curly quotes, dashes, and spaces that vary by word processor. Pasting from
# Google Docs or Word is the common case for this app, so these matter: an
# unnormalized U+2019 apostrophe changes tokenization and therefore features.
# Every entry maps exactly one character to exactly one character.
_CHARACTER_SUBSTITUTIONS: dict[str, str] = {
    # Single quotes / apostrophes
    "‘": "'",  # left single quotation mark
    "’": "'",  # right single quotation mark (Word's apostrophe)
    "‚": "'",  # single low-9 quotation mark
    "‛": "'",  # single high-reversed-9 quotation mark
    "′": "'",  # prime
    "´": "'",  # acute accent used as apostrophe
    "`": "'",  # grave accent used as apostrophe
    # Double quotes
    "“": '"',  # left double quotation mark
    "”": '"',  # right double quotation mark
    "„": '"',  # double low-9 quotation mark
    "‟": '"',  # double high-reversed-9 quotation mark
    "″": '"',  # double prime
    # Dashes and minus signs
    "‐": "-",  # hyphen
    "‑": "-",  # non-breaking hyphen
    "‒": "-",  # figure dash
    "–": "-",  # en dash
    "—": "-",  # em dash
    "―": "-",  # horizontal bar
    "−": "-",  # minus sign
    # Spaces. Mapped to a plain space so whitespace logic downstream is simple.
    " ": " ",  # no-break space
    " ": " ",  # ogham space mark
    " ": " ",  # en quad
    " ": " ",  # em quad
    " ": " ",  # en space
    " ": " ",  # em space
    " ": " ",  # three-per-em space
    " ": " ",  # four-per-em space
    " ": " ",  # six-per-em space
    " ": " ",  # figure space
    " ": " ",  # punctuation space
    " ": " ",  # thin space
    " ": " ",  # hair space
    " ": " ",  # narrow no-break space
    " ": " ",  # medium mathematical space
    "　": " ",  # ideographic space
    # Line separators normalized to newline so paragraph splitting sees them.
    " ": "\n",  # line separator
    " ": "\n",  # paragraph separator
    "\r": "\n",  # lone CR (CRLF becomes "\n\n"; blank-line splitting tolerates it)
    # Zero-width and invisible characters. These appear in text copied from web
    # pages and, notably, in text pasted out of some chat interfaces. They are
    # replaced with a space rather than removed, to preserve length.
    "​": " ",  # zero width space
    "‌": " ",  # zero width non-joiner
    "‍": " ",  # zero width joiner
    "﻿": " ",  # zero width no-break space / BOM
    "­": " ",  # soft hyphen
}

_TRANSLATION_TABLE = str.maketrans(_CHARACTER_SUBSTITUTIONS)

# Control characters that break spaCy or render as garbage, excluding newline
# and tab which carry structure we want.
_PRESERVED_CONTROL_CHARACTERS = {"\n", "\t"}


def strip_control_characters(text: str) -> str:
    """Replace Unicode control characters with spaces, preserving length.

    Newlines and tabs are kept because paragraph segmentation depends on them.
    """
    return "".join(
        character
        if character in _PRESERVED_CONTROL_CHARACTERS
        or unicodedata.category(character) != "Cc"
        else " "
        for character in text
    )


def normalize(raw_text: str) -> str:
    """Normalize an essay for analysis without changing its length.

    The returned string is index-aligned with `raw_text`, so any character
    offset computed against the result is also valid against the original.
    """
    # CRLF must be handled before the character table maps a bare CR to "\n".
    # Otherwise every Windows line ending becomes "\n\n", which paragraph
    # splitting would read as a paragraph break, silently doubling the
    # paragraph count of any essay pasted from a Windows editor.
    without_crlf = raw_text.replace("\r\n", " \n")
    substituted = without_crlf.translate(_TRANSLATION_TABLE)
    return strip_control_characters(substituted)


def is_length_preserving(raw_text: str, normalized_text: str) -> bool:
    """Check the core invariant. Used by tests and by defensive assertions."""
    return len(raw_text) == len(normalized_text)
