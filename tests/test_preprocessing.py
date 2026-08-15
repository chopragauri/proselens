"""Tests for normalization, centred on the length-preservation invariant."""

from __future__ import annotations

import pytest

from app.nlp.preprocessing import is_length_preserving, normalize


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "\n\n\n",
        "A simple sentence.",
        "Curly “quotes” and an apostrophe’s tail.",
        "Em—dash and en–dash and minus−sign.",
        "Zero​width‌and﻿invisible.",
        "Windows\r\nline\r\nendings.",
        "Classic\rMac\rendings.",
        "Tabs\tand\tnewlines\nkept.",
        "Non breaking spaces.",
        "Mixed “quotes”, —dashes, \r\n and ​ zero width.",
    ],
)
def test_normalize_preserves_length(raw: str) -> None:
    """The core invariant: offsets into the result are valid on the input."""
    normalized = normalize(raw)
    assert is_length_preserving(raw, normalized), (
        f"length changed: {len(raw)} -> {len(normalized)}"
    )


def test_normalize_is_idempotent() -> None:
    """Normalizing twice must equal normalizing once."""
    raw = "She said “hello”—then left.\r\nNext​line."
    once = normalize(raw)
    assert normalize(once) == once


def test_curly_quotes_become_straight() -> None:
    assert normalize("“Alpha” and ‘beta’") == '"Alpha" and \'beta\''


def test_dashes_collapse_to_hyphen() -> None:
    assert normalize("a—b–c−d") == "a-b-c-d"


def test_crlf_does_not_create_a_blank_line() -> None:
    """Regression: mapping CR to newline would double every Windows line break.

    A blank line is a paragraph boundary downstream, so this bug would have
    silently doubled the paragraph count of essays pasted from Windows.
    """
    normalized = normalize("First line.\r\nSecond line.")
    assert "\n\n" not in normalized
    assert normalized == "First line. \nSecond line."


def test_lone_carriage_return_becomes_newline() -> None:
    assert normalize("First.\rSecond.") == "First.\nSecond."


def test_control_characters_are_replaced_but_tabs_and_newlines_survive() -> None:
    normalized = normalize("bad\x00char\x07here\tand\nnewline")
    assert "\x00" not in normalized
    assert "\x07" not in normalized
    assert "\t" in normalized
    assert "\n" in normalized


def test_offsets_remain_valid_after_normalization() -> None:
    """Demonstrates why the invariant exists: slicing the raw text still works."""
    raw = "The “first” sentence. The second—one follows."
    normalized = normalize(raw)
    start = normalized.index("The second")
    end = normalized.index("follows.") + len("follows.")
    # The slice taken from the *original* covers the same words.
    assert raw[start:end].startswith("The second")
    assert raw[start:end].endswith("follows.")
