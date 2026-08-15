"""Formulaic-language features (Feature Group 8).

The brief explicitly rules out "a giant manually hard-coded phrase blacklist",
and it is right to: a list of phrases someone believes ChatGPT likes is not a
measurement, it is a hunch that cannot generalize, cannot be calibrated, and
quietly encodes the author's assumptions as if they were evidence.

What this module does instead is measure *morphosyntactic* properties that
formulaic writing exhibits regardless of the specific words chosen:

* **Discourse-marker openings** — detected by position and part of speech (an
  adverbial or prepositional phrase before the subject), not by matching a
  phrase.
* **Hedging density** — modal auxiliaries and epistemic adverbs, identified by
  spaCy's tags.
* **Nominalization** — derived abstract nouns identified by suffix morphology
  (`-tion`, `-ment`, `-ness`, ...). Heavy nominalization is the classic marker
  of generic academic register.
* **Abstract-vs-concrete balance** — proper nouns, numbers and named entities
  are evidence of specific lived detail; their absence is characteristic of
  essays generated from a prompt rather than from a life.

One small closed word list is used: the discourse connectives. That is a
closed grammatical class in English, like the stopword list spaCy already
ships, and it is fixed in advance rather than tuned on our corpus — so it is
not learning our test set's quirks. Its contribution is measured like any other
feature, and if it does not earn its place in the model it will be dropped.
"""

from __future__ import annotations

import re

from spacy.tokens import Doc, Span

from app.nlp.features.base import FeatureDict, prefix_features, safe_divide

__all__ = ["extract_formulaic_features", "FORMULAIC_FEATURE_NAMES"]

# A standard inventory of English discourse connectives (closed class, fixed in
# advance, not derived from our data). Multi-word entries are matched on the
# lowercased sentence opening.
_DISCOURSE_CONNECTIVES = frozenset(
    {
        "however", "moreover", "furthermore", "additionally", "consequently",
        "therefore", "thus", "hence", "nevertheless", "nonetheless",
        "meanwhile", "similarly", "conversely", "accordingly", "subsequently",
        "indeed", "notably", "specifically", "ultimately", "overall",
        "in conclusion", "in addition", "in summary", "for instance",
        "for example", "on the other hand", "as a result", "in contrast",
        "first of all", "in today's world", "in essence",
    }
)

# Epistemic adverbs that soften a claim. Closed class, same reasoning as above.
_HEDGING_ADVERBS = frozenset(
    {
        "perhaps", "possibly", "probably", "arguably", "seemingly",
        "apparently", "relatively", "somewhat", "fairly", "generally",
        "typically", "often", "usually", "largely", "essentially",
        "virtually", "potentially", "presumably",
    }
)

# Suffixes marking deverbal or deadjectival abstract nouns.
_NOMINALIZATION_SUFFIXES = ("tion", "sion", "ment", "ness", "ity", "ance", "ence", "ism")

# Minimum length before a suffix match is credible: "mention" ends in "tion"
# but is not a nominalization of "men".
_MINIMUM_NOMINALIZATION_LENGTH = 7

_LEADING_PHRASE = re.compile(r"^([^,]{1,40}),")


def _opens_with_discourse_marker(sentence: Doc | Span) -> bool:
    """True if the sentence begins with a recognized connective."""
    text = sentence.text.strip().lower()

    # Multi-word connectives, matched against the comma-delimited opening.
    phrase_match = _LEADING_PHRASE.match(text)
    if phrase_match and phrase_match.group(1).strip() in _DISCOURSE_CONNECTIVES:
        return True

    first_token = next(
        (token for token in sentence if not token.is_punct and not token.is_space),
        None,
    )
    if first_token is None:
        return False
    return first_token.text.lower() in _DISCOURSE_CONNECTIVES


def _has_fronted_adverbial(sentence: Doc | Span) -> bool:
    """True if an adverbial or prepositional phrase precedes the main subject.

    "Throughout my academic journey, I have..." fronts an adverbial; "I have..."
    does not. Detected structurally via part of speech and a comma, so it fires
    on constructions the module has never seen.
    """
    tokens = [token for token in sentence if not token.is_space]
    if not tokens:
        return False
    if tokens[0].pos_ not in {"ADP", "ADV", "SCONJ"}:
        return False
    # Require a comma within the first stretch of the sentence, which is what
    # marks the adverbial as fronted rather than simply sentence-initial.
    return any(token.text == "," for token in tokens[:8])


def extract_formulaic_features(sentence: Doc | Span) -> FeatureDict:
    """Register and formulaicity features for one sentence."""
    tokens = [token for token in sentence if not token.is_space]
    content_tokens = [token for token in tokens if not token.is_punct]
    total = len(content_tokens)

    features: FeatureDict = {
        "opens_with_connective": float(_opens_with_discourse_marker(sentence)),
        "has_fronted_adverbial": float(_has_fronted_adverbial(sentence)),
    }

    connective_count = sum(
        1 for token in content_tokens if token.text.lower() in _DISCOURSE_CONNECTIVES
    )
    features["connective_density"] = safe_divide(connective_count, total)

    # Hedging: modal auxiliaries plus epistemic adverbs.
    modal_count = sum(1 for token in content_tokens if token.tag_ == "MD")
    hedge_adverb_count = sum(
        1 for token in content_tokens if token.text.lower() in _HEDGING_ADVERBS
    )
    features["hedge_density"] = safe_divide(modal_count + hedge_adverb_count, total)

    nominalization_count = sum(
        1
        for token in content_tokens
        if token.pos_ == "NOUN"
        and len(token.text) >= _MINIMUM_NOMINALIZATION_LENGTH
        and token.text.lower().endswith(_NOMINALIZATION_SUFFIXES)
    )
    features["nominalization_density"] = safe_divide(nominalization_count, total)

    # Concrete specificity: proper nouns and numbers stand in for the names,
    # dates and places that anchor genuine personal narrative.
    specific_count = sum(
        1 for token in content_tokens if token.pos_ in {"PROPN", "NUM"}
    )
    features["specificity_density"] = safe_divide(specific_count, total)

    return prefix_features(features, "form")


FORMULAIC_FEATURE_NAMES: list[str] = [
    "form_opens_with_connective",
    "form_has_fronted_adverbial",
    "form_connective_density",
    "form_hedge_density",
    "form_nominalization_density",
    "form_specificity_density",
]
