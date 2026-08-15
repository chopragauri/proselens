"""Syntactic features (Feature Group 6).

Rationale: syntax is where machine prose is most consistently regular. Models
favour a stable clause architecture — a subject, a main verb, one subordinate
clause, a tidy close — while human writers vary sentence shape far more, and
lean harder on pronouns and first-person reference in personal writing.

These features come from spaCy's tagger and parser. They are the reason the
parser stays enabled in `pipeline.py` despite its cost.
"""

from __future__ import annotations

from spacy.tokens import Doc, Span, Token

from app.nlp.features.base import FeatureDict, prefix_features, safe_divide

__all__ = ["extract_syntax_features", "SYNTAX_FEATURE_NAMES"]

# Coarse part-of-speech tags whose share of the sentence we track. Restricted
# to the classes with a plausible stylistic story, rather than every tag spaCy
# emits, to keep the feature space interpretable.
_TRACKED_POS_TAGS = ("NOUN", "VERB", "ADJ", "ADV", "PRON", "PROPN", "ADP", "CCONJ")

# Dependency labels that introduce a subordinate or coordinate clause. Used as
# a parser-based proxy for clause count, which is more reliable than splitting
# on commas and conjunctions by hand.
_CLAUSE_DEPENDENCIES = frozenset(
    {"advcl", "ccomp", "xcomp", "relcl", "acl", "csubj", "conj"}
)


def _dependency_depth(token: Token, cache: dict[int, int]) -> int:
    """Depth of `token` below the sentence root, memoized within a sentence."""
    if token.i in cache:
        return cache[token.i]
    if token.head is token or token.dep_ == "ROOT":
        cache[token.i] = 0
        return 0
    depth = _dependency_depth(token.head, cache) + 1
    cache[token.i] = depth
    return depth


def extract_syntax_features(sentence: Doc | Span) -> FeatureDict:
    """Part-of-speech distribution and clause-complexity features."""
    tokens = [token for token in sentence if not token.is_space]
    non_punct_tokens = [token for token in tokens if not token.is_punct]
    total = len(non_punct_tokens)

    features: FeatureDict = {}

    for tag in _TRACKED_POS_TAGS:
        count = sum(1 for token in non_punct_tokens if token.pos_ == tag)
        features[f"{tag.lower()}_ratio"] = safe_divide(count, total)

    # Personal writing is first-person heavy; generic machine essay prose drifts
    # toward the impersonal even when asked for a personal statement.
    first_person_count = sum(
        1
        for token in non_punct_tokens
        if token.pos_ == "PRON" and token.text.lower() in {"i", "me", "my", "mine", "myself"}
    )
    features["first_person_ratio"] = safe_divide(first_person_count, total)

    depth_cache: dict[int, int] = {}
    depths = [_dependency_depth(token, depth_cache) for token in tokens]
    features["max_dependency_depth"] = float(max(depths)) if depths else None
    features["mean_dependency_depth"] = safe_divide(sum(depths), len(depths))

    clause_count = sum(
        1 for token in tokens if token.dep_ in _CLAUSE_DEPENDENCIES
    )
    # Every sentence has at least one (main) clause, so add it to make the count
    # a true clause total rather than a subordinate-clause total.
    features["clause_count"] = float(clause_count + 1)
    features["clauses_per_token"] = safe_divide(clause_count + 1, total)

    # Mean children per token measures how bushy the parse is: flat, list-like
    # sentences differ measurably from deeply nested ones.
    features["mean_branching_factor"] = safe_divide(
        sum(len(list(token.children)) for token in tokens), len(tokens)
    )

    return prefix_features(features, "syn")


SYNTAX_FEATURE_NAMES: list[str] = (
    [f"syn_{tag.lower()}_ratio" for tag in _TRACKED_POS_TAGS]
    + [
        "syn_first_person_ratio",
        "syn_max_dependency_depth",
        "syn_mean_dependency_depth",
        "syn_clause_count",
        "syn_clauses_per_token",
        "syn_mean_branching_factor",
    ]
)
