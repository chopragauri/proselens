"""Turning model arithmetic into explanations a person can check.

The rule this module exists to enforce: **every sentence shown to the user is
derived from a measured number**. No language model writes these explanations,
and no template asserts anything the feature values do not support.

How it works
------------
The classifier is linear on standardized inputs, so for each feature
`coefficient x standardized_value` is exactly the log-odds that feature
contributed. Those contributions are grouped into the eight signal families the
brief describes, summed within each family, and ranked. The explanation names
the families that actually moved the score, in the direction they moved it.

Each signal also carries a z-score against the **human reference
distribution** — the mean and standard deviation of that feature over training
human sentences. That is what lets the panel say "comma density is 2.1
standard deviations above the human average" rather than the uninterpretable
"comma density is 0.08".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "SIGNAL_FAMILIES",
    "ReferenceStatistics",
    "SignalSummary",
    "summarize_signals",
    "build_explanation",
]

# Feature-name prefix -> the signal family shown in the UI. Ordered longest
# prefix first so that "structctx_" is matched before "struct_".
SIGNAL_FAMILIES: list[tuple[str, str]] = [
    ("structctx_", "Sentence length"),
    ("struct_", "Sentence length"),
    ("repctx_", "Repetition"),
    ("rep_", "Repetition"),
    ("lex_", "Lexical diversity"),
    ("syn_", "Syntactic structure"),
    ("punct_", "Punctuation"),
    ("form_", "Formulaic phrasing"),
    ("pred_", "Predictability"),
    ("rhythm_", "Sentence rhythm"),
    ("sentscore_", "Sentence-level evidence"),
    # Whole-document lexical measures, which carry no group prefix of their own.
    ("ttr", "Lexical diversity"),
    ("mtld", "Lexical diversity"),
    ("unique_count", "Lexical diversity"),
    ("token_count", "Length"),
    ("sentence_count", "Length"),
]

# Aggregation prefixes attached to document-level features. They describe *how*
# a sentence-level measurement was summarized, not *what* was measured, so they
# are stripped before deciding which family a feature belongs to.
#
# Without this, every `docmean_*` and `docstd_*` feature matched a catch-all
# "doc" entry and collapsed into a single bucket — punctuation, lexis, syntax
# and formulaic phrasing all reported as one undifferentiated "document-wide
# style" signal, which is precisely the specificity the evidence panel exists
# to provide.
_AGGREGATION_PREFIXES = ("docmean_", "docstd_", "doc_")

# Below this absolute log-odds contribution a family is not worth naming: it
# did not meaningfully move the score, and listing it would pad the panel with
# noise that looks like reasoning.
MINIMUM_REPORTABLE_CONTRIBUTION = 0.05

# z-score magnitudes at which we are willing to use a qualitative word.
_NOTABLE_Z_SCORE = 1.0
_STRONG_Z_SCORE = 2.0


@dataclass(frozen=True)
class ReferenceStatistics:
    """Per-feature mean and standard deviation over training human sentences.

    Deliberately computed on humans only. A z-score against the *combined*
    distribution would answer "how unusual is this among all essays", which is
    not the question the panel is asking. The question is "how far is this from
    how people write".
    """

    means: dict[str, float]
    standard_deviations: dict[str, float]

    def z_score(self, feature: str, value: float | None) -> float | None:
        if value is None or feature not in self.means:
            return None
        deviation = self.standard_deviations.get(feature, 0.0)
        if not deviation:
            return None
        return (value - self.means[feature]) / deviation

    @classmethod
    def load(cls, path: Path) -> ReferenceStatistics:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            means=payload["means"],
            standard_deviations=payload["standard_deviations"],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"means": self.means, "standard_deviations": self.standard_deviations},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )


@dataclass(frozen=True)
class SignalSummary:
    """One signal family's contribution to a score."""

    family: str
    contribution: float
    direction: str
    strongest_feature: str
    strongest_value: float | None
    strongest_z_score: float | None
    descriptor: str


def family_for(feature_name: str) -> str:
    """Map a feature to the signal family shown in the evidence panel.

    Aggregation prefixes are stripped first so that `docmean_punct_comma_density`
    is recognized as punctuation rather than as a generic document feature.
    """
    base = feature_name.removesuffix("__missing")
    for prefix in _AGGREGATION_PREFIXES:
        if base.startswith(prefix):
            base = base[len(prefix) :]
            break

    for prefix, family in SIGNAL_FAMILIES:
        if base.startswith(prefix):
            return family
    return "Other"


def _descriptor(family: str, z_score: float | None, contribution: float) -> str:
    """A short qualitative reading, grounded in the z-score where one exists.

    When there is no z-score — a missingness indicator, or a feature absent
    from the reference set — we say only which way it pushed, rather than
    inventing a magnitude.
    """
    if z_score is None:
        return "above typical" if contribution > 0 else "below typical"

    magnitude = abs(z_score)
    if magnitude >= _STRONG_Z_SCORE:
        strength = "far"
    elif magnitude >= _NOTABLE_Z_SCORE:
        strength = "noticeably"
    else:
        return "close to the human average"

    return f"{strength} {'above' if z_score > 0 else 'below'} the human average"


def summarize_signals(
    contributions: dict[str, float],
    feature_values: dict[str, float | None],
    reference: ReferenceStatistics,
    limit: int = 6,
) -> list[SignalSummary]:
    """Group per-feature contributions into ranked signal families."""
    by_family: dict[str, list[tuple[str, float]]] = {}
    for feature, contribution in contributions.items():
        by_family.setdefault(family_for(feature), []).append((feature, contribution))

    summaries: list[SignalSummary] = []
    for family, entries in by_family.items():
        total = sum(contribution for _, contribution in entries)
        if abs(total) < MINIMUM_REPORTABLE_CONTRIBUTION:
            continue

        strongest_feature, _ = max(entries, key=lambda entry: abs(entry[1]))
        base_name = strongest_feature.removesuffix("__missing")
        value = feature_values.get(base_name)
        z_score = reference.z_score(base_name, value)

        summaries.append(
            SignalSummary(
                family=family,
                contribution=total,
                direction="machine" if total > 0 else "human",
                strongest_feature=base_name,
                strongest_value=value,
                strongest_z_score=z_score,
                descriptor=_descriptor(family, z_score, total),
            )
        )

    summaries.sort(key=lambda summary: abs(summary.contribution), reverse=True)
    return summaries[:limit]


def build_explanation(summaries: list[SignalSummary], risk: float) -> str:
    """Compose a sentence describing why a score came out where it did.

    Assembled entirely from the ranked families and their measured z-scores.
    If nothing moved the score appreciably, it says so instead of manufacturing
    a reason.
    """
    if not summaries:
        return (
            "No individual signal moved this score appreciably; the assessment "
            "rests on the combination of many small effects rather than any one "
            "measurable property."
        )

    verdict_direction = "machine" if risk >= 0.5 else "human"
    verdict = "machine-like" if risk >= 0.5 else "consistent with human writing"

    # Families are split by which way they pushed, and the explanation is built
    # from those two lists rather than from the overall ranking.
    #
    # Ranking by magnitude alone produced self-contradictory text: when the
    # single largest family opposed the verdict, it was named as what "drove"
    # the score and then named again as "pushing the other way" in the same
    # breath. What drove a verdict can only be a family that pointed toward it.
    supporting_families = [s for s in summaries if s.direction == verdict_direction]
    opposing_families = [s for s in summaries if s.direction != verdict_direction]

    parts: list[str] = []

    if supporting_families:
        leading = supporting_families[0]
        parts.append(
            f"Scored {verdict}, driven mainly by {leading.family.lower()}, "
            f"which is {leading.descriptor}."
        )
        supporting = supporting_families[1:3]
        if supporting:
            described = ", ".join(
                f"{s.family.lower()} ({s.descriptor})" for s in supporting
            )
            parts.append(f"Also contributing: {described}.")
    else:
        # Every named family opposed the verdict, which happens when the score
        # rests on many small effects none of which cleared the reporting
        # threshold. Saying so is more honest than promoting a contrary signal.
        parts.append(
            f"Scored {verdict}, though no single signal family drove it; the "
            "score rests on the combination of many small effects."
        )

    # Naming the counter-evidence keeps the panel honest: a one-sided
    # explanation reads as more certain than the model actually is.
    if opposing_families:
        strongest_opposing = opposing_families[0]
        parts.append(
            f"Pushing the other way: {strongest_opposing.family.lower()}, "
            f"{strongest_opposing.descriptor}."
        )

    return " ".join(parts)
