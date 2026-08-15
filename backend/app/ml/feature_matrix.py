"""Turning FeatureDict rows into a numeric matrix a classifier can consume.

Three jobs, all of which must be *fit on training data only* and then applied
unchanged to validation and test:

1. **Column selection** — a fixed, ordered feature list, so column `k` means
   the same thing at training time and at request time. A mismatch here is the
   classic silent production bug: the model still returns a number, it is just
   the wrong one.
2. **Missing-value handling** — impute the training median and add a companion
   `__missing` indicator, so "undefined" stays distinguishable from "zero".
3. **Standardization** — zero mean, unit variance, which logistic regression
   needs for its coefficients to be comparable to one another. Comparable
   coefficients are what lets the evidence panel rank signals by contribution.

Excluded features
-----------------
Raw document size is deliberately excluded by default. In this corpus machine
essays are shorter than human ones (340 vs 397 words on average), so document
length carries real signal — but it is signal about *how this dataset was
assembled*, not about how machines write. A real admissions essay has a word
limit, and a detector that leans on length would not survive contact with one.
`scripts/train.py` reports metrics with and without these features so the cost
of the decision is measured rather than assumed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "FeatureMatrixBuilder",
    "LENGTH_ARTEFACT_FEATURES",
    "NON_FEATURE_COLUMNS",
]

# Columns carrying identity or labels rather than measurements.
NON_FEATURE_COLUMNS = frozenset(
    {
        "essay_id", "label", "split", "prompt_name", "generator", "ell_status",
        "grade_level", "sentence_index", "start", "end", "is_scorable", "text",
    }
)

# Absolute document-size features. See the module docstring.
#
# The last two are easy to miss: they arrive through the sentence-score summary
# rather than the feature extractor, but "how many sentences were scorable" and
# "how many sentences were there" are document length by another name. Left in,
# they became the single largest coefficient in the document model — so
# excluding the three obvious size columns while these two came in through the
# back door would have made the exclusion decorative.
LENGTH_ARTEFACT_FEATURES = frozenset(
    {
        "doc_token_count",
        "doc_unique_count",
        "doc_sentence_count",
        "sentscore_scorable_count",
        "total_sentence_count",
    }
)

# A feature defined for fewer than this share of training rows is dropped: its
# imputed value would be mostly invention, and its missingness indicator would
# do the real work while masquerading as a measurement.
MINIMUM_DEFINED_FRACTION = 0.30

# Below this variance a feature is effectively constant and contributes nothing
# but noise to the coefficient ranking.
MINIMUM_VARIANCE = 1e-10


@dataclass
class FeatureMatrixBuilder:
    """Fits column selection, imputation and scaling on training data."""

    exclude_length_artefacts: bool = True
    feature_names: list[str] = field(default_factory=list)
    indicator_names: list[str] = field(default_factory=list)
    medians: dict[str, float] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)
    standard_deviations: dict[str, float] = field(default_factory=dict)
    is_fitted: bool = False

    @property
    def column_names(self) -> list[str]:
        """Ordered output columns: features first, then missingness indicators."""
        return [*self.feature_names, *self.indicator_names]

    def _candidate_columns(self, frame: pd.DataFrame) -> list[str]:
        excluded = set(NON_FEATURE_COLUMNS)
        if self.exclude_length_artefacts:
            excluded |= LENGTH_ARTEFACT_FEATURES
        return sorted(
            column
            for column in frame.columns
            if column not in excluded
            and pd.api.types.is_numeric_dtype(frame[column])
        )

    def fit(self, frame: pd.DataFrame) -> FeatureMatrixBuilder:
        """Learn columns, medians and scaling from a training frame."""
        candidates = self._candidate_columns(frame)

        kept: list[str] = []
        for column in candidates:
            series = frame[column]
            if series.notna().mean() < MINIMUM_DEFINED_FRACTION:
                continue
            if series.var(skipna=True) is np.nan or (series.var(skipna=True) or 0) < MINIMUM_VARIANCE:
                continue
            kept.append(column)

        self.feature_names = kept
        # Only features that are actually ever missing get an indicator; adding
        # an all-zero column for the rest would dilute the coefficient ranking.
        self.indicator_names = [
            f"{column}__missing"
            for column in kept
            if frame[column].isna().any()
        ]

        self.medians = {
            column: float(frame[column].median(skipna=True)) for column in kept
        }

        imputed = self._impute(frame)
        self.means = {column: float(imputed[column].mean()) for column in kept}
        self.standard_deviations = {
            # A zero standard deviation would divide by zero; 1.0 leaves the
            # (constant) column untouched instead.
            column: float(imputed[column].std(ddof=0)) or 1.0
            for column in kept
        }

        self.is_fitted = True
        return self

    def _impute(self, frame: pd.DataFrame) -> pd.DataFrame:
        columns = {}
        for column in self.feature_names:
            series = (
                frame[column]
                if column in frame.columns
                else pd.Series(np.nan, index=frame.index)
            )
            columns[column] = series.fillna(self.medians[column]).astype(float)
        return pd.DataFrame(columns, index=frame.index)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Apply the fitted transformation. Column order is guaranteed stable."""
        if not self.is_fitted:
            raise ValueError("FeatureMatrixBuilder must be fitted before transform.")

        imputed = self._impute(frame)
        scaled = np.column_stack(
            [
                (imputed[column].to_numpy() - self.means[column])
                / self.standard_deviations[column]
                for column in self.feature_names
            ]
        ) if self.feature_names else np.empty((len(frame), 0))

        if not self.indicator_names:
            return scaled

        indicators = np.column_stack(
            [
                (
                    frame[name.removesuffix("__missing")].isna().to_numpy()
                    if name.removesuffix("__missing") in frame.columns
                    else np.ones(len(frame), dtype=bool)
                ).astype(float)
                for name in self.indicator_names
            ]
        )
        return np.hstack([scaled, indicators])

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    def save(self, path: Path) -> None:
        """Serialize to JSON — no pickle, so loading cannot execute code."""
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted FeatureMatrixBuilder.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "exclude_length_artefacts": self.exclude_length_artefacts,
                    "feature_names": self.feature_names,
                    "indicator_names": self.indicator_names,
                    "medians": self.medians,
                    "means": self.means,
                    "standard_deviations": self.standard_deviations,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> FeatureMatrixBuilder:
        payload = json.loads(path.read_text(encoding="utf-8"))
        builder = cls(exclude_length_artefacts=payload["exclude_length_artefacts"])
        builder.feature_names = payload["feature_names"]
        builder.indicator_names = payload["indicator_names"]
        builder.medians = payload["medians"]
        builder.means = payload["means"]
        builder.standard_deviations = payload["standard_deviations"]
        builder.is_fitted = True
        return builder
