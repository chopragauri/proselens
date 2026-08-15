"""Grouped train/validation/test splitting, frozen and hashed.

The split decision
------------------
Essays are grouped by **prompt**, and whole prompts are assigned to a single
split. No prompt appears in more than one split.

Decision
    Prompt-disjoint grouped splitting.

Why
    A random row-level split lets the model see "Car-free cities" essays in
    both training and test. It can then reach a good test score by learning
    topic vocabulary — which words appear in essays about traffic — rather than
    by learning anything about authorship. That model would collapse on a real
    admissions essay, and its test accuracy would have given no warning. A
    prompt-disjoint split forces every test essay to be about something the
    model has never been trained on, which is the condition the deployed system
    actually faces.

Alternative
    Stratified random splitting, which is what most published AI-detector
    numbers use, and which would produce visibly higher accuracy here.

Trade-off
    We will report lower numbers than a random split would give. Those lower
    numbers are the honest ones. With only 15 prompts the split is also coarse,
    so a single unusual prompt can move test metrics more than it would under
    random splitting.

Verification
    `validate_split` asserts prompt-disjointness and that both classes and
    several distinct generators survive in every split. The assignment is
    hashed, and the hash is recorded in the manifest before any model is fit.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "SplitRatios",
    "SplitManifest",
    "assign_groups_to_splits",
    "compute_split_hash",
    "validate_split",
    "write_manifest",
]

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitRatios:
    """Target proportion of essays in each split."""

    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15

    def __post_init__(self) -> None:
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}.")

    def as_mapping(self) -> dict[str, float]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


@dataclass
class SplitManifest:
    """Everything needed to reproduce and audit a split.

    Written to disk *before* any model is trained, so that reported metrics
    cannot be quietly attributed to a different split than the one used.
    """

    created_at: str
    split_hash: str
    group_column: str
    ratios: dict[str, float]
    seed: int
    similarity_threshold: float
    total_essays: int
    exact_duplicates_removed: int
    near_duplicates_removed: int
    split_counts: dict[str, int]
    label_counts: dict[str, dict[str, int]]
    generator_counts: dict[str, dict[str, int]]
    group_assignment: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def assign_groups_to_splits(
    group_sizes: Mapping[str, int], ratios: SplitRatios
) -> dict[str, str]:
    """Assign whole groups to splits, approximating the target proportions.

    Greedy largest-first: the biggest group goes to whichever split is
    furthest below its target share. With few large groups this lands closer to
    the requested ratios than random assignment, and being deterministic it
    needs no seed — the same group sizes always produce the same split.
    """
    if not group_sizes:
        return {}

    total = sum(group_sizes.values())
    targets = {name: getattr(ratios, name) * total for name in SPLIT_NAMES}
    current = {name: 0.0 for name in SPLIT_NAMES}
    assignment: dict[str, str] = {}

    # Ties broken by group name so the result never depends on dict ordering.
    ordered_groups = sorted(group_sizes.items(), key=lambda item: (-item[1], item[0]))

    for group, size in ordered_groups:
        chosen = max(SPLIT_NAMES, key=lambda name: targets[name] - current[name])
        assignment[group] = chosen
        current[chosen] += size

    return assignment


def compute_split_hash(assignment: Mapping[str, str]) -> str:
    """SHA-256 over the group-to-split assignment.

    Any change to which group sits in which split changes this hash, so a
    result reported against one hash cannot silently be a result from another.
    """
    payload = "\n".join(f"{group}\t{assignment[group]}" for group in sorted(assignment))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_split(
    groups: Sequence[str],
    splits: Sequence[str],
    labels: Sequence[int],
    generators: Sequence[str],
    minimum_generators_per_split: int = 2,
) -> None:
    """Raise if the split violates a property we depend on.

    Checked here rather than trusted, because each of these failures would
    produce plausible-looking metrics that mean nothing.
    """
    if not (len(groups) == len(splits) == len(labels) == len(generators)):
        raise ValueError("Parallel sequences passed to validate_split differ in length.")

    group_to_splits: dict[str, set[str]] = {}
    for group, split in zip(groups, splits):
        group_to_splits.setdefault(group, set()).add(split)

    straddling = {
        group: sorted(split_set)
        for group, split_set in group_to_splits.items()
        if len(split_set) > 1
    }
    if straddling:
        raise ValueError(f"Groups appear in more than one split: {straddling}")

    for split_name in SPLIT_NAMES:
        indices = [index for index, split in enumerate(splits) if split == split_name]
        if not indices:
            raise ValueError(f"Split '{split_name}' is empty.")

        split_labels = {labels[index] for index in indices}
        if split_labels != {0, 1}:
            raise ValueError(
                f"Split '{split_name}' must contain both classes, found {split_labels}."
            )

        # A test set drawn from one generator measures detection of that
        # generator, not of machine text.
        machine_generators = {
            generators[index] for index in indices if labels[index] == 1
        }
        if len(machine_generators) < minimum_generators_per_split:
            raise ValueError(
                f"Split '{split_name}' has only {len(machine_generators)} machine "
                f"generator(s); at least {minimum_generators_per_split} required."
            )


def write_manifest(manifest: SplitManifest, path: Path) -> None:
    """Persist the manifest, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.to_json(), encoding="utf-8")


def build_manifest(
    group_column: str,
    assignment: Mapping[str, str],
    ratios: SplitRatios,
    seed: int,
    similarity_threshold: float,
    total_essays: int,
    exact_duplicates_removed: int,
    near_duplicates_removed: int,
    split_counts: Mapping[str, int],
    label_counts: Mapping[str, Mapping[str, int]],
    generator_counts: Mapping[str, Mapping[str, int]],
) -> SplitManifest:
    """Assemble a manifest with a timestamp and the split hash."""
    return SplitManifest(
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        split_hash=compute_split_hash(assignment),
        group_column=group_column,
        ratios=ratios.as_mapping(),
        seed=seed,
        similarity_threshold=similarity_threshold,
        total_essays=total_essays,
        exact_duplicates_removed=exact_duplicates_removed,
        near_duplicates_removed=near_duplicates_removed,
        split_counts=dict(split_counts),
        label_counts={key: dict(value) for key, value in label_counts.items()},
        generator_counts={key: dict(value) for key, value in generator_counts.items()},
        group_assignment=dict(assignment),
    )
