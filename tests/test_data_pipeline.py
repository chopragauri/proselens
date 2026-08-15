"""Tests for deduplication and grouped splitting.

These guard the two failure modes that would invalidate every number the
project reports: a near-duplicate straddling the split, and a group leaking
across it.
"""

from __future__ import annotations

import pytest

from app.data.deduplication import (
    deduplicate,
    exact_duplicate_indices,
    find_duplicate_clusters,
    shingles,
)
from app.data.splitting import (
    SplitRatios,
    assign_groups_to_splits,
    compute_split_hash,
    validate_split,
)

BASE_ESSAY = (
    "The city council should reconsider the proposal to close the waterfront "
    "road to private vehicles. Residents depend on that route for groceries "
    "and for school pickup, and the alternative adds twenty minutes to a "
    "journey that already takes half an hour in traffic."
)

# Same essay with a handful of words changed: the case exact hashing misses.
NEAR_DUPLICATE_ESSAY = (
    "The town council should reconsider the proposal to close the waterfront "
    "road to private vehicles. Residents rely on that route for groceries "
    "and for school pickup, and the alternative adds twenty minutes to a "
    "journey that already takes half an hour in traffic."
)

UNRELATED_ESSAY = (
    "Venus is often called Earth's twin, but its surface pressure would crush "
    "a submarine and its clouds are made of sulfuric acid. Studying it from "
    "orbit is the only practical approach for the foreseeable future."
)


# --------------------------------------------------------------------------
# deduplication
# --------------------------------------------------------------------------


def test_shingles_of_empty_text() -> None:
    assert shingles("") == set()


def test_shingles_ignore_case_and_whitespace_formatting() -> None:
    assert shingles("The  Cat\nSat") == shingles("the cat sat")


def test_exact_duplicates_detected() -> None:
    texts = [BASE_ESSAY, UNRELATED_ESSAY, BASE_ESSAY, BASE_ESSAY.upper()]
    # Index 2 is an exact repeat; index 3 differs only in case, which the
    # canonical form also treats as identical.
    assert exact_duplicate_indices(texts) == [2, 3]


def test_near_duplicates_are_clustered_together() -> None:
    """The case that motivates MinHash: exact hashing would miss this."""
    texts = [BASE_ESSAY, NEAR_DUPLICATE_ESSAY, UNRELATED_ESSAY]
    clusters = find_duplicate_clusters(texts)
    cluster_of = {
        index: position
        for position, cluster in enumerate(clusters)
        for index in cluster
    }
    assert cluster_of[0] == cluster_of[1]
    assert cluster_of[2] != cluster_of[0]


def test_unrelated_texts_are_not_merged() -> None:
    texts = [BASE_ESSAY, UNRELATED_ESSAY]
    assert len(find_duplicate_clusters(texts)) == 2


def test_deduplicate_keeps_one_representative_per_cluster() -> None:
    texts = [BASE_ESSAY, NEAR_DUPLICATE_ESSAY, UNRELATED_ESSAY]
    report = deduplicate(texts)
    assert report.kept_indices == [0, 2]
    assert report.removed_indices == [1]
    assert report.cluster_count == 2


def test_deduplicate_on_empty_input() -> None:
    report = deduplicate([])
    assert report.kept_indices == []
    assert report.cluster_count == 0


def test_deduplication_is_deterministic_across_runs() -> None:
    """Guards the use of blake2b over Python's randomized built-in hash."""
    texts = [BASE_ESSAY, NEAR_DUPLICATE_ESSAY, UNRELATED_ESSAY, BASE_ESSAY]
    first = find_duplicate_clusters(texts)
    second = find_duplicate_clusters(texts)
    assert first == second


def test_duplicate_clusters_are_transitive() -> None:
    """A near B, B near C implies all three share a cluster."""
    middle = NEAR_DUPLICATE_ESSAY
    third = NEAR_DUPLICATE_ESSAY.replace("twenty minutes", "twenty-five minutes")
    clusters = find_duplicate_clusters([BASE_ESSAY, middle, third])
    assert len(clusters) == 1
    assert sorted(clusters[0]) == [0, 1, 2]


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------


def test_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        SplitRatios(train=0.8, validation=0.3, test=0.1)


def test_every_group_lands_in_exactly_one_split() -> None:
    sizes = {f"prompt{index}": 100 + index * 10 for index in range(15)}
    assignment = assign_groups_to_splits(sizes, SplitRatios())
    assert set(assignment) == set(sizes)
    assert set(assignment.values()) <= {"train", "validation", "test"}


def test_assignment_approximates_target_ratios() -> None:
    sizes = {f"prompt{index}": 100 for index in range(20)}
    assignment = assign_groups_to_splits(sizes, SplitRatios())

    totals = {"train": 0, "validation": 0, "test": 0}
    for group, split in assignment.items():
        totals[split] += sizes[group]
    grand_total = sum(totals.values())

    assert totals["train"] / grand_total == pytest.approx(0.70, abs=0.10)
    assert totals["validation"] / grand_total == pytest.approx(0.15, abs=0.10)
    assert totals["test"] / grand_total == pytest.approx(0.15, abs=0.10)


def test_assignment_is_deterministic() -> None:
    sizes = {"a": 300, "b": 300, "c": 200, "d": 100}
    assert assign_groups_to_splits(sizes, SplitRatios()) == assign_groups_to_splits(
        sizes, SplitRatios()
    )


def test_assignment_on_empty_input() -> None:
    assert assign_groups_to_splits({}, SplitRatios()) == {}


def test_split_hash_changes_when_assignment_changes() -> None:
    first = {"a": "train", "b": "test"}
    second = {"a": "train", "b": "validation"}
    assert compute_split_hash(first) != compute_split_hash(second)


def test_split_hash_is_order_independent() -> None:
    """The hash must identify the split, not the dict insertion order."""
    first = {"a": "train", "b": "test"}
    second = {"b": "test", "a": "train"}
    assert compute_split_hash(first) == compute_split_hash(second)


def _valid_split_fixture():
    groups = ["p1"] * 4 + ["p2"] * 4 + ["p3"] * 4
    splits = ["train"] * 4 + ["validation"] * 4 + ["test"] * 4
    labels = [0, 1, 1, 0] * 3
    generators = ["human", "gpt4", "claude", "human"] * 3
    return groups, splits, labels, generators


def test_validate_split_accepts_a_well_formed_split() -> None:
    validate_split(*_valid_split_fixture())


def test_validate_split_rejects_a_group_spanning_two_splits() -> None:
    groups, splits, labels, generators = _valid_split_fixture()
    splits[0] = "test"  # p1 now appears in both train and test
    with pytest.raises(ValueError, match="more than one split"):
        validate_split(groups, splits, labels, generators)


def test_validate_split_rejects_a_single_class_split() -> None:
    groups, splits, labels, generators = _valid_split_fixture()
    for index in range(4, 8):
        labels[index] = 1  # validation becomes all-machine
    with pytest.raises(ValueError, match="both classes"):
        validate_split(groups, splits, labels, generators)


def test_validate_split_rejects_a_single_generator_test_set() -> None:
    """A test set from one generator measures that generator, not machine text."""
    groups, splits, labels, generators = _valid_split_fixture()
    for index in range(8, 12):
        generators[index] = "gpt4" if labels[index] == 1 else "human"
    with pytest.raises(ValueError, match="generator"):
        validate_split(groups, splits, labels, generators)


def test_validate_split_rejects_mismatched_sequence_lengths() -> None:
    groups, splits, labels, generators = _valid_split_fixture()
    with pytest.raises(ValueError, match="differ in length"):
        validate_split(groups[:-1], splits, labels, generators)
