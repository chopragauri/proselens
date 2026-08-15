"""Near-duplicate detection via MinHash with locality-sensitive hashing.

Why this exists
---------------
The corpus is assembled from several sources that overlap: the same PERSUADE
essays appear in more than one upstream release, and generated essays produced
from one prompt template are frequently near-identical to each other. If a
document and its near-twin land on opposite sides of the train/test split, the
model is tested on something it has effectively memorized, and every reported
metric is inflated. Exact hashing does not catch this, because the twins differ
by a few characters.

Why MinHash rather than pairwise comparison
-------------------------------------------
The corpus holds ~45,000 essays, which is ~10^9 pairs — far too many to compare
directly. MinHash reduces each essay to a fixed-length signature whose
agreement rate estimates Jaccard similarity, and LSH banding means we only ever
compare essays that already collide in at least one band. That turns a
quadratic problem into a near-linear one.

Determinism
-----------
Hashing uses blake2b rather than Python's built-in `hash`, which is randomized
per process by PYTHONHASHSEED. A split that changed between runs would make
"reproducible pipeline" a false claim.
"""

from __future__ import annotations

import re
import zlib
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "DuplicateReport",
    "shingles",
    "minhash_signature",
    "find_duplicate_clusters",
    "deduplicate",
]

# Character 5-grams. Character shingles rather than word shingles because they
# degrade gracefully under the small edits that distinguish near-duplicates
# (a changed word, different punctuation) where word shingles would not.
SHINGLE_SIZE = 5

# Signature length. 128 permutations put the standard error of the Jaccard
# estimate near 1/sqrt(128) ~= 0.09, which is precise enough to separate
# "near-duplicate" from "same topic" at the threshold used below.
SIGNATURE_SIZE = 128

# LSH banding: 16 bands of 8 rows. Two documents become candidates if any band
# matches exactly, which makes the detection probability rise steeply around
# Jaccard ~0.75 — close to the similarity threshold we actually care about.
BAND_COUNT = 16
ROWS_PER_BAND = SIGNATURE_SIZE // BAND_COUNT

# Documents at or above this estimated Jaccard similarity are treated as the
# same document for splitting purposes.
DEFAULT_SIMILARITY_THRESHOLD = 0.75

# Guard against a pathological LSH bucket turning the pass quadratic.
MAXIMUM_BUCKET_SIZE = 400

# Mersenne prime modulus for the universal hash family. 2**31 - 1 rather than a
# 61-bit prime so that `multiplier * value + offset` stays below 2**63 and the
# whole permutation step can run vectorized in int64 without overflow. At
# corpus scale that difference is the difference between minutes and hours.
_HASH_MODULUS = (1 << 31) - 1

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class DuplicateReport:
    """Outcome of a deduplication pass."""

    kept_indices: list[int]
    removed_indices: list[int]
    cluster_count: int

    @property
    def removed_count(self) -> int:
        return len(self.removed_indices)


def _canonical(text: str) -> str:
    """Lowercase and collapse whitespace so trivial formatting is not a difference."""
    return _WHITESPACE.sub(" ", text.lower()).strip()


def shingles(text: str, size: int = SHINGLE_SIZE) -> set[str]:
    """Character n-gram shingles of `text`."""
    canonical = _canonical(text)
    if len(canonical) < size:
        return {canonical} if canonical else set()
    return {canonical[index : index + size] for index in range(len(canonical) - size + 1)}


def _stable_hash(value: str) -> int:
    """Deterministic 32-bit hash, stable across processes and machines.

    CRC32 rather than Python's built-in `hash`, which is randomized per process
    by PYTHONHASHSEED and would make the split non-reproducible. CRC32 is not a
    cryptographic hash, which is fine — nothing here is adversarial, and it is
    roughly an order of magnitude faster than blake2b at the tens of millions
    of shingles this corpus produces.
    """
    return zlib.crc32(value.encode("utf-8")) % _HASH_MODULUS


def _permutation_coefficients(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw (a, b) for the universal hash family h(x) = (a*x + b) mod prime."""
    generator = np.random.default_rng(seed)
    multipliers = generator.integers(1, _HASH_MODULUS, size=SIGNATURE_SIZE, dtype=np.int64)
    offsets = generator.integers(0, _HASH_MODULUS, size=SIGNATURE_SIZE, dtype=np.int64)
    return multipliers, offsets


def minhash_signature(
    text: str, multipliers: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    """MinHash signature of `text` under the supplied permutations."""
    document_shingles = shingles(text)
    if not document_shingles:
        return np.full(SIGNATURE_SIZE, _HASH_MODULUS, dtype=np.int64)

    hashed = np.fromiter(
        (_stable_hash(shingle) for shingle in document_shingles),
        dtype=np.int64,
        count=len(document_shingles),
    )

    # Vectorized permutation: one (SIGNATURE_SIZE, n_shingles) matrix, then the
    # minimum along each row. Every value is below 2**31, so the products stay
    # inside int64 and no Python-level loop over shingles is needed.
    permuted = (
        multipliers[:, np.newaxis] * hashed[np.newaxis, :] + offsets[:, np.newaxis]
    ) % _HASH_MODULUS
    return permuted.min(axis=1)


def _estimated_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    """Fraction of signature positions that agree."""
    return float(np.mean(left == right))


class _UnionFind:
    """Disjoint-set forest, used to merge transitively similar documents."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, item: int) -> int:
        while self._parent[item] != item:
            self._parent[item] = self._parent[self._parent[item]]
            item = self._parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self._parent[right_root] = left_root


def find_duplicate_clusters(
    texts: Sequence[str],
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    seed: int = 20260814,
) -> list[list[int]]:
    """Group indices of `texts` into clusters of mutual near-duplicates.

    Returns one list per cluster, including singleton clusters, in ascending
    order of their smallest member.
    """
    if not texts:
        return []

    multipliers, offsets = _permutation_coefficients(seed)
    signatures = [minhash_signature(text, multipliers, offsets) for text in texts]

    # LSH: bucket documents by each band of their signature. Only documents
    # sharing a bucket are ever compared.
    buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)
    for index, signature in enumerate(signatures):
        for band in range(BAND_COUNT):
            start = band * ROWS_PER_BAND
            band_key = signature[start : start + ROWS_PER_BAND].tobytes()
            buckets[(band, band_key)].append(index)

    union_find = _UnionFind(len(texts))
    compared: set[tuple[int, int]] = set()

    for members in buckets.values():
        if len(members) < 2:
            continue
        if len(members) > MAXIMUM_BUCKET_SIZE:
            # A bucket this large means the band is not discriminating (very
            # short or highly templated texts). Comparing it exhaustively is
            # quadratic and contributes little, since genuine duplicates also
            # collide in the other fifteen bands.
            continue
        for position, left in enumerate(members):
            for right in members[position + 1 :]:
                pair = (left, right)
                if pair in compared:
                    continue
                compared.add(pair)
                if _estimated_jaccard(signatures[left], signatures[right]) >= threshold:
                    union_find.union(left, right)

    clusters: dict[int, list[int]] = defaultdict(list)
    for index in range(len(texts)):
        clusters[union_find.find(index)].append(index)

    return sorted(clusters.values(), key=min)


def deduplicate(
    texts: Sequence[str],
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    seed: int = 20260814,
) -> DuplicateReport:
    """Keep one representative per near-duplicate cluster.

    The lowest index in each cluster is kept, which makes the result depend
    only on input order and not on hash iteration order.
    """
    clusters = find_duplicate_clusters(texts, threshold=threshold, seed=seed)

    kept: list[int] = []
    removed: list[int] = []
    for cluster in clusters:
        ordered = sorted(cluster)
        kept.append(ordered[0])
        removed.extend(ordered[1:])

    return DuplicateReport(
        kept_indices=sorted(kept),
        removed_indices=sorted(removed),
        cluster_count=len(clusters),
    )


def exact_duplicate_indices(texts: Iterable[str]) -> list[int]:
    """Indices of texts whose canonical form has been seen earlier.

    A cheap first pass: exact duplicates are common and removing them before
    MinHash shrinks the problem at negligible cost.
    """
    seen: set[str] = set()
    duplicates: list[int] = []
    for index, text in enumerate(texts):
        canonical = _canonical(text)
        if canonical in seen:
            duplicates.append(index)
        else:
            seen.add(canonical)
    return duplicates
