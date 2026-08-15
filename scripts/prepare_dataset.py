"""Build the ProseLens working corpus with a frozen, hashed train/val/test split.

Pipeline order, and why it is this order:

1. **Load** DAIGT v2 (human + machine essays across 17 generators).
2. **Join ELL status** from PERSUADE 2.0 onto the human essays. This is what
   makes the fairness analysis in `docs/limitations.md` a measurement rather
   than a disclaimer, so it happens before any filtering that could drop the
   ELL essays disproportionately.
3. **Filter** to a sane length band.
4. **Subsample** to a working corpus, stratified by prompt and generator.
5. **Deduplicate** — after subsampling, because what matters is that no near
   duplicate straddles the split *within the corpus we actually use*, and
   deduplicating 45,000 essays to then discard four fifths of them would cost
   far more time than it saves.
6. **Split** by prompt, disjointly.
7. **Validate and freeze** — the manifest, including the split hash, is written
   before any model exists.

Usage:
    python3 scripts/prepare_dataset.py --per-class 4000
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "backend"))

from app.data.deduplication import (  # noqa: E402
    DEFAULT_SIMILARITY_THRESHOLD,
    deduplicate,
    exact_duplicate_indices,
)
from app.data.splitting import (  # noqa: E402
    SplitRatios,
    assign_groups_to_splits,
    build_manifest,
    validate_split,
    write_manifest,
)

DAIGT_PATH = REPOSITORY_ROOT / "data/raw/daigt_v2/train_v2_drcat_02.csv"
PERSUADE_PATH = (
    REPOSITORY_ROOT / "data/raw/persuade2/persuade_2.0_human_scores_demo_id_github.csv"
)
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "data/processed"

# Essays outside this band are dropped: below ~50 words there is too little
# text for variance-based features to mean anything, and the handful of
# very long documents are transcripts and scraped artefacts rather than essays.
MINIMUM_WORDS = 50
MAXIMUM_WORDS = 1000

# Human sources in DAIGT. Everything else is a generator name.
HUMAN_SOURCES = frozenset({"persuade_corpus", "train_essays"})

RANDOM_SEED = 20260814

_WHITESPACE = re.compile(r"\s+")


def canonical_text(text: str) -> str:
    return _WHITESPACE.sub(" ", str(text)).strip()


def essay_identifier(text: str) -> str:
    """Stable id derived from content, so ids survive a rebuild."""
    return hashlib.sha256(canonical_text(text).encode("utf-8")).hexdigest()[:16]


def load_corpus() -> pd.DataFrame:
    """Load DAIGT and attach ELL status to the human essays."""
    frame = pd.read_csv(DAIGT_PATH)
    frame = frame.rename(columns={"source": "generator"})
    frame["canonical"] = frame["text"].map(canonical_text)

    persuade = pd.read_csv(PERSUADE_PATH, low_memory=False)
    persuade["canonical"] = persuade["full_text"].map(canonical_text)
    ell_by_text = (
        persuade.drop_duplicates("canonical")
        .set_index("canonical")["ell_status"]
        .to_dict()
    )
    grade_by_text = (
        persuade.drop_duplicates("canonical")
        .set_index("canonical")["grade_level"]
        .to_dict()
    )

    frame["ell_status"] = frame["canonical"].map(ell_by_text)
    frame["grade_level"] = frame["canonical"].map(grade_by_text)

    # PERSUADE encodes unknown ELL status as an empty string in some rows and
    # as a missing value in others. Left as-is, the empty string becomes its
    # own category and silently pollutes the fairness comparison with essays
    # whose status nobody recorded.
    frame["ell_status"] = (
        frame["ell_status"].astype("string").str.strip().replace({"": pd.NA})
    )
    return frame


def filter_by_length(frame: pd.DataFrame) -> pd.DataFrame:
    word_counts = frame["text"].str.split().str.len()
    return frame[word_counts.between(MINIMUM_WORDS, MAXIMUM_WORDS)].copy()


def stratified_subsample(
    frame: pd.DataFrame, per_class: int, seed: int
) -> pd.DataFrame:
    """Sample `per_class` essays per label, spread over prompts and generators.

    Machine essays are sampled per (prompt, generator) cell so that no single
    generator dominates, which would turn the task into detecting that one
    model. Human essays are sampled per prompt.
    """
    selected: list[pd.DataFrame] = []

    for label in (0, 1):
        class_frame = frame[frame["label"] == label]
        cell_columns = ["prompt_name"] if label == 0 else ["prompt_name", "generator"]
        cells = list(class_frame.groupby(cell_columns, dropna=False))
        if not cells:
            continue

        per_cell = max(1, per_class // len(cells))
        chunks = [
            group.sample(min(per_cell, len(group)), random_state=seed)
            for _, group in cells
        ]
        combined = pd.concat(chunks)

        # Top up from whatever remains if even allocation undershot the target.
        if len(combined) < per_class:
            remainder = class_frame.drop(index=combined.index)
            shortfall = min(per_class - len(combined), len(remainder))
            if shortfall > 0:
                combined = pd.concat(
                    [combined, remainder.sample(shortfall, random_state=seed)]
                )

        selected.append(combined.sample(frac=1.0, random_state=seed).head(per_class))

    return pd.concat(selected).reset_index(drop=True)


def remove_duplicates(
    frame: pd.DataFrame, threshold: float, seed: int
) -> tuple[pd.DataFrame, int, int]:
    """Drop exact then near duplicates. Returns the frame and both counts."""
    exact = exact_duplicate_indices(frame["canonical"].tolist())
    if exact:
        frame = frame.drop(index=frame.index[exact]).reset_index(drop=True)

    report = deduplicate(frame["text"].tolist(), threshold=threshold, seed=seed)
    near_removed = report.removed_count
    if near_removed:
        frame = frame.iloc[report.kept_indices].reset_index(drop=True)

    return frame, len(exact), near_removed


def build_ell_evaluation_set(
    full_frame: pd.DataFrame, assignment: dict[str, str], seed: int
) -> pd.DataFrame:
    """Assemble a dedicated ELL fairness set from held-out prompts only.

    Why this is separate from the main corpus: the fairness question is whether
    the detector produces more false positives on essays by English language
    learners. Answering it needs enough ELL essays to compute a rate with a
    usable confidence interval, and the working corpus happens to contain only
    a few dozen in the test split — too few to conclude anything.

    This set therefore draws *every* available ELL essay from the validation
    and test prompts, plus a size-matched non-ELL comparison group from the
    same prompts. Restricting to held-out prompts is what keeps it honest: none
    of these essays come from a prompt the model trained on. The training
    distribution is untouched, so enriching this set cannot flatter the model —
    it only makes the fairness measurement more precise.

    Matching on prompt matters too. ELL and non-ELL essays are not evenly
    spread across prompts, so an unmatched comparison would partly measure
    topic difficulty rather than the writer's language background.
    """
    held_out_prompts = {
        prompt for prompt, split in assignment.items() if split in {"validation", "test"}
    }
    human = full_frame[
        (full_frame["label"] == 0)
        & (full_frame["prompt_name"].isin(held_out_prompts))
        & (full_frame["ell_status"].notna())
    ]

    ell_essays = human[human["ell_status"] == "Yes"]
    non_ell_pool = human[human["ell_status"] == "No"]
    if ell_essays.empty or non_ell_pool.empty:
        return pd.DataFrame(columns=full_frame.columns)

    # Match the comparison group prompt by prompt.
    matched_chunks: list[pd.DataFrame] = []
    for prompt, ell_group in ell_essays.groupby("prompt_name"):
        candidates = non_ell_pool[non_ell_pool["prompt_name"] == prompt]
        take = min(len(ell_group), len(candidates))
        if take:
            matched_chunks.append(candidates.sample(take, random_state=seed))

    comparison = (
        pd.concat(matched_chunks) if matched_chunks else pd.DataFrame(columns=human.columns)
    )
    combined = pd.concat([ell_essays, comparison]).reset_index(drop=True)
    combined["split"] = combined["prompt_name"].map(assignment)
    combined["essay_id"] = combined["text"].map(essay_identifier)
    return combined


def summarize_by_split(frame: pd.DataFrame, column: str) -> dict[str, dict[str, int]]:
    """Counts of `column` values within each split, as plain JSON-safe dicts."""
    summary: dict[str, dict[str, int]] = {}
    for split_name, group in frame.groupby("split"):
        summary[str(split_name)] = {
            str(key): int(value) for key, value in Counter(group[column]).items()
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-class",
        type=int,
        default=4000,
        help=(
            "Essays per class in the working corpus. Feature extraction costs "
            "roughly a third of a second per essay, so this is the main lever "
            "on pipeline runtime."
        ),
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--similarity-threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD
    )
    arguments = parser.parse_args()

    for path in (DAIGT_PATH, PERSUADE_PATH):
        if not path.exists():
            raise SystemExit(f"Missing corpus at {path}.")

    print("Loading corpus and joining ELL status...", flush=True)
    frame = load_corpus()
    print(f"  loaded {len(frame):,} essays")
    ell_known = frame["ell_status"].notna().sum()
    print(f"  ELL status attached to {ell_known:,} human essays")

    full_frame = filter_by_length(frame)
    print(f"  {len(full_frame):,} remain after length filter "
          f"({MINIMUM_WORDS}-{MAXIMUM_WORDS} words)")

    print(f"Subsampling {arguments.per_class:,} per class...", flush=True)
    frame = stratified_subsample(full_frame, arguments.per_class, arguments.seed)
    print(f"  sampled {len(frame):,} essays")

    print("Removing duplicates...", flush=True)
    frame, exact_removed, near_removed = remove_duplicates(
        frame, arguments.similarity_threshold, arguments.seed
    )
    print(f"  removed {exact_removed:,} exact and {near_removed:,} near duplicates")
    print(f"  {len(frame):,} essays remain")

    print("Assigning prompt-disjoint splits...", flush=True)
    group_sizes = frame["prompt_name"].value_counts().to_dict()
    assignment = assign_groups_to_splits(group_sizes, SplitRatios())
    frame["split"] = frame["prompt_name"].map(assignment)
    frame["essay_id"] = frame["text"].map(essay_identifier)

    validate_split(
        groups=frame["prompt_name"].tolist(),
        splits=frame["split"].tolist(),
        labels=frame["label"].tolist(),
        generators=frame["generator"].tolist(),
    )
    print("  split validated: prompt-disjoint, both classes and >=2 generators present")

    manifest = build_manifest(
        group_column="prompt_name",
        assignment=assignment,
        ratios=SplitRatios(),
        seed=arguments.seed,
        similarity_threshold=arguments.similarity_threshold,
        total_essays=len(frame),
        exact_duplicates_removed=exact_removed,
        near_duplicates_removed=near_removed,
        split_counts={
            str(key): int(value) for key, value in frame["split"].value_counts().items()
        },
        label_counts=summarize_by_split(frame, "label"),
        generator_counts=summarize_by_split(frame, "generator"),
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    corpus_path = OUTPUT_DIRECTORY / "corpus.csv"
    manifest_path = OUTPUT_DIRECTORY / "split_manifest.json"

    columns = [
        "essay_id", "text", "label", "prompt_name", "generator",
        "ell_status", "grade_level", "split",
    ]
    frame[columns].to_csv(corpus_path, index=False)
    write_manifest(manifest, manifest_path)

    print(f"\nSplit hash: {manifest.split_hash}")
    print("\nEssays per split:")
    for split_name in ("train", "validation", "test"):
        subset = frame[frame["split"] == split_name]
        human = int((subset["label"] == 0).sum())
        machine = int((subset["label"] == 1).sum())
        prompts = subset["prompt_name"].nunique()
        generators = subset[subset["label"] == 1]["generator"].nunique()
        print(
            f"  {split_name:<11} {len(subset):>6,}  "
            f"(human {human:>5,} / machine {machine:>5,})  "
            f"prompts {prompts:>2}  generators {generators:>2}"
        )

    ell_evaluation = build_ell_evaluation_set(full_frame, assignment, arguments.seed)
    ell_path = OUTPUT_DIRECTORY / "ell_evaluation.csv"
    if not ell_evaluation.empty:
        ell_evaluation[columns].to_csv(ell_path, index=False)
        counts = Counter(ell_evaluation["ell_status"])
        print(
            f"\nELL fairness set (held-out prompts only): {len(ell_evaluation):,} essays"
            f"  [ELL={counts.get('Yes', 0):,}, non-ELL={counts.get('No', 0):,}]"
        )
        print(f"  written to {ell_path}")
    else:
        print("\nNo ELL essays available in held-out prompts; fairness set not built.")

    print(f"\nCorpus written to {corpus_path}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
