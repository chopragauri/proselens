"""Measure how well each feature separates human from machine essays.

Run before training anything. The purpose is to find out whether the feature
set carries signal at all, and which groups carry it, so that model results
later can be attributed to specific measurable properties of the writing rather
than to an opaque fit.

Separation is reported as ROC AUC of the single feature used alone as a
classifier, which is threshold-free and directly interpretable: 0.50 is
worthless, 1.00 is perfect, and anything below 0.50 simply means the feature
points the other way. Cohen's d accompanies it as an effect size.

Usage:
    python3 scripts/inspect_features.py --per-class 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "backend"))

from app.nlp.features.aggregate import extract_features  # noqa: E402
from app.nlp.preprocessing import normalize  # noqa: E402
from app.nlp.segmentation import segment  # noqa: E402

DAIGT_PATH = REPOSITORY_ROOT / "data/raw/daigt_v2/train_v2_drcat_02.csv"

# Fixed so that repeated runs of this diagnostic are comparable to each other.
RANDOM_SEED = 20260814


def load_balanced_sample(per_class: int, seed: int) -> pd.DataFrame:
    """Draw a class-balanced sample, spread across prompts and generators.

    Sampling within each prompt keeps topic roughly matched between the classes,
    so a feature cannot score well merely by detecting what the essay is about.
    """
    frame = pd.read_csv(DAIGT_PATH)
    frame = frame[frame["text"].str.split().str.len().between(50, 1000)]

    sampled: list[pd.DataFrame] = []
    for label in (0, 1):
        class_frame = frame[frame["label"] == label]
        prompts = class_frame["prompt_name"].unique()
        per_prompt = max(1, per_class // max(len(prompts), 1))
        chunks = [
            group.sample(min(per_prompt, len(group)), random_state=seed)
            for _, group in class_frame.groupby("prompt_name")
        ]
        combined = pd.concat(chunks).sample(frac=1.0, random_state=seed)
        sampled.append(combined.head(per_class))

    return pd.concat(sampled).reset_index(drop=True)


def extract_document_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Extract document-level features for every essay in `frame`."""
    rows: list[dict[str, float | None]] = []
    total = len(frame)

    for position, (_, record) in enumerate(frame.iterrows(), start=1):
        if position % 50 == 0 or position == total:
            print(f"  extracted {position}/{total}", flush=True)
        document = segment(normalize(str(record["text"])))
        features = extract_features(document)
        if not features.sentence_features:
            continue
        row = dict(features.document_values)
        row["label"] = float(record["label"])
        rows.append(row)

    return pd.DataFrame(rows)


def roc_auc(values: np.ndarray, labels: np.ndarray) -> float:
    """AUC via the rank-sum identity, which needs no sklearn and no threshold."""
    ranks = pd.Series(values).rank().to_numpy()
    positive_count = float((labels == 1).sum())
    negative_count = float((labels == 0).sum())
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    positive_rank_sum = ranks[labels == 1].sum()
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)


def cohens_d(values: np.ndarray, labels: np.ndarray) -> float:
    """Standardized mean difference between the two classes."""
    machine, human = values[labels == 1], values[labels == 0]
    if len(machine) < 2 or len(human) < 2:
        return float("nan")
    pooled_variance = (
        (len(machine) - 1) * machine.var(ddof=1)
        + (len(human) - 1) * human.var(ddof=1)
    ) / (len(machine) + len(human) - 2)
    if pooled_variance <= 0:
        return float("nan")
    return (machine.mean() - human.mean()) / np.sqrt(pooled_variance)


def summarize_separation(features: pd.DataFrame) -> pd.DataFrame:
    """Rank features by how far their AUC departs from chance."""
    labels = features["label"].to_numpy()
    results: list[dict[str, object]] = []

    for column in features.columns:
        if column == "label":
            continue
        series = features[column]
        defined = series.notna().to_numpy()
        # A feature defined for only a handful of essays cannot be judged, and
        # its apparent separation would be an artefact of who it is defined for.
        if defined.sum() < 0.5 * len(series):
            continue

        values = series[defined].to_numpy(dtype=float)
        subset_labels = labels[defined]
        auc = roc_auc(values, subset_labels)
        results.append(
            {
                "feature": column,
                "auc": auc,
                "separation": abs(auc - 0.5),
                "cohens_d": cohens_d(values, subset_labels),
                "human_mean": values[subset_labels == 0].mean(),
                "machine_mean": values[subset_labels == 1].mean(),
                "defined_fraction": defined.mean(),
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values("separation", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=300)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "data/processed/feature_separation.csv",
    )
    arguments = parser.parse_args()

    if not DAIGT_PATH.exists():
        raise SystemExit(f"Missing corpus at {DAIGT_PATH}. Run the fetch step first.")

    print(f"Sampling {arguments.per_class} essays per class...", flush=True)
    sample = load_balanced_sample(arguments.per_class, arguments.seed)
    print(f"Sampled {len(sample)} essays. Extracting features...", flush=True)

    features = extract_document_frame(sample)
    separation = summarize_separation(features)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    separation.to_csv(arguments.output, index=False)

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)
    print("\nTop 25 features by class separation:\n")
    print(
        separation.head(25).to_string(
            index=False,
            float_format=lambda value: f"{value:8.4f}",
        )
    )
    print(f"\nFull table written to {arguments.output}")


if __name__ == "__main__":
    main()
