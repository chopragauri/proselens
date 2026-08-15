"""Evaluate the trained detector on the held-out test split.

Run once, after the baseline is recorded. Produces:

* Overall test metrics with a confusion matrix.
* A per-generator breakdown, because an average over seventeen models hides
  which ones the detector cannot see.
* The ELL fairness comparison (§32), with Wilson confidence intervals — the
  slice is small enough that a bare rate would overstate what we know.
* Grouped cross-validation across all prompts, to put an honest spread around
  the single test number, which comes from only two prompts.
* Three confidently wrong predictions with their actual feature contributions.

Nothing here refits anything. The threshold stays at 0.5.

Usage:
    python3 scripts/evaluate.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "backend"))

from app.ml.detector import DetectorArtifacts  # noqa: E402
from app.ml.evaluation import (  # noqa: E402
    build_document_frame,
    classification_metrics,
    expected_calibration_error,
    reliability_table,
    wilson_interval,
)
from app.ml.feature_matrix import FeatureMatrixBuilder  # noqa: E402

PROCESSED = REPOSITORY_ROOT / "data/processed"
MODELS = REPOSITORY_ROOT / "data/models"

FAILURE_CASE_COUNT = 3
CONTRIBUTIONS_SHOWN = 6


def score_everything(
    artifacts: DetectorArtifacts,
    documents: pd.DataFrame,
    sentences: pd.DataFrame,
) -> pd.DataFrame:
    """Attach document probabilities to `documents`."""
    sentence_scores = artifacts.score_sentences(sentences)
    frame = build_document_frame(documents, sentences, sentence_scores)
    frame = frame.copy()
    frame["probability"] = artifacts.score_document(frame)
    return frame


def per_generator_metrics(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Recall per machine generator, and false-positive rate per human source."""
    rows: list[dict[str, object]] = []
    for generator, group in frame.groupby("generator"):
        is_machine = bool(group["label"].iloc[0] == 1)
        detected = (group["probability"] >= 0.5).mean()
        rows.append(
            {
                "generator": str(generator),
                "kind": "machine" if is_machine else "human",
                "count": int(len(group)),
                "detected_as_machine": float(detected),
                "metric": "recall" if is_machine else "false_positive_rate",
            }
        )
    return sorted(rows, key=lambda row: (row["kind"], row["detected_as_machine"]))


def ell_fairness(
    artifacts: DetectorArtifacts, threshold: float = 0.5
) -> dict[str, object] | None:
    """False-positive rate for ELL versus prompt-matched non-ELL writers."""
    document_path = PROCESSED / "ell_document_features.parquet"
    sentence_path = PROCESSED / "ell_sentence_features.parquet"
    if not document_path.exists():
        return None

    documents = pd.read_parquet(document_path)
    sentences = pd.read_parquet(sentence_path)
    frame = score_everything(artifacts, documents, sentences)

    results: dict[str, object] = {"threshold": threshold}
    for status in ("Yes", "No"):
        group = frame[frame["ell_status"] == status]
        if group.empty:
            continue
        false_positives = int((group["probability"] >= threshold).sum())
        total = int(len(group))
        lower, upper = wilson_interval(false_positives, total)
        results["ell" if status == "Yes" else "non_ell"] = {
            "count": total,
            "false_positives": false_positives,
            "false_positive_rate": false_positives / total,
            "confidence_interval_95": [lower, upper],
            "mean_probability": float(group["probability"].mean()),
        }

    ell, non_ell = results.get("ell"), results.get("non_ell")
    if isinstance(ell, dict) and isinstance(non_ell, dict):
        results["rate_difference"] = (
            ell["false_positive_rate"] - non_ell["false_positive_rate"]
        )
        # Non-overlapping intervals are the minimum bar before calling a gap
        # real; with a few hundred essays per group, small gaps are noise.
        results["intervals_overlap"] = bool(
            ell["confidence_interval_95"][0] <= non_ell["confidence_interval_95"][1]
            and non_ell["confidence_interval_95"][0] <= ell["confidence_interval_95"][1]
        )
    return results


def grouped_cross_validation(
    documents: pd.DataFrame, sentences: pd.DataFrame, folds: int = 5
) -> dict[str, object]:
    """Prompt-grouped CV over the whole corpus, for a spread around the point estimate.

    The frozen test split contains only two prompts, so a single accuracy from
    it could be lucky or unlucky in ways nothing else would reveal. This refits
    from scratch on each fold — it does not reuse the trained model — and is
    reported alongside, not instead of, the test number.
    """
    sentence_scores_by_fold: list[float] = []
    document_scores: list[float] = []
    accuracies: list[float] = []

    groups = documents["prompt_name"].to_numpy()
    splitter = GroupKFold(n_splits=min(folds, len(np.unique(groups))))

    for train_index, held_index in splitter.split(documents, documents["label"], groups):
        train_documents = documents.iloc[train_index]
        held_documents = documents.iloc[held_index]

        train_ids = set(train_documents["essay_id"])
        train_sentences = sentences[
            sentences["essay_id"].isin(train_ids) & sentences["is_scorable"]
        ]
        held_sentences = sentences[~sentences["essay_id"].isin(train_ids)]

        sentence_builder = FeatureMatrixBuilder().fit(train_sentences)
        sentence_model = LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced"
        ).fit(
            sentence_builder.transform(train_sentences),
            train_sentences["label"].to_numpy(),
        )

        train_scores = sentence_model.predict_proba(
            sentence_builder.transform(
                sentences[sentences["essay_id"].isin(train_ids)]
            )
        )[:, 1]
        held_scores = sentence_model.predict_proba(
            sentence_builder.transform(held_sentences)
        )[:, 1]

        train_frame = build_document_frame(
            train_documents,
            sentences[sentences["essay_id"].isin(train_ids)],
            train_scores,
        )
        held_frame = build_document_frame(held_documents, held_sentences, held_scores)

        document_builder = FeatureMatrixBuilder().fit(train_frame)
        document_model = LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced"
        ).fit(document_builder.transform(train_frame), train_frame["label"].to_numpy())

        probabilities = document_model.predict_proba(
            document_builder.transform(held_frame)
        )[:, 1]
        labels = held_frame["label"].to_numpy()

        document_scores.append(float(roc_auc_score(labels, probabilities)))
        accuracies.append(float(((probabilities >= 0.5).astype(int) == labels).mean()))
        sentence_scores_by_fold.append(
            float(roc_auc_score(held_sentences["label"], held_scores))
        )

    return {
        "folds": len(document_scores),
        "document_auc_mean": float(np.mean(document_scores)),
        "document_auc_std": float(np.std(document_scores)),
        "document_auc_per_fold": document_scores,
        "document_accuracy_mean": float(np.mean(accuracies)),
        "document_accuracy_std": float(np.std(accuracies)),
        "sentence_auc_mean": float(np.mean(sentence_scores_by_fold)),
    }


def confidently_wrong_cases(
    artifacts: DetectorArtifacts,
    frame: pd.DataFrame,
    sentences: pd.DataFrame,
    corpus: pd.DataFrame,
    count: int = FAILURE_CASE_COUNT,
) -> list[dict[str, object]]:
    """The `count` most confident incorrect predictions, with real evidence.

    "Confidently wrong" is defined as the largest distance from the decision
    boundary among misclassified essays. Feature contributions come from the
    linear decomposition, so each explanation is arithmetic rather than a story
    written after the answer was known.
    """
    predictions = (frame["probability"] >= 0.5).astype(int)
    wrong = frame[predictions != frame["label"]].copy()
    if wrong.empty:
        return []

    wrong["distance_from_boundary"] = (wrong["probability"] - 0.5).abs()
    worst = wrong.nlargest(count, "distance_from_boundary")

    text_by_id = dict(zip(corpus["essay_id"], corpus["text"]))
    document_columns = artifacts.document_builder.column_names
    coefficients = artifacts.document_model.coef_[0]
    matrix = artifacts.document_builder.transform(worst)

    cases: list[dict[str, object]] = []
    for position, (_, record) in enumerate(worst.iterrows()):
        contributions = {
            name: float(value * coefficient)
            for name, value, coefficient in zip(
                document_columns, matrix[position], coefficients
            )
        }
        ranked = sorted(
            contributions.items(), key=lambda item: abs(item[1]), reverse=True
        )[:CONTRIBUTIONS_SHOWN]

        essay_sentences = sentences[sentences["essay_id"] == record["essay_id"]]
        text = str(text_by_id.get(record["essay_id"], ""))

        cases.append(
            {
                "essay_id": record["essay_id"],
                "actual": "human" if record["label"] == 0 else "machine",
                "predicted": "machine" if record["probability"] >= 0.5 else "human",
                "probability_machine": float(record["probability"]),
                "prompt_name": record["prompt_name"],
                "generator": record["generator"],
                "ell_status": record.get("ell_status"),
                "sentence_count": int(len(essay_sentences)),
                "word_count": len(text.split()),
                "top_contributions": [
                    {"feature": name, "log_odds_contribution": value}
                    for name, value in ranked
                ],
                "excerpt": " ".join(text.split()[:60]),
            }
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODELS / "baseline")
    parser.add_argument("--skip-cross-validation", action="store_true")
    arguments = parser.parse_args()

    artifacts = DetectorArtifacts.load(arguments.model)
    documents = pd.read_parquet(PROCESSED / "document_features.parquet")
    sentences = pd.read_parquet(PROCESSED / "sentence_features.parquet")
    corpus = pd.read_csv(PROCESSED / "corpus.csv")

    frame = score_everything(artifacts, documents, sentences)
    test_frame = frame[frame["split"] == "test"].copy()
    labels = test_frame["label"].to_numpy()
    probabilities = test_frame["probability"].to_numpy()

    results: dict[str, object] = {
        "model_metadata": artifacts.metadata,
        "test": classification_metrics(labels, probabilities),
    }

    reliability = reliability_table(labels, probabilities)
    results["test_calibration"] = {
        "expected_calibration_error": expected_calibration_error(
            reliability, len(labels)
        ),
        "reliability": reliability,
    }

    # Sentence-level test performance, judged against inherited document labels.
    test_sentences = sentences[
        (sentences["split"] == "test") & (sentences["is_scorable"])
    ]
    sentence_scores = artifacts.score_sentences(test_sentences)
    results["test_sentence"] = classification_metrics(
        test_sentences["label"].to_numpy(), sentence_scores
    )

    results["per_generator"] = per_generator_metrics(test_frame)
    results["ell_fairness"] = ell_fairness(artifacts)
    results["failure_cases"] = confidently_wrong_cases(
        artifacts, test_frame, sentences, corpus
    )

    if not arguments.skip_cross_validation:
        print("Running prompt-grouped cross-validation...", flush=True)
        results["cross_validation"] = grouped_cross_validation(documents, sentences)

    output_path = arguments.model / "evaluation.json"
    output_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    # ---- console report ----
    test = results["test"]
    print("\n=== Test set (held-out prompts) ===")
    print(f"  essays            {test['count']:,}")
    print(f"  accuracy          {test['accuracy']:.4f}")
    print(f"  precision         {test['precision']:.4f}")
    print(f"  recall            {test['recall']:.4f}")
    print(f"  F1                {test['f1']:.4f}")
    print(f"  ROC AUC           {test['roc_auc']:.4f}")
    print(f"  false positives   {test['false_positive_rate']:.4f}")
    print(f"  false negatives   {test['false_negative_rate']:.4f}")
    print(
        f"  confusion         TN {test['true_negative']}  FP {test['false_positive']}  "
        f"FN {test['false_negative']}  TP {test['true_positive']}"
    )
    print(
        f"  calibration ECE   "
        f"{results['test_calibration']['expected_calibration_error']:.4f}"
    )
    print(f"\n  sentence-level AUC {results['test_sentence']['roc_auc']:.4f}")

    if "cross_validation" in results:
        cross = results["cross_validation"]
        print(f"\n=== Prompt-grouped {cross['folds']}-fold cross-validation ===")
        print(
            f"  document AUC      {cross['document_auc_mean']:.4f} "
            f"+/- {cross['document_auc_std']:.4f}"
        )
        print(
            f"  document accuracy {cross['document_accuracy_mean']:.4f} "
            f"+/- {cross['document_accuracy_std']:.4f}"
        )
        print(f"  per fold          "
              f"{[round(value, 3) for value in cross['document_auc_per_fold']]}")

    print("\n=== Detection by generator (test set) ===")
    for row in results["per_generator"]:
        print(
            f"  {row['generator']:<36} n={row['count']:>4}  "
            f"{row['metric']:<20} {row['detected_as_machine']:.3f}"
        )

    fairness = results["ell_fairness"]
    if fairness:
        print("\n=== ELL fairness (human essays, held-out prompts) ===")
        for key, label in (("ell", "ELL writers"), ("non_ell", "non-ELL writers")):
            entry = fairness.get(key)
            if entry:
                lower, upper = entry["confidence_interval_95"]
                print(
                    f"  {label:<18} n={entry['count']:>4}  "
                    f"FPR {entry['false_positive_rate']:.4f}  "
                    f"95% CI [{lower:.4f}, {upper:.4f}]"
                )
        if "rate_difference" in fairness:
            print(f"  difference         {fairness['rate_difference']:+.4f}")
            print(f"  intervals overlap  {fairness['intervals_overlap']}")

    print("\n=== Confidently wrong predictions ===")
    for index, case in enumerate(results["failure_cases"], start=1):
        print(f"\n  Case {index}: {case['essay_id']}")
        print(f"    actual {case['actual']}, predicted {case['predicted']} "
              f"(P(machine) = {case['probability_machine']:.3f})")
        print(f"    prompt: {case['prompt_name']}  source: {case['generator']}")
        print(f"    {case['word_count']} words, {case['sentence_count']} sentences")
        print("    top contributions to the log-odds:")
        for contribution in case["top_contributions"]:
            print(f"      {contribution['log_odds_contribution']:+7.3f}  "
                  f"{contribution['feature']}")

    print(f"\nFull results written to {output_path}")


if __name__ == "__main__":
    main()
