"""Train the baseline detector and record its honest validation performance.

Order of operations matters here and is deliberate:

1. Fit the **sentence model** on training-split sentences.
2. Generate **out-of-fold** sentence scores for the training split, using
   GroupKFold over prompts. This is the step most easily skipped and most
   damaging to skip: the document model consumes summaries of sentence scores,
   and if those summaries came from a model that had already seen the same
   sentences in training, they would be unrealistically clean. The document
   model would learn to trust them, and would then be disappointed by the
   ordinary, noisier scores it sees at test time.
3. Fit the **document model** on document features plus those out-of-fold
   summaries.
4. Report validation metrics, including a length-artefact ablation and a
   calibration check.

Nothing here touches the test split. Test metrics come from
`scripts/evaluate.py`, run once, after the baseline is recorded.

Usage:
    python3 scripts/train.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "backend"))

from app.ml.detector import DetectorArtifacts  # noqa: E402
from app.ml.evaluation import (  # noqa: E402
    build_document_frame,
    classification_metrics,
    expected_calibration_error,
    reliability_table,
)
from app.ml.feature_matrix import FeatureMatrixBuilder  # noqa: E402
from app.services.evidence import ReferenceStatistics  # noqa: E402

PROCESSED = REPOSITORY_ROOT / "data/processed"
MODELS = REPOSITORY_ROOT / "data/models"

# Regularization strength for both models. Chosen once, before seeing any
# result, and left alone; tuning it against validation and then reporting
# validation numbers would make those numbers optimistic.
REGULARIZATION_C = 1.0
MAXIMUM_ITERATIONS = 2000
CROSS_VALIDATION_FOLDS = 5


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    documents = pd.read_parquet(PROCESSED / "document_features.parquet")
    sentences = pd.read_parquet(PROCESSED / "sentence_features.parquet")
    return documents, sentences


def fit_logistic(matrix: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(
        C=REGULARIZATION_C,
        max_iter=MAXIMUM_ITERATIONS,
        # Machine and human counts differ per split; balancing keeps the
        # decision threshold meaningful at 0.5 rather than drifting with the
        # class ratio of whichever corpus we happened to assemble.
        class_weight="balanced",
    )
    model.fit(matrix, labels)
    return model


def out_of_fold_sentence_scores(
    sentences: pd.DataFrame, builder_template: FeatureMatrixBuilder
) -> np.ndarray:
    """Sentence scores for training rows, each predicted by a model that never saw them."""
    scores = np.zeros(len(sentences))
    groups = sentences["prompt_name"].to_numpy()
    labels = sentences["label"].to_numpy()

    splitter = GroupKFold(n_splits=min(CROSS_VALIDATION_FOLDS, len(np.unique(groups))))
    for train_index, held_index in splitter.split(sentences, labels, groups):
        fold_train = sentences.iloc[train_index]
        fold_held = sentences.iloc[held_index]

        builder = FeatureMatrixBuilder(
            exclude_length_artefacts=builder_template.exclude_length_artefacts
        ).fit(fold_train)
        model = fit_logistic(builder.transform(fold_train), fold_train["label"].to_numpy())
        scores[held_index] = model.predict_proba(builder.transform(fold_held))[:, 1]

    return scores


def train_pipeline(
    documents: pd.DataFrame,
    sentences: pd.DataFrame,
    exclude_length_artefacts: bool,
) -> tuple[DetectorArtifacts, dict[str, object]]:
    """Fit both models and return them with validation metrics."""
    train_sentences = sentences[
        (sentences["split"] == "train") & (sentences["is_scorable"])
    ].reset_index(drop=True)
    validation_sentences = sentences[
        (sentences["split"] == "validation") & (sentences["is_scorable"])
    ].reset_index(drop=True)

    sentence_builder = FeatureMatrixBuilder(
        exclude_length_artefacts=exclude_length_artefacts
    ).fit(train_sentences)
    sentence_model = fit_logistic(
        sentence_builder.transform(train_sentences),
        train_sentences["label"].to_numpy(),
    )

    validation_sentence_scores = sentence_model.predict_proba(
        sentence_builder.transform(validation_sentences)
    )[:, 1]
    sentence_metrics = classification_metrics(
        validation_sentences["label"].to_numpy(), validation_sentence_scores
    )

    # Out-of-fold scores for training documents; direct predictions elsewhere.
    train_all_sentences = sentences[sentences["split"] == "train"].reset_index(drop=True)
    train_oof = out_of_fold_sentence_scores(train_all_sentences, sentence_builder)

    other_sentences = sentences[sentences["split"] != "train"].reset_index(drop=True)
    other_scores = sentence_model.predict_proba(
        sentence_builder.transform(other_sentences)
    )[:, 1]

    all_sentences = pd.concat([train_all_sentences, other_sentences]).reset_index(drop=True)
    all_scores = np.concatenate([train_oof, other_scores])

    document_frame = build_document_frame(documents, all_sentences, all_scores)
    train_documents = document_frame[document_frame["split"] == "train"]
    validation_documents = document_frame[document_frame["split"] == "validation"]

    document_builder = FeatureMatrixBuilder(
        exclude_length_artefacts=exclude_length_artefacts
    ).fit(train_documents)
    document_model = fit_logistic(
        document_builder.transform(train_documents),
        train_documents["label"].to_numpy(),
    )

    validation_probabilities = document_model.predict_proba(
        document_builder.transform(validation_documents)
    )[:, 1]
    validation_labels = validation_documents["label"].to_numpy()
    document_metrics = classification_metrics(validation_labels, validation_probabilities)

    reliability = reliability_table(validation_labels, validation_probabilities)
    calibration_error = expected_calibration_error(reliability, len(validation_labels))

    artifacts = DetectorArtifacts(
        sentence_builder=sentence_builder,
        sentence_model=sentence_model,
        document_builder=document_builder,
        document_model=document_model,
        metadata={
            "exclude_length_artefacts": exclude_length_artefacts,
            "regularization_c": REGULARIZATION_C,
            "sentence_feature_count": len(sentence_builder.column_names),
            "document_feature_count": len(document_builder.column_names),
            "training_sentences": int(len(train_sentences)),
            "training_documents": int(len(train_documents)),
        },
    )

    metrics = {
        "sentence_validation": sentence_metrics,
        "document_validation": document_metrics,
        "calibration": {
            "expected_calibration_error": calibration_error,
            "reliability": reliability,
        },
    }
    return artifacts, metrics


def build_reference_statistics(
    sentences: pd.DataFrame, documents: pd.DataFrame
) -> ReferenceStatistics:
    """Mean and standard deviation of each feature over training human writing.

    Humans only, and training split only. This is the distribution the evidence
    panel compares against when it tells a user that some property of their
    writing sits above or below "the human average", so it must describe human
    writing specifically and must not include any essay used for evaluation.

    Both scopes are covered. An earlier version summarized sentence features
    only, which left every document-level signal in the evidence panel with no
    reference to compare against — the panel rendered "z=n/a" for exactly the
    signals driving the headline score, and fell back to saying merely which
    way each pushed. Document feature names are disjoint from sentence ones
    (`doc*`, `rhythm_*` versus `lex_*`, `syn_*`), so one merged mapping serves
    both without collision.
    """
    human_sentences = sentences[
        (sentences["split"] == "train")
        & (sentences["label"] == 0)
        & (sentences["is_scorable"])
    ]
    human_documents = documents[
        (documents["split"] == "train") & (documents["label"] == 0)
    ]

    means: dict[str, float] = {}
    deviations: dict[str, float] = {}

    for frame in (human_sentences, human_documents):
        numeric = frame.select_dtypes(include="number")
        for column in numeric.columns:
            if column in ("label",) or not numeric[column].notna().any():
                continue
            means[column] = float(numeric[column].mean())
            deviations[column] = float(numeric[column].std(ddof=0))

    return ReferenceStatistics(means=means, standard_deviations=deviations)


def top_coefficients(
    artifacts: DetectorArtifacts, count: int = 15
) -> list[dict[str, object]]:
    """Largest-magnitude document-model coefficients, with their direction."""
    names = artifacts.document_builder.column_names
    coefficients = artifacts.document_model.coef_[0]
    order = np.argsort(np.abs(coefficients))[::-1][:count]
    return [
        {
            "feature": names[index],
            "coefficient": float(coefficients[index]),
            "points_toward": "machine" if coefficients[index] > 0 else "human",
        }
        for index in order
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MODELS / "baseline")
    arguments = parser.parse_args()

    documents, sentences = load_tables()
    print(
        f"Loaded {len(documents):,} documents and {len(sentences):,} sentences.\n",
        flush=True,
    )

    results: dict[str, object] = {}

    for exclude in (True, False):
        label = "without length features" if exclude else "with length features"
        print(f"=== Training {label} ===", flush=True)
        artifacts, metrics = train_pipeline(documents, sentences, exclude)

        document_metrics = metrics["document_validation"]
        sentence_metrics = metrics["sentence_validation"]
        print(
            f"  sentence  AUC {sentence_metrics['roc_auc']:.4f}  "
            f"acc {sentence_metrics['accuracy']:.4f}"
        )
        print(
            f"  document  AUC {document_metrics['roc_auc']:.4f}  "
            f"acc {document_metrics['accuracy']:.4f}  "
            f"F1 {document_metrics['f1']:.4f}  "
            f"FPR {document_metrics['false_positive_rate']:.4f}"
        )
        print(
            f"  calibration ECE "
            f"{metrics['calibration']['expected_calibration_error']:.4f}\n"
        )

        key = "excluded" if exclude else "included"
        results[f"length_features_{key}"] = metrics

        if exclude:
            artifacts.save(arguments.output)
            build_reference_statistics(sentences, documents).save(
                arguments.output / "reference_statistics.json"
            )
            results["top_coefficients"] = top_coefficients(artifacts)

    metrics_path = arguments.output / "validation_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("Top document-model coefficients:")
    for entry in results["top_coefficients"][:12]:  # type: ignore[index]
        print(
            f"  {entry['coefficient']:+7.3f}  {entry['feature']:<45} "
            f"-> {entry['points_toward']}"
        )

    print(f"\nArtifacts saved to {arguments.output}")
    print(f"Validation metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
