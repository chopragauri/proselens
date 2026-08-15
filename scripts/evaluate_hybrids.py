"""Span-level evaluation on hybrid essays.

Why this is the most important evaluation in the project
---------------------------------------------------------
Every other sentence-level number is measured against labels *inherited* from
the document: a sentence inside a machine essay is called machine. That is
noisy by construction and it cannot answer the question the product actually
makes: **does the highlighting point at the sentences a machine wrote?**

Hybrids can answer it, because we chose which sentences to rewrite and so know
the truth for every span.

Label assignment
----------------
The hybrid text is re-segmented from scratch, exactly as a user's pasted essay
would be. The detector's sentence boundaries therefore need not match the ones
used during generation — a rewrite can merge or split sentences. Each detected
sentence is labelled by character overlap: machine if more than half its
characters fall inside a rewritten span. Anything else would quietly assume the
segmentation agreed, which is the sort of assumption that turns a hard
evaluation into an easy one.

Usage:
    /opt/homebrew/bin/python3 scripts/evaluate_hybrids.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "backend"))

from app.ml.detector import DetectorArtifacts  # noqa: E402
from app.ml.evaluation import build_document_frame, classification_metrics  # noqa: E402
from app.nlp.features.aggregate import extract_features  # noqa: E402
from app.nlp.features.predictability import NgramLanguageModel  # noqa: E402
from app.nlp.preprocessing import normalize  # noqa: E402
from app.nlp.segmentation import segment  # noqa: E402

PROCESSED = REPOSITORY_ROOT / "data/processed"
MODELS = REPOSITORY_ROOT / "data/models"
HYBRID_PATH = PROCESSED / "hybrid_essays.jsonl"

# A detected sentence counts as machine-written when more than this share of
# its characters lie inside a rewritten span.
OVERLAP_THRESHOLD = 0.5


def overlap_fraction(start: int, end: int, spans: list[dict]) -> float:
    """Share of [start, end) covered by rewritten spans."""
    length = end - start
    if length <= 0:
        return 0.0
    covered = 0
    for span in spans:
        if not span["is_rewritten"]:
            continue
        covered += max(0, min(end, span["end"]) - max(start, span["start"]))
    return covered / length


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODELS / "baseline")
    arguments = parser.parse_args()

    if not HYBRID_PATH.exists():
        raise SystemExit(f"Missing {HYBRID_PATH}. Run scripts/generate_hybrids.py.")

    records = [json.loads(line) for line in HYBRID_PATH.open(encoding="utf-8")]
    print(f"Loaded {len(records):,} hybrid essays.", flush=True)

    artifacts = DetectorArtifacts.load(arguments.model)
    language_model = NgramLanguageModel.load(MODELS / "reference_ngram_lm.json")

    sentence_rows: list[dict] = []
    document_rows: list[dict] = []

    for position, record in enumerate(records, start=1):
        if position % 50 == 0 or position == len(records):
            print(f"  scored {position}/{len(records)}", flush=True)

        text = normalize(record["hybrid_text"])
        document = segment(text)
        features = extract_features(document, language_model=language_model)
        if not features.sentence_features:
            continue

        essay_id = record["source_essay_id"]
        for sentence_features in features.sentence_features:
            sentence = sentence_features.sentence
            share = overlap_fraction(sentence.start, sentence.end, record["spans"])
            sentence_rows.append(
                {
                    "essay_id": essay_id,
                    "label": int(share > OVERLAP_THRESHOLD),
                    "overlap": share,
                    "is_scorable": sentence.is_scorable,
                    "split": record["split"],
                    "prompt_name": record["prompt_name"],
                    "generator": "gemini-hybrid",
                    **sentence_features.values,
                }
            )

        document_rows.append(
            {
                "essay_id": essay_id,
                "label": 1,  # every hybrid contains machine-written text
                "split": record["split"],
                "prompt_name": record["prompt_name"],
                "generator": "gemini-hybrid",
                "rewritten_fraction": record["rewritten_fraction"],
                **features.document_values,
            }
        )

    sentences = pd.DataFrame(sentence_rows)
    documents = pd.DataFrame(document_rows)

    # ---- span-level ----
    scorable = sentences[sentences["is_scorable"]].reset_index(drop=True)
    span_scores = artifacts.score_sentences(scorable)
    span_labels = scorable["label"].to_numpy()
    span_metrics = classification_metrics(span_labels, span_scores)

    # ---- document-level ----
    all_scores = artifacts.score_sentences(sentences)
    document_frame = build_document_frame(documents, sentences, all_scores)
    document_scores = artifacts.score_document(document_frame)

    flagged = float((document_scores >= 0.5).mean())

    # How detection varies with how much of the essay was rewritten.
    document_frame = document_frame.assign(_probability=document_scores)
    bins = pd.cut(
        document_frame["rewritten_fraction"], [0.0, 0.35, 0.45, 0.55, 1.0]
    )
    by_fraction = (
        document_frame.groupby(bins, observed=True)
        .agg(
            essays=("_probability", "size"),
            detected=("_probability", lambda values: float((values >= 0.5).mean())),
            mean_risk=("_probability", "mean"),
        )
        .reset_index()
    )

    results = {
        "hybrid_count": len(records),
        "sentence_count": int(len(sentences)),
        "scorable_sentence_count": int(len(scorable)),
        "rewritten_sentence_count": int(span_labels.sum()),
        "span_level": span_metrics,
        "document_level": {
            "essays": int(len(document_frame)),
            "flagged_as_machine": flagged,
            "mean_risk": float(document_scores.mean()),
        },
        "by_rewritten_fraction": [
            {
                "range": str(row["rewritten_fraction"]),
                "essays": int(row["essays"]),
                "detected": float(row["detected"]),
                "mean_risk": float(row["mean_risk"]),
            }
            for _, row in by_fraction.iterrows()
        ],
    }

    output = arguments.model / "hybrid_evaluation.json"
    output.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    print("\n=== Span-level detection (does highlighting find the rewrites?) ===")
    print(f"  scorable sentences   {span_metrics['count']:,}")
    print(f"  actually rewritten   {int(span_labels.sum()):,}")
    print(f"  ROC AUC              {span_metrics['roc_auc']:.4f}")
    print(f"  precision            {span_metrics['precision']:.4f}")
    print(f"  recall               {span_metrics['recall']:.4f}")
    print(f"  F1                   {span_metrics['f1']:.4f}")
    print(f"  false positive rate  {span_metrics['false_positive_rate']:.4f}")

    print("\n=== Document-level on hybrids (unseen generator) ===")
    print(f"  essays               {len(document_frame):,}")
    print(f"  flagged as machine   {flagged:.4f}")
    print(f"  mean risk            {document_scores.mean():.4f}")

    print("\n=== Detection by how much was rewritten ===")
    for row in results["by_rewritten_fraction"]:
        print(
            f"  {row['range']:<14} n={row['essays']:>4}  "
            f"detected {row['detected']:.3f}  mean risk {row['mean_risk']:.3f}"
        )

    print(f"\nWritten to {output}")


if __name__ == "__main__":
    main()
