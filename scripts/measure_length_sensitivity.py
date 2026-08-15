"""Measure how the detector degrades as the input gets shorter.

Motivation
----------
Scoring a handwritten seven-sentence paragraph produced risk 100/100 on text
that is plainly human. Real test essays (250-475 words) score correctly, so the
pipeline is sound — the failure is length. This script finds where the failure
starts instead of guessing at a threshold.

Method
------
Take held-out test essays, truncate each to the first N sentences, and score
the truncation. Because the label is unchanged by truncation, any change in
accuracy is attributable to length alone.

The output sets `MINIMUM_SENTENCES_FOR_VERDICT` in the analysis service and
supplies the numbers quoted in `docs/limitations.md`.

Usage:
    python3 scripts/measure_length_sensitivity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "backend"))

from app.nlp.preprocessing import normalize  # noqa: E402
from app.nlp.segmentation import segment  # noqa: E402
from app.services.analysis import get_analysis_service  # noqa: E402

PROCESSED = REPOSITORY_ROOT / "data/processed"
OUTPUT_PATH = REPOSITORY_ROOT / "data/models/baseline/length_sensitivity.json"

SENTENCE_COUNTS = [3, 5, 8, 12, 16, 20, 25]
ESSAYS_PER_CLASS = 60
RANDOM_SEED = 20260814


def truncate_to_sentences(text: str, count: int) -> str | None:
    """First `count` sentences of `text`, or None if it has fewer."""
    document = segment(normalize(text))
    sentences = document.sentences
    if len(sentences) < count:
        return None
    return " ".join(sentence.text for sentence in sentences[:count])


def main() -> None:
    service = get_analysis_service()
    corpus = pd.read_csv(PROCESSED / "corpus.csv")
    test = corpus[corpus["split"] == "test"]

    sample = pd.concat(
        [
            test[test["label"] == label].sample(
                min(ESSAYS_PER_CLASS, len(test[test["label"] == label])),
                random_state=RANDOM_SEED,
            )
            for label in (0, 1)
        ]
    )
    print(f"Scoring {len(sample)} test essays at {len(SENTENCE_COUNTS)} lengths...\n")

    rows: list[dict[str, object]] = []
    for count in SENTENCE_COUNTS:
        scores: list[float] = []
        labels: list[int] = []
        confidences: list[float] = []

        for _, record in sample.iterrows():
            truncated = truncate_to_sentences(str(record["text"]), count)
            if truncated is None:
                continue
            result = service.analyze(truncated)
            scores.append(result.risk_score / 100.0)
            labels.append(int(record["label"]))
            confidences.append(result.confidence)

        if not scores:
            continue

        score_array = np.array(scores)
        label_array = np.array(labels)
        predictions = (score_array >= 0.5).astype(int)

        human = label_array == 0
        machine = label_array == 1
        rows.append(
            {
                "sentences": count,
                "essays_scored": len(scores),
                "accuracy": float((predictions == label_array).mean()),
                "false_positive_rate": float(predictions[human].mean())
                if human.any()
                else float("nan"),
                "recall": float(predictions[machine].mean())
                if machine.any()
                else float("nan"),
                "mean_confidence": float(np.mean(confidences)),
            }
        )
        print(
            f"  {count:>2} sentences  n={len(scores):>3}  "
            f"accuracy {rows[-1]['accuracy']:.3f}  "
            f"FPR {rows[-1]['false_positive_rate']:.3f}  "
            f"recall {rows[-1]['recall']:.3f}  "
            f"mean confidence {rows[-1]['mean_confidence']:.3f}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWritten to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
