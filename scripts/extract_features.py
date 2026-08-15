"""Extract features for the whole working corpus and persist them.

Separated from training so that the expensive step runs once. Feature
extraction dominates the pipeline's runtime; model fitting on the resulting
table takes seconds, which means model choices can be explored without paying
for spaCy again.

Leakage control
---------------
The reference n-gram language model is fit on **training-split human essays
only**. Fitting it on all human text would let a test essay contribute to the
statistics used to score itself, and every predictability feature would then be
optimistically biased. The model is saved alongside the features so that the
API scores incoming essays against exactly the same reference distribution.

Usage:
    python3 scripts/extract_features.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "backend"))

from app.nlp.features.aggregate import extract_features  # noqa: E402
from app.nlp.features.base import word_forms  # noqa: E402
from app.nlp.features.predictability import NgramLanguageModel  # noqa: E402
from app.nlp.pipeline import get_nlp  # noqa: E402
from app.nlp.preprocessing import normalize  # noqa: E402
from app.nlp.segmentation import segment  # noqa: E402

PROCESSED = REPOSITORY_ROOT / "data/processed"
MODELS = REPOSITORY_ROOT / "data/models"

CORPUS_PATH = PROCESSED / "corpus.csv"
ELL_PATH = PROCESSED / "ell_evaluation.csv"
LANGUAGE_MODEL_PATH = MODELS / "reference_ngram_lm.json"


def fit_reference_language_model(corpus: pd.DataFrame) -> NgramLanguageModel:
    """Fit the trigram reference model on training-split human essays only."""
    training_human = corpus[(corpus["split"] == "train") & (corpus["label"] == 0)]
    print(
        f"Fitting reference language model on {len(training_human):,} "
        "training human essays...",
        flush=True,
    )

    nlp = get_nlp()
    tokenized: list[list[str]] = []
    for parsed in nlp.pipe(training_human["text"].map(normalize).tolist(), batch_size=32):
        for sentence in parsed.sents:
            tokens = word_forms(sentence)
            if tokens:
                tokenized.append(tokens)

    print(f"  {len(tokenized):,} sentences collected", flush=True)
    model = NgramLanguageModel().fit(tokenized)
    MODELS.mkdir(parents=True, exist_ok=True)
    model.save(LANGUAGE_MODEL_PATH)
    size_mb = LANGUAGE_MODEL_PATH.stat().st_size / 1e6
    print(f"  saved to {LANGUAGE_MODEL_PATH.name} ({size_mb:.1f} MB)", flush=True)
    return model


def extract_corpus_features(
    frame: pd.DataFrame, language_model: NgramLanguageModel, label: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (document_features, sentence_features) for every essay in `frame`."""
    document_rows: list[dict[str, object]] = []
    sentence_rows: list[dict[str, object]] = []

    total = len(frame)
    started = time.time()

    for position, (_, record) in enumerate(frame.iterrows(), start=1):
        if position % 250 == 0 or position == total:
            elapsed = time.time() - started
            rate = position / elapsed if elapsed else 0.0
            remaining = (total - position) / rate if rate else 0.0
            print(
                f"  [{label}] {position:,}/{total:,}  "
                f"{rate:.1f} essays/s  ~{remaining / 60:.1f} min left",
                flush=True,
            )

        document = segment(normalize(str(record["text"])))
        features = extract_features(document, language_model=language_model)
        if not features.sentence_features:
            continue

        metadata = {
            "essay_id": record["essay_id"],
            "label": int(record["label"]),
            "split": record["split"],
            "prompt_name": record["prompt_name"],
            "generator": record["generator"],
            "ell_status": record.get("ell_status"),
        }

        document_rows.append({**metadata, **features.document_values})

        for sentence_features in features.sentence_features:
            sentence_rows.append(
                {
                    **metadata,
                    "sentence_index": sentence_features.sentence.index,
                    "start": sentence_features.sentence.start,
                    "end": sentence_features.sentence.end,
                    "is_scorable": sentence_features.sentence.is_scorable,
                    **sentence_features.values,
                }
            )

    return pd.DataFrame(document_rows), pd.DataFrame(sentence_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-ell", action="store_true")
    arguments = parser.parse_args()

    if not CORPUS_PATH.exists():
        raise SystemExit(f"Missing {CORPUS_PATH}. Run scripts/prepare_dataset.py first.")

    corpus = pd.read_csv(CORPUS_PATH)
    print(f"Loaded {len(corpus):,} essays.", flush=True)

    language_model = fit_reference_language_model(corpus)

    print("\nExtracting corpus features...", flush=True)
    documents, sentences = extract_corpus_features(corpus, language_model, "corpus")
    documents.to_parquet(PROCESSED / "document_features.parquet", index=False)
    sentences.to_parquet(PROCESSED / "sentence_features.parquet", index=False)
    print(
        f"  wrote {len(documents):,} document rows and {len(sentences):,} sentence rows",
        flush=True,
    )

    if not arguments.skip_ell and ELL_PATH.exists():
        print("\nExtracting ELL fairness-set features...", flush=True)
        ell_frame = pd.read_csv(ELL_PATH)
        ell_documents, ell_sentences = extract_corpus_features(
            ell_frame, language_model, "ell"
        )
        ell_documents.to_parquet(PROCESSED / "ell_document_features.parquet", index=False)
        ell_sentences.to_parquet(PROCESSED / "ell_sentence_features.parquet", index=False)
        print(f"  wrote {len(ell_documents):,} ELL document rows", flush=True)

    print("\nFeature extraction complete.")


if __name__ == "__main__":
    main()
