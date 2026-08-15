"""Recompute predictability features out-of-fold, removing a training leak.

The bug this fixes
------------------
`extract_features.py` fits the reference n-gram language model on training-split
human essays, then uses it to score every essay — including the very essays it
was fit on. Training humans are therefore scored in-sample and look
artificially predictable, while every other human in the corpus is scored
out-of-sample and looks unpredictable.

The classifier learns the only rule consistent with that data — "unpredictable
means machine" — and then flags 77% of validation humans. Measured: validation
AUC 0.801 with the feature, 0.951 without it.

The feature was not measuring how predictable the text is. It was measuring
whether the text had been in the language model's training set.

The fix
-------
Out-of-fold computation, the same principle already used for sentence scores in
`train.py`. Training prompts are divided into folds; for each fold, a language
model is fit on human sentences from the *other* folds and used to score the
held-out fold. Validation, test and ELL essays keep the full model fit on all
training humans, which is already out-of-sample for them.

Every essay is then scored by a language model that never saw it, so the
feature means the same thing in every split — which is the only condition under
which a classifier can learn something true from it.

Why not simply delete the feature
---------------------------------
Because "predictability does not help" and "my predictability feature was
broken" are different conclusions, and only the second one is currently
supported. After this script runs, `train.py` reports the honest comparison and
the feature is kept or dropped on that evidence.

Usage:
    python3 scripts/recompute_predictability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "backend"))

from app.nlp.features.base import word_forms  # noqa: E402
from app.nlp.features.predictability import NgramLanguageModel  # noqa: E402
from app.nlp.pipeline import get_nlp  # noqa: E402
from app.nlp.preprocessing import normalize  # noqa: E402

PROCESSED = REPOSITORY_ROOT / "data/processed"
MODELS = REPOSITORY_ROOT / "data/models"

FOLD_COUNT = 5


def tokenize_sentences(
    corpus: pd.DataFrame, sentences: pd.DataFrame
) -> list[list[str]]:
    """Recover each sentence's tokens from stored character offsets.

    Only the tokenizer is used, not the parser: `is_punct` and `is_space` are
    lexical attributes, so `word_forms` works without the expensive pipeline
    components. That turns a fifteen-minute job into a one-minute one, and
    produces exactly the same tokens as the original extraction because it is
    the same tokenizer.
    """
    text_by_id = dict(zip(corpus["essay_id"], corpus["text"]))
    normalized_cache: dict[str, str] = {}

    tokenizer = get_nlp().tokenizer
    tokens: list[list[str]] = []

    for essay_id, start, end in zip(
        sentences["essay_id"], sentences["start"], sentences["end"]
    ):
        if essay_id not in normalized_cache:
            normalized_cache[essay_id] = normalize(str(text_by_id[essay_id]))
        sentence_text = normalized_cache[essay_id][int(start) : int(end)]
        tokens.append(word_forms(tokenizer(sentence_text)))

    return tokens


def score_with_model(
    model: NgramLanguageModel, tokens: list[list[str]], indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-entropy and perplexity for the sentences at `indices`."""
    cross_entropy = np.full(len(indices), np.nan)
    perplexity = np.full(len(indices), np.nan)

    for position, index in enumerate(indices):
        value = model.cross_entropy(tokens[index])
        if value is not None:
            cross_entropy[position] = value
            perplexity[position] = 2.0**value

    return cross_entropy, perplexity


def main() -> None:
    corpus = pd.read_csv(PROCESSED / "corpus.csv")
    sentences = pd.read_parquet(PROCESSED / "sentence_features.parquet")
    print(f"Loaded {len(sentences):,} sentences.", flush=True)

    print("Tokenizing sentences from stored offsets...", flush=True)
    tokens = tokenize_sentences(corpus, sentences)

    cross_entropy = np.full(len(sentences), np.nan)
    perplexity = np.full(len(sentences), np.nan)

    is_train = (sentences["split"] == "train").to_numpy()
    is_human = (sentences["label"] == 0).to_numpy()
    prompts = sentences["prompt_name"].to_numpy()
    train_indices = np.flatnonzero(is_train)

    # --- training split: out-of-fold ---
    unique_train_prompts = np.unique(prompts[train_indices])
    fold_count = min(FOLD_COUNT, len(unique_train_prompts))
    print(
        f"Out-of-fold scoring across {fold_count} prompt folds "
        f"({len(unique_train_prompts)} training prompts)...",
        flush=True,
    )

    splitter = GroupKFold(n_splits=fold_count)
    for fold, (fit_positions, held_positions) in enumerate(
        splitter.split(train_indices, groups=prompts[train_indices]), start=1
    ):
        fit_indices = train_indices[fit_positions]
        held_indices = train_indices[held_positions]

        # Fit only on human sentences from the folds not being scored.
        fit_human = fit_indices[is_human[fit_indices]]
        model = NgramLanguageModel().fit(
            [tokens[index] for index in fit_human if tokens[index]]
        )

        fold_entropy, fold_perplexity = score_with_model(model, tokens, held_indices)
        cross_entropy[held_indices] = fold_entropy
        perplexity[held_indices] = fold_perplexity
        print(
            f"  fold {fold}: fit on {len(fit_human):,} human sentences, "
            f"scored {len(held_indices):,}",
            flush=True,
        )

    # --- everything else: the full training-human model, already out-of-sample ---
    other_indices = np.flatnonzero(~is_train)
    full_model = NgramLanguageModel.load(MODELS / "reference_ngram_lm.json")
    other_entropy, other_perplexity = score_with_model(full_model, tokens, other_indices)
    cross_entropy[other_indices] = other_entropy
    perplexity[other_indices] = other_perplexity
    print(f"  scored {len(other_indices):,} non-training sentences", flush=True)

    sentences["pred_cross_entropy"] = cross_entropy
    sentences["pred_perplexity"] = perplexity
    sentences.to_parquet(PROCESSED / "sentence_features.parquet", index=False)

    # --- rebuild the document-level summaries from the corrected values ---
    print("Rebuilding document-level predictability summaries...", flush=True)
    documents = pd.read_parquet(PROCESSED / "document_features.parquet")
    scorable = sentences[sentences["is_scorable"]]
    grouped = scorable.groupby("essay_id")["pred_cross_entropy"]
    summary = pd.DataFrame(
        {
            "docmean_pred_cross_entropy": grouped.mean(),
            "docstd_pred_cross_entropy": grouped.std(ddof=0),
        }
    )
    documents = documents.drop(
        columns=["docmean_pred_cross_entropy", "docstd_pred_cross_entropy"],
        errors="ignore",
    ).merge(summary, left_on="essay_id", right_index=True, how="left")
    documents.to_parquet(PROCESSED / "document_features.parquet", index=False)

    print("\nMean cross-entropy by split and class (should now be comparable):")
    merged = sentences.merge(
        sentences[["essay_id"]].drop_duplicates(), on="essay_id", how="left"
    )
    table = merged.groupby(["split", "label"])["pred_cross_entropy"].mean().unstack()
    print(table.round(3).to_string())
    print("\nDone.")


if __name__ == "__main__":
    main()
