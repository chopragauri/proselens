# Methodology

How ProseLens decides, and why each decision was made that way.

---

## The two-model architecture

The product requires per-sentence scores. The data provides per-document
labels. Bridging that gap is the core design problem.

**Decision.** Two logistic regressions: a sentence model over sentence-scoped
features, and a document model over document-scoped features plus summary
statistics of the sentence-score distribution.

**Why.** Many features — burstiness, coefficient of variation, length
variance — are mathematically undefined for a single sentence. Running one
document-trained model over individual sentences would produce numbers, but
meaningless ones, and the evidence panel would then be explaining noise. The
feature framework enforces this through `FeatureScope`: a sentence classifier
can only consume SENTENCE and CONTEXTUAL features.

**Alternative.** One model applied at both granularities. Simpler, and it is
what most implementations do.

**Trade-off.** Sentence training labels are noisy — each sentence inherits its
document's label, so human sentences inside machine essays are mislabelled and
vice versa. We accept noisy training labels in exchange for honest evaluation:
the hybrid set provides real span labels, so the sentence model is *evaluated*
against ground truth even though it is *trained* on inherited labels.

**Verification.** Span-level metrics on 200 hybrid essays — AUC 0.7546. See
[evaluation.md](evaluation.md).

---

## Why the document model does not average sentence scores

An essay with every sentence at 0.5 and an essay with half its sentences at 0.1
and half at 0.9 have identical means and completely different stories. The
second is what partial machine editing looks like — the realistic threat.

So the document model receives the distribution's *shape*: mean, standard
deviation, min, max, range, five quantiles, high-risk fraction, and top-quartile
mean. It learns what to do with them rather than us guessing a formula.

`test_summary_distinguishes_distributions_that_share_a_mean` pins this.

---

## Risk versus confidence

Two separate outputs, never combined.

**Risk** is the document model's estimate that the text is machine-generated.

**Confidence** is a defined function — not a model output — of three
multiplicative components, each in [0, 1]:

- **Quantity** — scorable sentences, saturating at 12.
- **Coverage** — the share of sentences that were scorable at all.
- **Consistency** — `1 − min(1, 2σ)` over the sentence scores.

Because it is a formula over measurable quantities, it can be explained
exactly. A short essay of unusual sentences legitimately produces high risk with
low confidence, and saying so is more honest than one number that conflates
them.

---

## Calibration

Measured expected calibration error on the test split is **0.0141**, and the
reliability table tracks well at the extremes:

| Predicted | Observed | n |
|---|---|---|
| 0.011 | 0.015 | 471 |
| 0.143 | 0.194 | 31 |
| 0.662 | **0.364** | 11 |
| 0.850 | 0.929 | 28 |
| 0.988 | 0.981 | 366 |

**Decision.** The UI reports "Risk 62/100", never "62% probability".

**Why.** The mid-range bins — exactly where a user needs the number to mean
something — are thin and unstable. The [0.6, 0.7) bin predicts 0.662 against an
observed 0.364 on eleven essays. Calling the output a probability would borrow
authority the evaluation does not support, and calibration under domain shift
is not guaranteed by calibration on this test set.

---

## Length gating

Truncating held-out test essays and rescoring shows the failure is entirely
one-sided. Recall stays at or above 0.95 at every length; what collapses is the
false-positive rate on human writing:

| Sentences | 3 | 5 | 8 | 12 | 16 | 25 |
|---|---|---|---|---|---|---|
| FPR | 0.695 | 0.475 | 0.220 | 0.075 | 0.025 | 0.000 |

Two gates follow, both set from these numbers rather than intuition:

- **Below 5 scorable sentences**: no verdict at all. Flagging roughly half of
  all human writing is worse than useless.
- **Below 12**: the score and highlighting are shown, but a "likely machine"
  verdict is withheld and the assessment is capped at mixed signals. The user
  can still see the analysis; what is withheld is the accusation.

---

## Excluded features

Absolute document size is excluded by default: `doc_token_count`,
`doc_unique_count`, `doc_sentence_count`, `sentscore_scorable_count`,
`total_sentence_count`.

In this corpus machine essays are shorter than human ones (340 vs 397 words),
so length carries real signal — but signal about *how the dataset was
assembled*, not about how machines write. A real admissions essay has a word
limit.

Cost of excluding them: AUC 0.9725 versus 0.9775. Cheap.

---

## Two leaks found and fixed

Both were introduced by me and both are documented here rather than quietly
patched, because catching them is the part worth understanding.

### 1. The in-sample language model

The reference trigram model was fit on training-split human essays, then used
to score every essay — including the ones it was fit on. Training humans were
scored **in-sample** and looked artificially predictable. The classifier learned
the only rule consistent with that data — "unpredictable means machine" — and
then flagged every validation human writing on an unseen prompt.

**Symptom.** 77% of validation human essays classified as machine. AUC 0.838
with accuracy 0.514 and ECE 0.477.

**Diagnosis.** Dropping the predictability features entirely *raised* validation
AUC from 0.801 to 0.951. A feature that actively harms a model is usually
measuring something other than what it claims. This one was measuring *"was this
essay in the language model's training set"*.

**Fix.** Out-of-fold computation across prompt folds
(`scripts/recompute_predictability.py`). Every essay is now scored by a language
model that never saw it, so the feature means the same thing in every split.

**Result.** Validation accuracy 0.51 → 0.92, ECE 0.477 → 0.032.

### 2. The length proxy through the back door

Three document-size columns were excluded as dataset artifacts — and then
`sentscore_scorable_count` and `total_sentence_count` arrived through the
sentence-score summary and became the **largest coefficient in the document
model**. The exclusion was decorative until this was closed.

---

## Evidence generation

The classifier is linear on standardized inputs, so `coefficient ×
standardized_value` is exactly the log-odds that feature contributed.
`test_contributions_decompose_the_log_odds` asserts the contributions sum to
the actual log-odds within 1e-6.

Contributions are grouped into signal families, summed, and ranked. Each signal
carries a **z-score against the human reference distribution** — the mean and
standard deviation of that feature over *training human sentences only* — which
is what lets the panel say "comma density is 2.1 standard deviations above the
human average" rather than the uninterpretable "comma density is 0.08".

No language model writes any explanation. Two bugs were caught here in
development:

- Every `docmean_*` and `docstd_*` feature collapsed into one undifferentiated
  "document-wide style" bucket, destroying exactly the specificity the panel
  exists to provide.
- The explanation ranked families by magnitude alone, so a family opposing the
  verdict could be named both as what drove the score and as what pushed the
  other way — in the same sentence. The leading family is now chosen from those
  pointing toward the verdict.

---

## Missing values

A feature returns `None` when it is *undefined*, not when it is zero. Type-token
ratio on an empty sentence is undefined; comma density on a comma-free sentence
is genuinely 0.0. `None` becomes the training median plus a `__missing`
indicator, so the model can learn that missingness is informative.

This mattered concretely: an early MTLD implementation returned `None` for text
with no repeated word. Since such sentences are disproportionately human, the
missingness indicator would have carried label information the feature was
supposed to carry, inflating every downstream metric.
