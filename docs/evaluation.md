# Evaluation

All numbers below come from `scripts/evaluate.py` and
`scripts/evaluate_hybrids.py`, run against the frozen split
`8599bccd1d049678b048f879b0dcdfad96286bde4c3a1411a697675915225622`. Raw output
is in `data/models/baseline/evaluation.json` and `hybrid_evaluation.json`.

Nothing here was tuned against the test split. The operating threshold is 0.5
throughout.

---

## Document-level test performance

963 essays across 2 held-out prompts.

| Metric | Value |
|---|---|
| Accuracy | 0.9512 |
| Precision | 0.9506 |
| Recall | 0.9395 |
| F1 | 0.9450 |
| ROC AUC | 0.9886 |
| False positive rate | 0.0394 |
| False negative rate | 0.0605 |
| Brier score | 0.0397 |
| Expected calibration error | 0.0141 |

**Confusion matrix**

|  | Predicted human | Predicted machine |
|---|---|---|
| **Actually human** | 512 | 21 |
| **Actually machine** | 26 | 404 |

## Cross-validation

The test split contains only two prompts, so a single accuracy from it could be
lucky. Prompt-grouped 5-fold CV over the whole corpus, refitting from scratch
each fold:

| | |
|---|---|
| Document AUC | **0.9841 ± 0.0044** |
| Document accuracy | 0.9318 ± 0.0148 |
| Per fold | 0.980, 0.978, 0.989, 0.986, 0.987 |

The test number is not an outlier.

## Sentence-level

AUC **0.8641** on test — but measured against labels *inherited* from the
document, which is noisy by construction. The honest sentence-level measurement
is the hybrid span evaluation below.

---

## Per-generator detection

Averages hide which models evade the detector.

| Generator | n | Recall |
|---|---|---|
| llama_70b_v1 | 60 | **0.733** |
| mistral7binstruct_v2 | 32 | 0.875 |
| cohere-command | 28 | 0.929 |
| llama2_chat | 45 | 0.956 |
| falcon_180b_v1 | 54 | 0.963 |
| chat_gpt_moth | 34 | 1.000 |
| darragh_claude_v6 | 61 | 1.000 |
| darragh_claude_v7 | 57 | 1.000 |
| mistral7binstruct_v1 | 32 | 1.000 |
| palm-text-bison1 | 27 | 1.000 |
| *persuade_corpus (human)* | 533 | *FPR 0.039* |

A 27-point spread. Reporting only the 0.94 average would conceal that
Llama-70B slips past a quarter of the time while Claude never does.

---

## Hybrid essays — the realistic case

200 human essays with 30–60% of sentences rewritten by Gemini
(`gemini-3.1-flash-lite`), drawn from held-out prompts. Gemini is absent from
the 17 training generators, so this is also an unseen-model test.

Sentence boundaries are re-derived from scratch, and each detected sentence is
labelled machine if more than half its characters fall inside a rewritten span.

### Span-level: does the highlighting find the rewrites?

| | |
|---|---|
| Scorable sentences | 3,638 |
| Actually rewritten | 1,647 |
| **ROC AUC** | **0.7546** |
| Precision | 0.6650 |
| Recall | 0.6400 |
| F1 | 0.6522 |
| False positive rate | 0.2667 |

The highlighting is meaningfully better than chance and far from reliable. It
finds about 64% of AI-rewritten sentences while falsely flagging 27% of
human-written ones.

### Document-level on hybrids

| | |
|---|---|
| Essays | 200 |
| **Flagged as machine** | **0.255** |
| Mean risk | 0.267 |

**Three quarters of partly-AI essays are not detected.** Detection rises with
how much was rewritten, but only weakly:

| Fraction rewritten | n | Detected | Mean risk |
|---|---|---|---|
| ≤ 35% | 45 | 0.178 | 0.186 |
| 35–45% | 71 | 0.268 | 0.284 |
| 45–55% | 59 | 0.271 | 0.283 |
| > 55% | 25 | 0.320 | 0.324 |

This is the most important result in the project. The detector is strong on
fully machine-generated essays and weak on the case that actually happens.

---

## ESL / non-native English

609 ELL essays and 609 prompt-matched non-ELL essays, all human, all from
held-out prompts. Prompt matching matters: ELL and non-ELL essays are not evenly
distributed across prompts, so an unmatched comparison would partly measure
topic difficulty.

| Group | n | False positives | FPR | 95% CI (Wilson) |
|---|---|---|---|---|
| ELL writers | 609 | 29 | 0.0476 | [0.0334, 0.0676] |
| non-ELL writers | 609 | 40 | 0.0657 | [0.0486, 0.0882] |

Difference **−0.018**, confidence intervals **overlap**. No detectable bias
against English language learners in this population; directionally ELL writers
are flagged slightly less.

Wilson intervals rather than the normal approximation, which misbehaves near 0
on samples this size.

**Do not over-read this.** See [limitations.md](limitations.md#the-esl-result-is-narrower-than-it-looks).

---

## Three confidently wrong predictions

Selected programmatically as the misclassified test essays furthest from the
decision boundary. Contributions are the actual log-odds decomposition.

### Case 1 — human scored as machine

| | |
|---|---|
| Essay | `2c7e6148db8951e7` |
| Actual | **human** (persuade_corpus) |
| Predicted | **machine**, P = 0.996 |
| Size | 266 words, 17 sentences |

```
+3.293  docmean_punct_comma_density
+2.309  sentscore_q25
+1.671  sentscore_mean
-1.121  docstd_punct_comma_density
```

> "Dear Principal, I am aware that you might have a good reason for not
> allowing students with at least a B grade average to participate in school
> sporting…"

**Why it failed.** Comma density alone contributed +3.29 log-odds — enough to
decide the essay by itself. This student writes long, comma-spliced,
formally-addressed sentences. The model has learned that heavy comma use plus
formal register means machine, because in the training data it usually does.
This is the sophistication confound made concrete: a capable human writer moves
toward the machine end of the strongest features.

### Case 2 — machine scored as human

| | |
|---|---|
| Essay | `0bbe654933c965a9` |
| Actual | **machine** (llama_70b_v1) |
| Predicted | **human**, P = 0.006 |
| Size | 280 words, 24 sentences |

```
+2.422  sentscore_std
-1.883  docmean_syn_mean_dependency_depth
-1.299  sentscore_range
-1.207  sentscore_q10
```

> "I think gun controll is a really importint issue that we should take
> seriusly. Some peopel say we need stricter gun laws to keep peopel saif…"

**Why it failed.** The generator was prompted to imitate a student and produced
deliberate misspellings — *controll*, *importint*, *peopel*, *saif* — along
with flat, shallow syntax. Every feature the detector relies on says human:
low dependency depth, simple vocabulary, error-strewn surface. This is the
clearest evidence that the detector is measuring **polish**, not authorship. A
model told to write badly defeats it completely.

Note the internal disagreement: `sentscore_std` pushed +2.42 toward machine and
was outvoted. The signal was there and the aggregate buried it.

### Case 3 — machine scored as human

| | |
|---|---|
| Essay | `58bb93068c803e0e` |
| Actual | **machine** (cohere-command) |
| Predicted | **human**, P = 0.008 |
| Size | 558 words, 27 sentences |

```
-1.463  sentscore_mean
-1.367  sentscore_q25
-1.237  sentscore_q10
-1.187  sentscore_range
```

> "Driverless cars are becoming more and more of a reality. The development of
> this technology is offers a world of possibility, but there are safety and…"

**Why it failed.** Note "is offers" — a genuine grammatical error in generated
text. Cohere's output here is longer and less uniform than the ChatGPT and
Claude essays the model saw most of; every sentence-score quantile reads human.
Unlike Case 2 there was no dissenting signal at all, which makes this the more
worrying failure: the model was uniformly, quietly wrong.

---

## Feature separation

Measured before any model was fit, on 480 balanced essays sampled *within*
prompt so a feature cannot score well by detecting topic. Standalone ROC AUC:

| Feature | AUC | Human | Machine |
|---|---|---|---|
| `docmean_lex_stopword_ratio` | 0.836 ↓ | 0.526 | 0.440 |
| `docmean_punct_comma_density` | 0.776 | 0.029 | 0.048 |
| `docmean_form_nominalization_density` | 0.764 | 0.017 | 0.035 |
| `docstd_lex_mtld` | 0.753 ↓ | 8.09 | 5.41 |
| `rhythm_length_std` | 0.732 ↓ | **9.35** | **6.61** |
| `rhythm_burstiness` | 0.698 ↓ | −0.374 | −0.464 |

(↓ = predictive in the inverse direction.)

Machine sentence-length standard deviation is 6.61 against humans' 9.35 —
"machine prose has more even sentence rhythms" as a measured quantity.

One finding contradicts the folk wisdom: machine essays have *higher*
document-level lexical diversity than human ones (MTLD 91.0 vs 69.3). The true
pattern is subtler — machine text is *uniformly* diverse (high mean, low
variance) while humans swing between rich and plain sentences. Only measuring
mean and spread separately reveals it.
