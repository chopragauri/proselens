# Limitations

ProseLens measures statistical properties of writing. It cannot establish who
wrote a text, and it should never be the sole basis for an accusation.

Everything below is a measured limitation or an explicitly untested case. None
of it is hypothetical caution.

---

## 1. Partial AI editing — the largest gap

**Measured: 25.5% of hybrid essays detected.**

Human essays with 30–60% of sentences rewritten by a model are flagged a
quarter of the time. Span-level AUC is 0.7546, precision 0.665, recall 0.640.

This is the realistic scenario — a student who writes their own essay and runs
it through a chatbot to polish it — and it is where the detector is weakest.
The document score is dominated by essay-wide style statistics, and an essay
that is 60% human retains enough human style to look human.

Detection barely improves with the amount rewritten (17.8% at ≤35% → 32.0% at
>55%), so this is not a threshold that can be tuned away.

## 2. The detector measures polish, not authorship

The strongest features — stopword ratio, comma density, nominalization density,
dependency depth — are markers of *writing sophistication*. Human training data
is grades 6–12; machine data is fluent LLM output. The model has substantially
learned to separate polished prose from unpolished prose.

Two consequences, both observed:

- **A capable human writer is flagged.** Failure Case 1: a human essay scored
  0.996 machine, with comma density alone contributing +3.29 log-odds.
- **A model told to write badly escapes.** Failure Case 2: a Llama-70B essay
  with deliberate misspellings ("controll", "importint", "peopel") scored 0.006
  — confidently human.

Anyone able to prompt "write this like a rushed student" defeats the detector.

## 3. Short text

Below five scorable sentences the false-positive rate on human writing reaches
**0.695**. The system refuses to give a verdict there, and withholds a
"likely machine" verdict below twelve sentences:

| Sentences | 3 | 5 | 8 | 12 | 16 | 25 |
|---|---|---|---|---|---|---|
| FPR | 0.695 | 0.475 | 0.220 | 0.075 | 0.025 | 0.000 |

Recall stays above 0.95 throughout, so the failure is one-sided: short text
produces false accusations, not missed detections.

## 4. Domain shift — training data is not admissions essays

The corpus is **argumentative student essays** on prompts like "Car-free
cities" and "Does the electoral college work?". The product is aimed at
**admissions personal statements**: first-person narrative, heavily revised,
often workshopped by teachers and counsellors.

Test performance is measured on held-out *prompts within the same genre*, not
on a different genre. The 0.951 accuracy has not been demonstrated on a single
real admissions essay, because no such labelled corpus was available. Expect it
to be lower — admissions essays are far more polished than 8th-grade homework,
which by §2 pushes them toward the machine end.

## 5. The ESL result is narrower than it looks

The measurement is real: ELL FPR 0.0476 versus non-ELL 0.0657, intervals
overlapping, on 609 prompt-matched essays each. There is no detectable bias
against English language learners **in this population**.

That population is grade 6–12 students in US schools. Their ELL writing is less
polished than their non-ELL peers', which by §2 pushes it *away* from the
machine end. The favourable result may be an artifact of that.

**Untested and genuinely concerning:** an adult ESL writer — for example an
international applicant who has been explicitly taught formal academic register,
connectives, and nominalization — writes in exactly the style the detector
associates with machines. This is the population most likely to be harmed and
the one this evaluation says nothing about.

Testing it needs a corpus of adult L2 academic writing with authorship labels
(TOEFL11 or ICNALE, both access-restricted). Until then, this limitation is
open.

## 6. Unseen generators

Recall by generator on the test set spans 0.733 (Llama-70B) to 1.000 (Claude,
ChatGPT, PaLM). Detection of a model family not in training is unknown; the one
data point is Gemini via the hybrid set, where document detection was poor —
though that is confounded with partial rewriting.

Newer models produce less formulaic prose than the 2023-era generators in DAIGT.
Performance on current frontier models is likely worse than reported here.

## 7. Adversarial rewriting

Untested and expected to be weak. The features are public, documented in this
repository, and individually easy to manipulate: vary sentence length, remove
commas, avoid nominalizations, insert errors. Failure Case 2 shows this works
even without adversarial intent.

ProseLens is designed for the non-adversarial case — someone who used a chatbot
without trying to hide it.

## 8. Predictability measures domain fit as well as fluency

The reference trigram model is fit on our own human training corpus, so a
sentence on an unusual topic scores as "unpredictable" regardless of who wrote
it. This is why predictability is one input among many and never the verdict.

An earlier version of this feature was computed in-sample and actively harmed
the model — validation AUC 0.801 with it, 0.951 without. It is now computed
out-of-fold. See [methodology.md](methodology.md#two-leaks-found-and-fixed).

## 9. Multilingual and code-switched writing

Untested. The spaCy pipeline, stopword lists, and reference language model are
all English. Text containing substantial non-English content will produce
unreliable features rather than an error.

## 10. Dataset bias

- Human essays are US students, grades 6–12, on 15 prompts. Not representative
  of applicant writing generally.
- Machine essays are 2023-era models.
- The test split is **2 prompts**. Cross-validation (AUC 0.9841 ± 0.0044)
  bounds the variance but cannot replace topic diversity.
- Hybrids come from a single generator.

## 11. Calibration is not guaranteed off-distribution

Test ECE is 0.0141, but the mid-range bins are thin — the [0.6, 0.7) bin
predicts 0.662 against an observed 0.364 on eleven essays. The UI reports a
risk score rather than a probability for this reason.

---

## How this should be used

As an instrument that directs attention, not one that produces verdicts. A high
score means "these passages have measurable properties common in
machine-generated text" — a reason to read more closely or start a conversation.

It is not evidence of misconduct. Given §2 and §5, using it to make
consequential decisions about individuals would systematically disadvantage
strong writers and, plausibly, second-language writers whose training in formal
register the detector cannot distinguish from a machine's.
