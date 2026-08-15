# Demo essays

Four essays for trying the interface, chosen to show what ProseLens does well
**and** where it fails.

> These are illustrations, not evaluation data. None is in any split, and no
> metric in [../docs/evaluation.md](../docs/evaluation.md) is computed from
> them. They exist so you can click through the UI in two minutes.

## Provenance

The corpora this project trains on (DAIGT v2, PERSUADE 2.0) carry
redistribution terms, which is why `data/raw/` is gitignored. So none of these
files is a corpus row.

| File | Origin |
|---|---|
| `human-written.txt` | Original prose, authored for this repository |
| `machine-generated.txt` | Generated fresh by `gemini-3.1-flash-lite` |
| `hybrid-edited.txt` | `human-written.txt` with 44% of sentences rewritten by Gemini |
| `short-text.txt` | First four sentences of `human-written.txt` |

Regenerate with `scripts/build_examples.py`. The hybrid's exact span labels —
which sentence was rewritten, and its original wording — are recorded in
`hybrid-edited.spans.json`, so the claim is checkable rather than asserted.

## What each one shows

Run against `data/models/baseline`, threshold 0.5:

| Essay | Risk | Confidence | Assessment | Sentences (H/M/L/unscored) |
|---|---|---|---|---|
| `human-written.txt` | **1** | 52% | Consistent with human writing | 36 (4/13/16/3) |
| `machine-generated.txt` | **100** | 47% | Machine-like patterns | 28 (16/6/3/3) |
| `hybrid-edited.txt` | **5** | 54% | Consistent with human writing | 37 (5/15/14/3) |
| `short-text.txt` | 99 | 14% | **Not enough text to assess** | 4 (2/1/0/1) |

### 1. `human-written.txt` — the easy case

Risk 1/100. Concrete detail, uneven sentence rhythm, first-person specificity.

### 2. `machine-generated.txt` — the other easy case

Risk 100/100, 16 of 28 sentences high-risk. Click any of them: the evidence
panel shows comma density, nominalization and formulaic phrasing all well above
the human average, with the log-odds each contributed.

### 3. `hybrid-edited.txt` — **the detector fails this one**

44% of its sentences were rewritten by a model, and ProseLens scores it **5/100
and calls it human**.

This is the most useful file in the directory. It is the documented weakness
reproduced on demand: on 200 such essays, only **25.5%** are flagged, with
span-level AUC 0.755. An essay that is still 56% human retains enough human
style to read as human overall.

The sentence highlighting does better than the document score — 5 high and 15
medium-risk sentences against 16 actually rewritten — which is why the tool
shows passage-level evidence rather than a single verdict. Compare the
highlighting against `hybrid-edited.spans.json` to see what it caught and
missed. A representative rewrite:

> **Original:** My grandmother could not read the bus timetable.
> **Rewritten:** My grandmother was unable to read the bus timetable.

### 4. `short-text.txt` — the length gate

Risk 99/100, but the assessment is **"Not enough text to assess"** and
confidence is 14%.

Below five scorable sentences the measured false-positive rate on human writing
is 0.695, so no verdict is offered. This text is human — it is the opening of
`human-written.txt` — and the raw score is wrong. The gate is what stops that
becoming an accusation.

## Trying them

Paste the contents into the editor, or use **Upload .txt**. Then click any
highlighted sentence to open its evidence panel.
