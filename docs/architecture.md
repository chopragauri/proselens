# Architecture

## Pipeline

```
essay text
  │
  ├─ preprocessing/normalize        length-preserving character substitution
  ├─ segmentation/segment           paragraphs → sentences, exact char offsets
  ├─ features/aggregate             8 groups × 3 scopes, one batched spaCy pass
  │    ├─ sentence scope            computable from one sentence alone
  │    ├─ contextual scope          sentence relative to its document
  │    └─ document scope            rhythm, whole-essay lexis, summaries
  ├─ ml/detector                    sentence model → per-sentence risk
  │                                 document model → document risk
  ├─ services/evidence              log-odds decomposition → signal families
  └─ api/routes                     FastAPI → React
```

## Layout

```
backend/app/
  core/config.py            env-driven settings, repo-root-relative paths
  nlp/
    preprocessing.py        normalization (length-preserving by construction)
    segmentation.py         paragraph/sentence spans with offsets
    pipeline.py             the one shared spaCy pipeline (lru_cache)
    features/
      base.py               FeatureDict, FeatureScope, word_forms, safe_divide
      lexical.py            TTR, MTLD, hapax, stopword ratio
      syntax.py             POS ratios, dependency depth, clause complexity
      repetition.py         n-gram and POS-template echo
      punctuation.py        per-token punctuation densities
      structure.py          length, burstiness, rhythm
      formulaic.py          morphosyntactic formulaicity
      predictability.py     trigram LM, cross-entropy
      aggregate.py          composition and scope discipline
  data/
    deduplication.py        MinHash + LSH near-duplicate detection
    splitting.py            grouped split, validation, manifest, hashing
  ml/
    feature_matrix.py       column selection, imputation, standardization
    detector.py             the two models, evidence decomposition, JSON I/O
    evaluation.py           metrics shared by training and evaluation
  services/
    analysis.py             the request path; models loaded once
    evidence.py             contributions → signal families → explanations
  schemas/analysis.py       Pydantic request/response models
  api/routes.py             endpoints, upload validation
  main.py                   app, CORS, startup preload

frontend/src/
  App.tsx                   state and orchestration
  components/               EssayInput, OverallAssessment, HighlightedEssay,
                            EvidencePanel, AnalyticsPanel
  services/api.ts           fetch wrapper surfacing backend detail messages
  types/analysis.ts         mirrors the Pydantic schemas
  utils/risk.ts             band → colour, underline, marker, label
```

## Load-bearing decisions

### Length-preserving normalization

Every substitution maps exactly one character to one character, so a character
offset computed anywhere downstream is valid against the raw text the user
pasted. The frontend slices the original string by those offsets to render
highlights.

The alternative — collapsing whitespace and maintaining an offset mapping
table — is standard, but any bug in the table silently mis-highlights sentences.
Forbidding length changes deletes that class of bug;
`test_normalize_preserves_length` enforces it.

This caught a real bug: mapping `\r → \n` turns every Windows CRLF into a blank
line, which paragraph splitting reads as a paragraph break, silently doubling
the paragraph count of any essay pasted from a Windows editor.

### Feature scope discipline

`FeatureScope` distinguishes what a feature is computed over. Burstiness is
undefined for a single sentence, so a sentence classifier must not receive it —
it would get a number, but a meaningless one, and the evidence panel would then
explain noise. `aggregate.py` enforces the separation and
`test_sentence_vectors_exclude_document_scoped_features` pins it.

### One spaCy pipeline, batched

`get_nlp()` is `lru_cache`d; NER is disabled (unused), the parser is kept
(supplies both sentence boundaries and dependency features). Paragraphs and
sentences go through `nlp.pipe()` in single batched passes rather than
per-item calls.

### Models loaded once

`get_analysis_service()` is `lru_cache`d and preloaded in the FastAPI lifespan
hook. The spaCy pipeline takes ~0.3 s to construct and the reference language
model is 12 MB of JSON; paying that per request would dominate latency and
multiply memory under concurrency.

Preload failure is logged and tolerated — `/api/health` then reports
`degraded`, which is more useful than a server that refuses to start.

### JSON artifacts, never pickle

Model coefficients, scalers and the language model all serialize to JSON.
Unpickling is arbitrary code execution, and these files are loaded by a web
service at startup. `test_saved_artifacts_contain_no_pickle` asserts the
artifact directory contains only `.json`.

### Deterministic hashing

Deduplication uses CRC32 rather than Python's built-in `hash`, which is
randomized per process by `PYTHONHASHSEED`. A split that changed between runs
would make "reproducible pipeline" a false claim.

MinHash reduces ~45,000 essays from a quadratic pairwise problem to a
near-linear one; the permutation step is vectorized in int64 with a 2³¹−1
modulus so products cannot overflow.

## Request path

1. `POST /api/analyze` → Pydantic validation (length, non-whitespace).
2. Normalize → segment → extract features (one batched spaCy pass).
3. Score scorable sentences; unscorable ones are returned but never scored.
4. Build the document frame (document features + sentence-score summaries) and
   score it.
5. Decompose both models' log-odds into signal families with z-scores against
   the human reference distribution.
6. Assemble sentences, passages, analytics; apply length gates.

Uploads decode in memory, are size-capped while reading, are strictly UTF-8
validated, and join the same path — no second code path to drift.

## Security

- Extension allow-list (`.txt`), byte cap enforced during read, character cap
  after decode.
- Strict UTF-8 decoding rather than `errors="replace"`, so a non-text file is
  rejected instead of analysed as mojibake.
- Nothing written to disk; no filesystem paths derived from user input.
- Explicit CORS origin list, not a wildcard.
- No secrets in the repository; the Gemini key used for hybrid generation is
  read from the environment only.
