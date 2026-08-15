"""Generate hybrid essays: human text with a fraction of sentences AI-rewritten.

Why this matters more than its size suggests
--------------------------------------------
Every sentence-level number reported so far is measured against labels
*inherited* from the document: a sentence in a machine essay is called machine,
a sentence in a human essay is called human. That is noisy by construction, and
it cannot measure the case the brief calls realistic — "a paragraph a person
wrote and a model later polished".

Hybrids fix that. Because we choose which sentences to rewrite, we know the
ground truth for every span. That makes the sentence-level scorer evaluable for
the first time, and it is the only honest way to answer "does the highlighting
actually point at the machine-written parts?"

Design choices
--------------
*Held-out prompts only.* Source essays come from validation and test prompts,
so no hybrid derives from anything the model trained on.

*Gemini rather than a Llama model.* Gemini is absent from DAIGT's seventeen
training generators, so this is a genuine unseen-model test. Using a Llama
endpoint would partly re-test models the classifier already learned.

*Varied instructions.* Five rewriting styles, cycled, so the hybrid set does
not become a study of one prompt's quirks — the same reasoning the brief
applies to generating machine essays.

*Scattered, not contiguous, rewrites.* Sentences are chosen throughout the
essay rather than as one block, because a model asked to "polish this essay"
touches sentences all through it.

*Resumable.* Free-tier rate limits mean this runs for a while; results are
appended to JSONL after every essay, so an interruption costs one request.

Usage:
    python3 scripts/generate_hybrids.py --count 200
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "backend"))

from app.nlp.preprocessing import normalize  # noqa: E402
from app.nlp.segmentation import segment  # noqa: E402

PROCESSED = REPOSITORY_ROOT / "data/processed"
OUTPUT_PATH = PROCESSED / "hybrid_essays.jsonl"

# gemini-2.5-flash is closed to new API keys. These are current and confirmed
# working on v1beta; the first that responds is used.
MODEL_CANDIDATES = [
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
    "gemini-2.5-flash-lite",
]
API_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

# Google returns HTTP 404 for temporary capacity problems as well as for models
# that genuinely do not exist — the body says "experiencing high demand" while
# the status code says not-found. Treating 404 as fatal would abandon a run
# over a transient spike, so it is retried like any other transient failure and
# only reported after the retries are exhausted.
_RETRYABLE_STATUS_CODES = frozenset({404, 429, 500, 502, 503, 504})

# Fraction of sentences rewritten per essay, drawn uniformly in this range.
# Spanning 30-60% keeps the set from being uniformly easy or uniformly hard.
MINIMUM_REWRITE_FRACTION = 0.30
MAXIMUM_REWRITE_FRACTION = 0.60

# Essays need enough sentences for a partial rewrite to be meaningful.
MINIMUM_SENTENCES = 8

# Free-tier pacing. Requests are spaced rather than fired in parallel.
SECONDS_BETWEEN_REQUESTS = 4.0
MAXIMUM_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 8.0

REQUEST_TIMEOUT_SECONDS = 120

# Five instruction variants. Deliberately phrased the way a student would ask,
# because that is the realistic threat model: not an adversary evading a
# detector, but someone running their own essay through a chatbot.
INSTRUCTION_VARIANTS: list[str] = [
    "Rewrite each of these sentences so they read more smoothly and "
    "professionally. Keep the same meaning and the same first-person voice.",
    "Polish these sentences. Fix awkward phrasing and improve the flow, but "
    "do not change what they say or add new information.",
    "Make these sentences clearer and better written. Preserve the original "
    "meaning, tone and point of view exactly.",
    "Improve the wording of these sentences so they sound more articulate. "
    "Do not add facts, and do not change the argument.",
    "Edit these sentences for style and readability. Keep every idea the "
    "same and keep them roughly the same length.",
]

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "rewrites": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["index", "text"],
            },
        }
    },
    "required": ["rewrites"],
}


def call_gemini(prompt: str, api_key: str, model: str) -> dict:
    """POST to the Gemini REST API with retry and backoff.

    Raw HTTP rather than an SDK: no new dependency, no disk cost on a machine
    with little to spare, and no wheel-availability risk on Python 3.14.
    """
    payload = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                # Varied sampling so the rewrites are not all in one register.
                "temperature": 0.9,
                "responseMimeType": "application/json",
                "responseSchema": _RESPONSE_SCHEMA,
            },
        }
    ).encode("utf-8")

    url = f"{API_ENDPOINT.format(model=model)}?key={api_key}"
    backoff = INITIAL_BACKOFF_SECONDS

    for attempt in range(1, MAXIMUM_RETRIES + 1):
        request = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(
                request, timeout=REQUEST_TIMEOUT_SECONDS
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code not in _RETRYABLE_STATUS_CODES or attempt == MAXIMUM_RETRIES:
                detail = error.read().decode("utf-8", errors="replace")[:300]
                raise RuntimeError(f"Gemini HTTP {error.code}: {detail}") from error
            print(f"    HTTP {error.code}; waiting {backoff:.0f}s", flush=True)
            time.sleep(backoff)
            backoff *= 2
        except urllib.error.URLError as error:
            if attempt == MAXIMUM_RETRIES:
                raise RuntimeError(f"Network error: {error.reason}") from error
            time.sleep(backoff)
            backoff *= 2
        except OSError as error:
            # ConnectionResetError and friends surface as bare OSError from the
            # ssl layer rather than being wrapped in URLError, so they slipped
            # past the handler above and killed a long run at essay 44. They
            # are as transient as any 503 and are retried the same way.
            if attempt == MAXIMUM_RETRIES:
                raise RuntimeError(f"Connection error: {error}") from error
            print(f"    {type(error).__name__}; waiting {backoff:.0f}s", flush=True)
            time.sleep(backoff)
            backoff *= 2

    raise RuntimeError("Exhausted retries.")


def select_model(api_key: str) -> str:
    """Return the first candidate model that answers a trivial request."""
    for model in MODEL_CANDIDATES:
        try:
            call_gemini("Reply with exactly: ok", api_key, model)
        except (RuntimeError, ValueError, json.JSONDecodeError) as error:
            print(f"  {model} unavailable ({str(error)[:80]})", flush=True)
            continue
        print(f"  using {model}", flush=True)
        return model
    raise SystemExit(
        "No Gemini model responded. All candidates returned errors; the API may "
        "be under load. Try again shortly."
    )


def extract_rewrites(response: dict) -> dict[int, str]:
    """Pull the index-to-text mapping out of a Gemini response."""
    try:
        candidates = response["candidates"]
        text = candidates[0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as error:
        raise ValueError(f"Unexpected response shape: {str(response)[:200]}") from error

    parsed = json.loads(text)
    return {int(item["index"]): str(item["text"]).strip() for item in parsed["rewrites"]}


def build_prompt(
    sentences: list[str], selected: list[int], instruction: str
) -> str:
    """Assemble the rewriting request, numbered so replies can be matched up."""
    numbered = "\n".join(f"{index}. {sentences[index]}" for index in selected)
    return (
        f"{instruction}\n\n"
        "Return one rewrite for every numbered sentence, using the same index "
        "numbers. Return only the sentences themselves, with no numbering "
        "inside the text and no commentary.\n\n"
        f"{numbered}"
    )


def choose_sentences(count: int, generator: random.Random) -> list[int]:
    """Pick a scattered subset of sentence indices to rewrite."""
    fraction = generator.uniform(MINIMUM_REWRITE_FRACTION, MAXIMUM_REWRITE_FRACTION)
    how_many = max(1, round(count * fraction))
    return sorted(generator.sample(range(count), how_many))


def load_completed_ids() -> set[str]:
    """Source essay ids already generated, so a rerun resumes."""
    if not OUTPUT_PATH.exists():
        return set()
    completed: set[str] = set()
    with OUTPUT_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                completed.add(json.loads(line)["source_essay_id"])
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260814)
    arguments = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY is not set in this shell. It lives in ~/.zshrc, which "
            "zsh reads only for interactive shells; run this with "
            "`source ~/.zshrc && python3 scripts/generate_hybrids.py`."
        )

    print("Selecting a Gemini model...", flush=True)
    model = select_model(api_key)

    corpus = pd.read_csv(PROCESSED / "corpus.csv")
    # Held-out prompts only: nothing here may derive from a training prompt.
    candidates = corpus[
        (corpus["label"] == 0) & (corpus["split"].isin(["validation", "test"]))
    ]

    completed = load_completed_ids()
    if completed:
        print(f"Resuming: {len(completed):,} essays already generated.", flush=True)
        candidates = candidates[~candidates["essay_id"].isin(completed)]

    generator = random.Random(arguments.seed)
    remaining = arguments.count - len(completed)
    if remaining <= 0:
        print("Requested count already generated. Nothing to do.")
        return

    candidates = candidates.sample(
        min(remaining * 2, len(candidates)), random_state=arguments.seed
    )

    written = 0
    skipped = 0
    failed = 0

    with OUTPUT_PATH.open("a", encoding="utf-8") as handle:
        for _, record in candidates.iterrows():
            if written >= remaining:
                break

            normalized = normalize(str(record["text"]))
            document = segment(normalized)
            sentences = document.sentences
            if len(sentences) < MINIMUM_SENTENCES:
                skipped += 1
                continue

            texts = [sentence.text for sentence in sentences]
            selected = choose_sentences(len(texts), generator)
            variant_index = (written + len(completed)) % len(INSTRUCTION_VARIANTS)
            instruction = INSTRUCTION_VARIANTS[variant_index]

            try:
                response = call_gemini(
                    build_prompt(texts, selected, instruction), api_key, model
                )
                rewrites = extract_rewrites(response)
            except (RuntimeError, ValueError, json.JSONDecodeError) as error:
                print(f"  [{record['essay_id']}] failed: {error}", flush=True)
                failed += 1
                time.sleep(SECONDS_BETWEEN_REQUESTS)
                continue

            # Keep only rewrites that were asked for, are non-empty, and
            # actually changed something. A model that echoes the input would
            # otherwise silently produce hybrids with no machine text in them,
            # and the span labels would be wrong in the worst possible way.
            usable = {
                index: text
                for index, text in rewrites.items()
                if index in set(selected) and text and text != texts[index]
            }
            if not usable:
                print(f"  [{record['essay_id']}] no usable rewrites", flush=True)
                failed += 1
                time.sleep(SECONDS_BETWEEN_REQUESTS)
                continue

            # Rebuild the essay, recording the span of every sentence in the
            # new text so span labels are exact rather than re-derived later.
            pieces: list[str] = []
            spans: list[dict[str, object]] = []
            cursor = 0
            for index, original in enumerate(texts):
                text = usable.get(index, original)
                spans.append(
                    {
                        "sentence_index": index,
                        "start": cursor,
                        "end": cursor + len(text),
                        "is_rewritten": index in usable,
                    }
                )
                pieces.append(text)
                cursor += len(text) + 1  # the joining space

            hybrid_text = " ".join(pieces)

            handle.write(
                json.dumps(
                    {
                        "source_essay_id": record["essay_id"],
                        "prompt_name": record["prompt_name"],
                        "split": record["split"],
                        "ell_status": record.get("ell_status")
                        if pd.notna(record.get("ell_status"))
                        else None,
                        "model": model,
                        "instruction_variant": variant_index,
                        "original_text": normalized,
                        "hybrid_text": hybrid_text,
                        "sentence_count": len(texts),
                        "rewritten_count": len(usable),
                        "rewritten_fraction": len(usable) / len(texts),
                        "spans": spans,
                    }
                )
                + "\n"
            )
            handle.flush()

            written += 1
            if written % 10 == 0:
                print(
                    f"  {written}/{remaining} generated "
                    f"({failed} failed, {skipped} too short)",
                    flush=True,
                )

            time.sleep(SECONDS_BETWEEN_REQUESTS)

    print(
        f"\nGenerated {written} hybrid essays "
        f"({failed} failed, {skipped} skipped as too short)."
    )
    print(f"Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
