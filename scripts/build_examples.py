"""Build the demo essays in `examples/`.

Provenance matters here. The corpora this project trains on (DAIGT v2,
PERSUADE 2.0) carry redistribution terms, which is why `data/raw/` is
gitignored — so the demo set cannot simply be a handful of corpus rows.

Instead:

* `human-written.txt` is original prose authored for this repository.
* `machine-generated.txt` is produced fresh by Gemini here, so it is model
  output we generated and may redistribute.
* `hybrid-edited.txt` takes the human essay and has Gemini rewrite a subset of
  its sentences, recording exactly which — the same construction used for the
  span-level evaluation.
* `short-text.txt` is a trimmed excerpt, included to demonstrate the length
  gate rather than detection.

These are illustrations for trying the interface. They are **not** evaluation
data, they are not in any split, and no metric in `docs/evaluation.md` is
computed from them.

Usage:
    source ~/.zshrc && /opt/homebrew/bin/python3 scripts/build_examples.py
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "backend"))

from app.nlp.preprocessing import normalize  # noqa: E402
from app.nlp.segmentation import segment  # noqa: E402

# Reuse the generation client so the demo hybrid is built exactly the way the
# evaluation hybrids were.
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
from generate_hybrids import (  # noqa: E402
    build_prompt,
    call_gemini,
    extract_rewrites,
    select_model,
)

EXAMPLES = REPOSITORY_ROOT / "examples"

MACHINE_PROMPT = (
    "Write a 450-word college admissions personal statement about a student "
    "who volunteered as a tutor and learned the value of empathy. Write it in "
    "the first person. Do not use headings. Return it as JSON in the form "
    '{"rewrites": [{"index": 0, "text": "<the full essay>"}]}.'
)

REWRITE_FRACTION = 0.45
SEED = 20260815


def write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    print(f"  wrote {path.relative_to(REPOSITORY_ROOT)} ({len(text.split())} words)")


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY is not set. It lives in ~/.zshrc, which zsh reads "
            "only for interactive shells; run with "
            "`source ~/.zshrc && /opt/homebrew/bin/python3 scripts/build_examples.py`."
        )

    EXAMPLES.mkdir(exist_ok=True)
    human_path = EXAMPLES / "human-written.txt"
    if not human_path.exists():
        raise SystemExit(f"Missing {human_path}; it is authored, not generated.")

    print("Selecting a Gemini model...", flush=True)
    model = select_model(api_key)

    # ---- machine-generated ----
    print("Generating the machine essay...", flush=True)
    response = call_gemini(MACHINE_PROMPT, api_key, model)
    essay = extract_rewrites(response)[0]
    write(EXAMPLES / "machine-generated.txt", essay)

    # ---- hybrid ----
    print("Building the hybrid essay...", flush=True)
    human_text = normalize(human_path.read_text(encoding="utf-8"))
    sentences = [sentence.text for sentence in segment(human_text).sentences]

    generator = random.Random(SEED)
    chosen = sorted(
        generator.sample(
            range(len(sentences)), max(1, round(len(sentences) * REWRITE_FRACTION))
        )
    )
    instruction = (
        "Polish these sentences. Fix awkward phrasing and improve the flow, but "
        "do not change what they say or add new information."
    )
    rewrites = extract_rewrites(
        call_gemini(build_prompt(sentences, chosen, instruction), api_key, model)
    )
    usable = {
        index: text
        for index, text in rewrites.items()
        if index in set(chosen) and text and text != sentences[index]
    }

    pieces = [usable.get(index, text) for index, text in enumerate(sentences)]
    write(EXAMPLES / "hybrid-edited.txt", " ".join(pieces))

    # The span record is what makes this example checkable rather than a claim.
    (EXAMPLES / "hybrid-edited.spans.json").write_text(
        json.dumps(
            {
                "source": "examples/human-written.txt",
                "model": model,
                "rewritten_sentence_indices": sorted(usable),
                "sentence_count": len(sentences),
                "rewritten_count": len(usable),
                "rewritten_fraction": round(len(usable) / len(sentences), 3),
                "sentences": [
                    {
                        "index": index,
                        "is_rewritten": index in usable,
                        "original": sentences[index],
                        "text": pieces[index],
                    }
                    for index in range(len(sentences))
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("  wrote examples/hybrid-edited.spans.json")

    # ---- short text, for the length gate ----
    short = " ".join(sentences[:4])
    write(EXAMPLES / "short-text.txt", short)

    print("\nDone.")


if __name__ == "__main__":
    main()
