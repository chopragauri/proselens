"""Shared spaCy pipeline.

Loaded once per process and cached. The API must never construct a pipeline
per request: `en_core_web_sm` takes roughly 0.3 s to load, which would dominate
analysis latency and would also mean holding several copies in memory under
concurrency.
"""

from __future__ import annotations

from functools import lru_cache

import spacy
from spacy.language import Language

# Named entity recognition costs time and contributes nothing to our features,
# which are lexical, syntactic and distributional. The parser is required: it
# supplies both sentence boundaries and the dependency tree used by the syntax
# features.
_DISABLED_COMPONENTS = ["ner"]

SPACY_MODEL_NAME = "en_core_web_sm"


@lru_cache(maxsize=1)
def get_nlp() -> Language:
    """Return the process-wide spaCy pipeline."""
    return spacy.load(SPACY_MODEL_NAME, disable=_DISABLED_COMPONENTS)
