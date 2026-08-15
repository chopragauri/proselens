"""Application settings.

Paths are resolved relative to the repository root rather than the working
directory, so the API behaves the same whether it is started from the repo
root, from `backend/`, or by a process manager with its own cwd. Every value
can be overridden by an environment variable, so nothing is hard-coded to one
machine and no path needs editing to deploy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

__all__ = ["Settings", "get_settings", "REPOSITORY_ROOT"]

# config.py -> core -> app -> backend -> repository root
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

# Comma-separated origins allowed to call the API. The default covers the Vite
# dev server only; a deployment must set PROSELENS_CORS_ORIGINS explicitly
# rather than inheriting a permissive default.
DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"

# Upload limit. Matches MAXIMUM_CHARACTERS in the schema closely enough that a
# file which passes this check will usually pass validation too, while still
# bounding what a single request can force the server to read.
MAXIMUM_UPLOAD_BYTES = 1_000_000

ALLOWED_UPLOAD_EXTENSIONS = frozenset({".txt"})


@dataclass(frozen=True)
class Settings:
    """Resolved configuration for one process."""

    model_directory: Path
    language_model_path: Path
    cors_origins: tuple[str, ...]
    maximum_upload_bytes: int
    allowed_upload_extensions: frozenset[str]

    @property
    def models_are_available(self) -> bool:
        return (
            (self.model_directory / "document_model.json").exists()
            and self.language_model_path.exists()
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build settings from the environment, cached for the process lifetime."""
    model_directory = Path(
        os.environ.get(
            "PROSELENS_MODEL_DIR", REPOSITORY_ROOT / "data/models/baseline"
        )
    )
    language_model_path = Path(
        os.environ.get(
            "PROSELENS_LANGUAGE_MODEL",
            REPOSITORY_ROOT / "data/models/reference_ngram_lm.json",
        )
    )
    origins = os.environ.get("PROSELENS_CORS_ORIGINS", DEFAULT_CORS_ORIGINS)

    return Settings(
        model_directory=model_directory,
        language_model_path=language_model_path,
        cors_origins=tuple(
            origin.strip() for origin in origins.split(",") if origin.strip()
        ),
        maximum_upload_bytes=int(
            os.environ.get("PROSELENS_MAX_UPLOAD_BYTES", MAXIMUM_UPLOAD_BYTES)
        ),
        allowed_upload_extensions=ALLOWED_UPLOAD_EXTENSIONS,
    )
