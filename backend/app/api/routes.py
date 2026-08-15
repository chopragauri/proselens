"""API routes.

Upload handling is the security-sensitive surface here, so it is deliberately
strict: extension allow-list, byte cap enforced while reading rather than after,
strict UTF-8 decoding, and nothing ever written to disk. The uploaded bytes are
decoded in memory and handed to the same analysis path as pasted text, so there
is no second code path that could drift.
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.nlp.pipeline import SPACY_MODEL_NAME
from app.schemas.analysis import (
    MAXIMUM_CHARACTERS,
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
)
from app.services.analysis import get_analysis_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Liveness plus whether the models actually loaded.

    Reports degraded rather than raising: a health endpoint that errors when
    the model is missing is useless for diagnosing that the model is missing.
    """
    settings = get_settings()
    if not settings.models_are_available:
        return HealthResponse(status="degraded", model_loaded=False)

    try:
        service = get_analysis_service()
    except (OSError, ValueError, KeyError):
        return HealthResponse(status="degraded", model_loaded=False)

    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=service.model_version,
        spacy_model=SPACY_MODEL_NAME,
    )


@router.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze pasted essay text."""
    return _run_analysis(request.text)


@router.post("/upload", response_model=AnalyzeResponse, tags=["analysis"])
async def upload(file: UploadFile = File(...)) -> AnalyzeResponse:
    """Analyze an uploaded .txt file."""
    settings = get_settings()

    filename = file.filename or ""
    extension = filename[filename.rfind(".") :].lower() if "." in filename else ""
    if extension not in settings.allowed_upload_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Only "
                + ", ".join(sorted(settings.allowed_upload_extensions))
                + " files are accepted."
            ),
        )

    # Read with a cap rather than reading everything and checking afterwards,
    # so an oversized upload cannot force a large allocation first.
    limit = settings.maximum_upload_bytes
    payload = await file.read(limit + 1)
    if len(payload) > limit:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File exceeds the {limit:,}-byte limit.",
        )

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        # Strict decoding, not errors="replace": a file that is not UTF-8 text
        # is very likely not an essay, and silently mangling it into mojibake
        # would produce a confident analysis of garbage.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be valid UTF-8 text.",
        ) from error

    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File contains no text.",
        )

    if len(text) > MAXIMUM_CHARACTERS:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Text exceeds the {MAXIMUM_CHARACTERS:,}-character limit.",
        )

    return _run_analysis(text)


def _run_analysis(text: str) -> AnalyzeResponse:
    """Shared path for pasted and uploaded text."""
    settings = get_settings()
    if not settings.models_are_available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model artifacts are not available. Run scripts/train.py to "
                "produce them."
            ),
        )

    try:
        return get_analysis_service().analyze(text)
    except (OSError, ValueError, KeyError) as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {error}",
        ) from error
