"""FastAPI application entry point.

Run with:
    uvicorn app.main:app --reload --app-dir backend
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings

API_PREFIX = "/api"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup rather than on the first request.

    Without this the first user to submit an essay pays for the spaCy pipeline
    and a 12 MB language model, and two concurrent first requests would load
    both twice. Failure is logged and tolerated: /api/health then reports
    degraded, which is more useful than a server that refuses to start.
    """
    settings = get_settings()
    if settings.models_are_available:
        try:
            from app.services.analysis import get_analysis_service

            get_analysis_service()
        except (OSError, ValueError, KeyError) as error:  # pragma: no cover
            print(f"Model preload failed: {error}. /api/health will report degraded.")
    else:
        print(
            "Model artifacts not found. Run scripts/train.py; "
            "/api/health will report degraded until then."
        )
    yield


app = FastAPI(
    title="ProseLens",
    description=(
        "Explainable analysis of writing patterns associated with "
        "machine-generated text."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # An explicit origin list, not "*": the API is called from a known frontend,
    # and a wildcard would let any page on the internet drive it from a user's
    # browser.
    allow_origins=list(get_settings().cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(router, prefix=API_PREFIX)
