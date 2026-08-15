"""Pydantic request and response models for the analysis API.

Naming reflects a decision made in `docs/methodology.md`: the API returns a
**risk score**, not a probability. Measured expected calibration error on the
test split is 0.014 overall, but the mid-range bins — exactly where a user
needs the number to mean something — are thin and unstable. Calling the output
`risk_score` rather than `probability` keeps the interface honest about what
has actually been established.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "RiskBand",
    "Assessment",
    "AnalyzeRequest",
    "SignalEvidence",
    "SentenceResult",
    "PassageResult",
    "DocumentSummary",
    "AnalyzeResponse",
    "HealthResponse",
    "MAXIMUM_CHARACTERS",
]

# Roughly a 15,000-word essay. Generous for an admissions essay while bounding
# the work a single request can ask for.
MAXIMUM_CHARACTERS = 100_000
MINIMUM_CHARACTERS = 1


class RiskBand(str, Enum):
    """Coarse band for display. The UI must not rely on colour alone."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    NOT_SCORED = "not_scored"


class Assessment(str, Enum):
    """Document-level verdict, deliberately hedged in wording."""

    LIKELY_HUMAN = "likely_human"
    MIXED_SIGNALS = "mixed_signals"
    LIKELY_MACHINE = "likely_machine"
    INSUFFICIENT_TEXT = "insufficient_text"


class AnalyzeRequest(BaseModel):
    """An essay submitted for analysis."""

    text: str = Field(
        ...,
        min_length=MINIMUM_CHARACTERS,
        max_length=MAXIMUM_CHARACTERS,
        description="The essay to analyze.",
    )

    @field_validator("text")
    @classmethod
    def must_not_be_only_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must contain more than whitespace.")
        return value


class SignalEvidence(BaseModel):
    """One signal family's contribution to a score."""

    family: str = Field(..., description="Display name, e.g. 'Lexical diversity'.")
    direction: str = Field(..., description="'machine' or 'human'.")
    contribution: float = Field(
        ..., description="Log-odds this family contributed to the score."
    )
    descriptor: str = Field(
        ..., description="Qualitative reading grounded in the z-score."
    )
    measured_feature: str = Field(..., description="Strongest feature in the family.")
    measured_value: float | None = Field(
        None, description="That feature's raw measured value."
    )
    z_score: float | None = Field(
        None, description="Standard deviations from the human training average."
    )


class SentenceResult(BaseModel):
    """Per-sentence analysis, including the offsets needed for highlighting."""

    index: int
    text: str
    start: int = Field(..., description="Character offset into the submitted text.")
    end: int
    paragraph_index: int
    token_count: int
    is_scorable: bool = Field(
        ...,
        description=(
            "False when the sentence is too short for stable measurement. "
            "Such sentences are returned and rendered but are not scored."
        ),
    )
    risk_score: int | None = Field(None, ge=0, le=100)
    risk_band: RiskBand
    signals: list[SignalEvidence] = Field(default_factory=list)
    explanation: str | None = None


class PassageResult(BaseModel):
    """A sliding window of consecutive sentences."""

    start_sentence_index: int
    end_sentence_index: int
    start: int
    end: int
    risk_score: int = Field(..., ge=0, le=100)
    risk_band: RiskBand


class DocumentSummary(BaseModel):
    """Counts and aggregate measurements for the whole essay."""

    sentence_count: int
    scorable_sentence_count: int
    paragraph_count: int
    word_count: int
    character_count: int
    high_risk_sentences: int
    medium_risk_sentences: int
    low_risk_sentences: int
    unscored_sentences: int
    mean_sentence_risk: float | None
    lexical_diversity: float | None = Field(
        None, description="Document MTLD; higher means more varied vocabulary."
    )
    sentence_length_variation: float | None = Field(
        None, description="Coefficient of variation of sentence lengths."
    )
    mean_predictability: float | None = Field(
        None, description="Mean cross-entropy under the human reference model."
    )


class AnalyzeResponse(BaseModel):
    """The complete analysis of one essay."""

    risk_score: int = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Document risk on a 0-100 scale. Deliberately not labelled a "
            "probability; see docs/methodology.md."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "How much evidence supports the score, from evidence quantity, "
            "coverage and consistency. Independent of risk."
        ),
    )
    confidence_band: RiskBand
    assessment: Assessment
    summary_text: str = Field(..., description="Evidence-based prose summary.")
    signals: list[SignalEvidence]
    sentences: list[SentenceResult]
    passages: list[PassageResult]
    summary: DocumentSummary
    model_version: str


class HealthResponse(BaseModel):
    """Liveness and model-loading status."""

    status: str
    model_loaded: bool
    model_version: str | None = None
    spacy_model: str | None = None
