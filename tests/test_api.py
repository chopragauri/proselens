"""API tests.

Tests that need trained artifacts are skipped when those artifacts are absent,
because `data/models/` is not committed — it is rebuilt by the pipeline. The
validation and upload-safety tests do not need a model and always run, since
those are the paths that face untrusted input.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app
from app.schemas.analysis import MAXIMUM_CHARACTERS

REQUIRES_MODEL = pytest.mark.skipif(
    not get_settings().models_are_available,
    reason="Trained artifacts not present; run scripts/train.py.",
)

# Long enough to clear the five-sentence floor for a real verdict.
SAMPLE_ESSAY = (
    "I burned the dumplings, every single one of them. My grandmother did not "
    "say anything about it. She simply handed me another wrapper and waited. "
    "I tried again and the filling spilled out across the counter. The third "
    "one tore before it reached the pan. By the fourteenth attempt something "
    "finally held together, and she nodded once. That kitchen in Queens is "
    "where I learned what patience actually costs a person. It is not a calm "
    "feeling at all. It is mostly just refusing to stop when stopping would "
    "be easier. I think about that every time a problem does not yield on the "
    "first attempt. The dumplings were still bad, but I had made them."
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


# --------------------------------------------------------------------------
# health
# --------------------------------------------------------------------------


def test_health_always_responds(client: TestClient) -> None:
    """Health must answer even when the model is missing — that is its job."""
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert isinstance(body["model_loaded"], bool)


@REQUIRES_MODEL
def test_health_reports_loaded_model(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["spacy_model"] == "en_core_web_sm"


# --------------------------------------------------------------------------
# request validation (no model required)
# --------------------------------------------------------------------------


def test_empty_text_is_rejected(client: TestClient) -> None:
    assert client.post("/api/analyze", json={"text": ""}).status_code == 422


def test_whitespace_only_text_is_rejected(client: TestClient) -> None:
    assert client.post("/api/analyze", json={"text": "   \n\t  "}).status_code == 422


def test_missing_text_field_is_rejected(client: TestClient) -> None:
    assert client.post("/api/analyze", json={}).status_code == 422


def test_oversized_text_is_rejected(client: TestClient) -> None:
    oversized = "word " * (MAXIMUM_CHARACTERS // 2)
    assert client.post("/api/analyze", json={"text": oversized}).status_code == 422


# --------------------------------------------------------------------------
# upload safety (no model required)
# --------------------------------------------------------------------------


def test_upload_rejects_non_txt_extension(client: TestClient) -> None:
    response = client.post(
        "/api/upload",
        files={"file": ("essay.pdf", io.BytesIO(b"content"), "application/pdf")},
    )
    assert response.status_code == 400
    assert ".txt" in response.json()["detail"]


def test_upload_rejects_executable_disguised_by_content_type(client: TestClient) -> None:
    """The extension allow-list is what decides, not the declared content type."""
    response = client.post(
        "/api/upload",
        files={"file": ("payload.sh", io.BytesIO(b"rm -rf /"), "text/plain")},
    )
    assert response.status_code == 400


def test_upload_rejects_oversized_file(client: TestClient) -> None:
    limit = get_settings().maximum_upload_bytes
    payload = b"a" * (limit + 1024)
    response = client.post(
        "/api/upload", files={"file": ("big.txt", io.BytesIO(payload), "text/plain")}
    )
    assert response.status_code == 413


def test_upload_rejects_invalid_utf8(client: TestClient) -> None:
    """Strict decoding: mangled bytes must not become a confident analysis."""
    response = client.post(
        "/api/upload",
        files={"file": ("essay.txt", io.BytesIO(b"\xff\xfe\x00binary"), "text/plain")},
    )
    assert response.status_code == 400
    assert "UTF-8" in response.json()["detail"]


def test_upload_rejects_empty_file(client: TestClient) -> None:
    response = client.post(
        "/api/upload", files={"file": ("empty.txt", io.BytesIO(b"   \n  "), "text/plain")}
    )
    assert response.status_code == 400


# --------------------------------------------------------------------------
# analysis (model required)
# --------------------------------------------------------------------------


@REQUIRES_MODEL
def test_analyze_returns_a_complete_response(client: TestClient) -> None:
    response = client.post("/api/analyze", json={"text": SAMPLE_ESSAY})
    assert response.status_code == 200
    body = response.json()

    assert 0 <= body["risk_score"] <= 100
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["assessment"] in {
        "likely_human", "mixed_signals", "likely_machine", "insufficient_text"
    }
    assert body["summary_text"]
    assert body["sentences"]
    assert body["model_version"]


@REQUIRES_MODEL
def test_sentence_offsets_slice_the_submitted_text(client: TestClient) -> None:
    """The property the whole highlighting feature depends on."""
    body = client.post("/api/analyze", json={"text": SAMPLE_ESSAY}).json()
    for sentence in body["sentences"]:
        assert SAMPLE_ESSAY[sentence["start"] : sentence["end"]] == sentence["text"]


@REQUIRES_MODEL
def test_summary_counts_match_the_sentence_list(client: TestClient) -> None:
    body = client.post("/api/analyze", json={"text": SAMPLE_ESSAY}).json()
    summary = body["summary"]
    assert summary["sentence_count"] == len(body["sentences"])

    banded = (
        summary["high_risk_sentences"]
        + summary["medium_risk_sentences"]
        + summary["low_risk_sentences"]
    )
    assert banded == summary["scorable_sentence_count"]
    assert summary["unscored_sentences"] == (
        summary["sentence_count"] - summary["scorable_sentence_count"]
    )


@REQUIRES_MODEL
def test_every_scored_sentence_carries_evidence(client: TestClient) -> None:
    """No sentence may show a score without the measurements behind it."""
    body = client.post("/api/analyze", json={"text": SAMPLE_ESSAY}).json()
    scored = [s for s in body["sentences"] if s["risk_score"] is not None]
    assert scored
    for sentence in scored:
        assert sentence["explanation"]
        assert sentence["signals"]
        for signal in sentence["signals"]:
            assert signal["direction"] in {"machine", "human"}
            assert signal["measured_feature"]


@REQUIRES_MODEL
def test_unscorable_sentences_are_returned_but_not_scored(client: TestClient) -> None:
    body = client.post(
        "/api/analyze", json={"text": "Yes. " + SAMPLE_ESSAY}
    ).json()
    unscorable = [s for s in body["sentences"] if not s["is_scorable"]]
    assert unscorable
    for sentence in unscorable:
        assert sentence["risk_score"] is None
        assert sentence["risk_band"] == "not_scored"


@REQUIRES_MODEL
def test_short_text_does_not_receive_a_machine_verdict(client: TestClient) -> None:
    """Measured: below 12 sentences the false-positive rate exceeds 20%.

    The assessment must therefore never read 'likely_machine' on short input,
    however high the raw score goes.
    """
    body = client.post(
        "/api/analyze",
        json={"text": "The results were encouraging. Moreover, the data supported it. "
                      "Furthermore, the conclusion followed. Additionally, it was clear. "
                      "Ultimately, this demonstrates the point."},
    ).json()
    assert body["assessment"] != "likely_machine"


@REQUIRES_MODEL
def test_very_short_text_is_refused_a_verdict(client: TestClient) -> None:
    body = client.post("/api/analyze", json={"text": "Two short lines. Nothing more."}).json()
    assert body["assessment"] == "insufficient_text"


@REQUIRES_MODEL
def test_passages_stay_within_the_document(client: TestClient) -> None:
    body = client.post("/api/analyze", json={"text": SAMPLE_ESSAY}).json()
    for passage in body["passages"]:
        assert passage["start"] < passage["end"] <= len(SAMPLE_ESSAY)
        assert passage["start_sentence_index"] <= passage["end_sentence_index"]


@REQUIRES_MODEL
def test_upload_and_paste_agree(client: TestClient) -> None:
    """One analysis path, reached two ways: results must be identical."""
    pasted = client.post("/api/analyze", json={"text": SAMPLE_ESSAY}).json()
    uploaded = client.post(
        "/api/upload",
        files={"file": ("essay.txt", io.BytesIO(SAMPLE_ESSAY.encode()), "text/plain")},
    ).json()
    assert uploaded["risk_score"] == pasted["risk_score"]
    assert uploaded["assessment"] == pasted["assessment"]
    assert len(uploaded["sentences"]) == len(pasted["sentences"])


@REQUIRES_MODEL
def test_analysis_is_deterministic(client: TestClient) -> None:
    first = client.post("/api/analyze", json={"text": SAMPLE_ESSAY}).json()
    second = client.post("/api/analyze", json={"text": SAMPLE_ESSAY}).json()
    assert first["risk_score"] == second["risk_score"]
    assert first["confidence"] == second["confidence"]
    assert first["summary_text"] == second["summary_text"]
