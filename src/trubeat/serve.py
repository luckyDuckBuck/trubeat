"""REST service (Phase 4). FR-15. Offline by default — no network calls in the path."""

from fastapi import FastAPI, UploadFile

app = FastAPI(title="TrueBeat")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(video: UploadFile):
    """Multipart video upload → Prediction JSON (FR-12 schema)."""
    raise NotImplementedError  # Phase 4
