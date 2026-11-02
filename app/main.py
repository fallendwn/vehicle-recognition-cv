from __future__ import annotations

import io
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from app.dinov2_inference import DINOv2TransportClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "app" / "static"
DINO_MODEL_PATH = PROJECT_ROOT / "models" / "dinov2_transport_classifier.pt"
RETRIEVAL_PATH = PROJECT_ROOT / "models" / "dinov2_retrieval.pt"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(title="Transport Classifier", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

classifier: DINOv2TransportClassifier | None = None
model_error: str | None = None


@app.on_event("startup")
def load_model() -> None:
    global classifier, model_error
    model_path = Path(os.getenv("MODEL_PATH", str(DINO_MODEL_PATH)))
    retrieval_path = Path(os.getenv("RETRIEVAL_PATH", str(RETRIEVAL_PATH)))
    try:
        classifier = DINOv2TransportClassifier(model_path, retrieval_path)
        model_error = None
    except (FileNotFoundError, RuntimeError, KeyError, ValueError) as error:
        classifier = None
        model_error = str(error)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str | bool | list[str] | None]:
    return {
        "status": "ok" if classifier else "model_not_ready",
        "model_loaded": classifier is not None,
        "classes": classifier.class_names if classifier else [],
        "model": classifier.backbone_name if classifier else None,
        "device": str(classifier.device).upper() if classifier else None,
        "detail": model_error,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Модель ещё не обучена. Сначала запустите train_dinov2.py.",
        )
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Поддерживаются только JPG, PNG и WebP.",
        )

    raw_image = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if len(raw_image) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Размер файла не должен превышать 10 МБ.")

    try:
        image = Image.open(io.BytesIO(raw_image))
        image.verify()
        image = Image.open(io.BytesIO(raw_image)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Не удалось прочитать изображение.")

    result = classifier.predict(image)
    return {
        "label": result.label,
        "confidence": result.confidence,
        "probabilities": result.probabilities,
        "latency_ms": result.latency_ms,
        "device": result.device,
        "heatmap": result.heatmap_data_url,
        "neighbors": result.neighbors,
    }


@app.get("/api/reference/{index}", include_in_schema=False)
def reference_image(index: int) -> FileResponse:
    if classifier is None or index < 0 or index >= len(classifier.retrieval_paths):
        raise HTTPException(status_code=404, detail="Reference image was not found.")
    stored_path = Path(classifier.retrieval_paths[index])
    path = (
        stored_path.resolve()
        if stored_path.is_absolute()
        else (PROJECT_ROOT / stored_path).resolve()
    )
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        raise HTTPException(status_code=404, detail="Reference image was not found.")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Reference image was not found.")
    return FileResponse(path)

