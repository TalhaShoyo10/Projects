import io
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from baseline.model_simple import load_simple_model, default_device
from baseline.preprocess_simple import (
    preprocess_pil_for_mnist,
    tensor_to_pil_unnormalized,
    encode_png_base64,
    debug_summary,
)


ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "weights" / "mnist_cnn.pt"

DEVICE = default_device()
MODEL = None


def get_model():
    global MODEL
    if MODEL is None:
        if not WEIGHTS_PATH.exists():
            raise RuntimeError(f"Missing weights: {WEIGHTS_PATH}. Run `python baseline/train_simple.py` first.")
        MODEL = load_simple_model(WEIGHTS_PATH, DEVICE)
    return MODEL


app = FastAPI(title="Baseline MNIST API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    predicted_digit: int
    confidence: float
    probabilities: list[float]


class DebugResponse(BaseModel):
    summary: dict
    preprocessed_image_base64: str
    prediction: PredictionResponse


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "weights": str(WEIGHTS_PATH)}


def _infer_from_pil(img: Image.Image) -> tuple[int, list[float], dict]:
    model = get_model()
    res = preprocess_pil_for_mnist(img)
    x = res.tensor.unsqueeze(0).to(DEVICE)  # (1,1,28,28)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)
    pred = int(probs.argmax())
    return pred, probs.tolist(), debug_summary(res)


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    image_bytes = await file.read()
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to decode image: {e}")

    pred, probs, _summary = _infer_from_pil(img)
    return PredictionResponse(
        predicted_digit=pred,
        confidence=round(float(probs[pred]), 4),
        probabilities=[round(float(p), 4) for p in probs],
    )


@app.post("/debug", response_model=DebugResponse)
async def debug(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    image_bytes = await file.read()
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to decode image: {e}")

    pred, probs, summary = _infer_from_pil(img)

    # Visualize what the model actually saw (unnormalized tensor back to grayscale)
    # We re-run preprocess to get the tensor for visualization (cheap and keeps code simple).
    res = preprocess_pil_for_mnist(img)
    vis = tensor_to_pil_unnormalized(res.tensor)
    b64 = encode_png_base64(vis)

    prediction = PredictionResponse(
        predicted_digit=pred,
        confidence=round(float(probs[pred]), 4),
        probabilities=[round(float(p), 4) for p in probs],
    )
    return DebugResponse(summary=summary, preprocessed_image_base64=b64, prediction=prediction)
