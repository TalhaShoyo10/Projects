import io
import sys
import uuid
import base64
import random
import torch
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append("../model")
from model import load_model

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MNIST CAPTCHA API",
    description="""
A CAPTCHA service powered by a neural network trained on MNIST.

## How to integrate

1. Call `GET /captcha` to get a noisy digit image and a unique `captcha_id`
2. Display the image to your user and collect their answer
3. Call `POST /verify` with the `captcha_id` and the user's answer
4. Use the `success` field in the response to allow or block the action

For direct digit classification without the CAPTCHA flow, use `POST /predict`.
    """,
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model + Data (loaded once at startup) ─────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    MODEL = load_model("../model/weights/best_model.bin", DEVICE)
except Exception as e:
    raise RuntimeError(
        "Failed to load `model/weights/best_model.bin`. If you changed the model/training "
        "code, retrain the model (cd model; python train.py) to regenerate compatible weights. "
        f"Original error: {e}"
    )

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

MNIST_TEST = datasets.MNIST(
    root="./data", train=False,
    transform=transforms.ToTensor(), download=True
)

# In-memory CAPTCHA store {captcha_id: {"answer": "1234", "length": 4}}
# For production replace with Redis (and add expiry).
captcha_store: dict[str, dict[str, object]] = {}

# No normalization — model was trained without it (eval_transform = ToTensor() only)
PREDICT_TRANSFORM = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
])


def _open_grayscale(image_bytes: bytes) -> Image.Image:
    """
    Robustly open an image and return grayscale, handling alpha by compositing
    onto white (common for canvas exports).
    """
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, rgba).convert("L")
    else:
        img = img.convert("L")
    return img


# ── Noise helpers ─────────────────────────────────────────────────────────────

def add_gaussian_noise(tensor: torch.Tensor, std: float = 0.3) -> torch.Tensor:
    return torch.clamp(tensor + torch.randn_like(tensor) * std, 0.0, 1.0)

def add_salt_pepper_noise(tensor: torch.Tensor, amount: float = 0.05) -> torch.Tensor:
    noisy = tensor.clone()
    noisy[torch.rand_like(tensor) < amount / 2] = 1.0
    noisy[torch.rand_like(tensor) < amount / 2] = 0.0
    return noisy


# ── Shared inference helper ───────────────────────────────────────────────────

def run_inference(tensor: torch.Tensor) -> tuple[int, list[float]]:
    """Add batch dim, run model, return (predicted_digit, probabilities)."""
    inp = tensor.unsqueeze(0).to(DEVICE)   # (1, 1, 28, 28) for CNN
    with torch.no_grad():
        probs = torch.softmax(MODEL(inp), dim=1).squeeze().tolist()
    return int(np.argmax(probs)), probs


# ── Schemas ───────────────────────────────────────────────────────────────────

class CaptchaResponse(BaseModel):
    captcha_id: str
    image_base64: str
    noise_type: str
    noise_level: float
    length: int

class VerifyRequest(BaseModel):
    captcha_id: str
    answer: int | str

class VerifyResponse(BaseModel):
    success: bool
    true_label: int | str | None    # only revealed on failure so user can learn

class PredictionResponse(BaseModel):
    predicted_digit: int
    confidence: float
    probabilities: list[float]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
def root():
    return {
        "message": "MNIST CAPTCHA API is running. Visit /docs for full documentation."
    }


@app.get("/health", tags=["General"])
def health():
    return {"status": "ok"}


@app.get("/captcha", response_model=CaptchaResponse, tags=["CAPTCHA"])
def get_captcha(
    noise_type: str = "gaussian",
    noise_level: float = 0.3,
    display_size: int = 196,
    length: int = 4,
):
    """
    Generate a noisy MNIST CAPTCHA image.

    - **noise_type**: `gaussian` or `salt_pepper`
    - **noise_level**: 0.0 (clean) to 1.0 (very noisy). Default 0.3
    - **display_size**: height (px) of returned image. Default 196
    - **length**: number of digits in the CAPTCHA. Default 4

    Returns a base64 PNG and a `captcha_id` to pass into `/verify`.
    """
    if noise_type not in ("gaussian", "salt_pepper"):
        raise HTTPException(
            status_code=400,
            detail="noise_type must be 'gaussian' or 'salt_pepper'"
        )
    if not 0.0 <= noise_level <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="noise_level must be between 0.0 and 1.0"
        )

    if not (1 <= length <= 8):
        raise HTTPException(status_code=400, detail="length must be between 1 and 8")

    digits: list[int] = []
    tiles: list[Image.Image] = []

    for _ in range(length):
        idx = random.randint(0, len(MNIST_TEST) - 1)
        image_tensor, true_label = MNIST_TEST[idx]  # (1,28,28), int
        digits.append(int(true_label))

        if noise_type == "gaussian":
            noisy = add_gaussian_noise(image_tensor, std=noise_level)
        else:
            noisy = add_salt_pepper_noise(image_tensor, amount=noise_level)

        arr = (noisy.squeeze().numpy() * 255).astype(np.uint8)
        tiles.append(Image.fromarray(arr))

    answer = "".join(str(d) for d in digits)

    gap = 4
    tile_w, tile_h = 28, 28
    total_w = tile_w * length + gap * (length - 1)
    canvas = Image.new("L", (total_w, tile_h), color=0)

    x = 0
    for tile in tiles:
        canvas.paste(tile, (x, 0))
        x += tile_w + gap

    out_w = int(display_size * (total_w / tile_h))
    out_h = int(display_size)
    pil = canvas.resize((out_w, out_h), Image.NEAREST)

    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode()

    # Store true label server-side — external service never sees this
    captcha_id = str(uuid.uuid4())
    captcha_store[captcha_id] = {"answer": answer, "length": length}

    return CaptchaResponse(
        captcha_id=captcha_id,
        image_base64=b64,
        noise_type=noise_type,
        noise_level=noise_level,
        length=length,
    )


@app.post("/verify", response_model=VerifyResponse, tags=["CAPTCHA"])
def verify_captcha(body: VerifyRequest):
    """
    Verify a user's answer to a CAPTCHA.

    - **captcha_id**: the ID returned by `/captcha`
    - **answer**: the digits the user entered (e.g. `5301`)

    Each `captcha_id` is single-use — deleted after verification.
    On failure, reveals the true label so the integrating service can handle it.
    """
    if body.captcha_id not in captcha_store:
        raise HTTPException(
            status_code=404,
            detail="CAPTCHA ID not found. It may have already been used or expired."
        )

    record = captcha_store.pop(body.captcha_id)    # delete after one use
    true_answer = str(record.get("answer", ""))
    length = int(record.get("length", len(true_answer) or 1))

    user_answer = body.answer
    if isinstance(user_answer, int):
        user_answer_str = str(user_answer)
    else:
        user_answer_str = "".join(ch for ch in str(user_answer) if ch.isdigit())

    success = (user_answer_str == true_answer)

    return VerifyResponse(
        success=success,
        true_label=None if success else (int(true_answer) if length == 1 else true_answer),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Direct Inference"])
async def predict(file: UploadFile = File(...)):
    """
    Directly classify an uploaded digit image without the CAPTCHA flow.

    Accepts any standard image format (PNG, JPG, etc).
    Preprocessing: inverts white-background images, crops tightly around
    the digit, pads with margin to match MNIST centering, then resizes to 28x28.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()

    try:
        img = _open_grayscale(image_bytes)
        arr = np.array(img)

        # Invert if white background
        if arr.mean() > 127:
            arr = 255 - arr

        # Crop tightly around the digit
        rows = np.any(arr > 30, axis=1)
        cols = np.any(arr > 30, axis=0)
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            arr = arr[rmin:rmax+1, cmin:cmax+1]

        # Pad to square with margin to match MNIST centering
        h, w = arr.shape
        size = max(h, w)
        margin = size // 4
        padded = np.zeros((size + margin * 2, size + margin * 2), dtype=np.uint8)
        y_off = margin + (size - h) // 2
        x_off = margin + (size - w) // 2
        padded[y_off:y_off+h, x_off:x_off+w] = arr

        img = Image.fromarray(padded)
        tensor = PREDICT_TRANSFORM(img)

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Image processing failed: {e}")

    predicted, probs = run_inference(tensor)

    return PredictionResponse(
        predicted_digit=predicted,
        confidence=round(probs[predicted], 4),
        probabilities=[round(p, 4) for p in probs],
    )

@app.post("/debug", tags=["Debug"])
async def debug_predict(file: UploadFile = File(...)):
    """Returns the preprocessed image as base64 so you can see what the model sees."""
    image_bytes = await file.read()
    
    img = _open_grayscale(image_bytes)
    arr = np.array(img)
    
    # Save what arrives
    raw_mean = arr.mean()
    
    # Invert if white background
    if arr.mean() > 127:
        arr = 255 - arr
    
    # Crop
    rows = np.any(arr > 30, axis=1)
    cols = np.any(arr > 30, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        arr = arr[rmin:rmax+1, cmin:cmax+1]
    
    # Pad
    h, w = arr.shape
    size = max(h, w)
    margin = size // 4
    padded = np.zeros((size + margin * 2, size + margin * 2), dtype=np.uint8)
    y_off = margin + (size - h) // 2
    x_off = margin + (size - w) // 2
    padded[y_off:y_off+h, x_off:x_off+w] = arr
    
    # Resize to 28x28
    final = Image.fromarray(padded).resize((28, 28), Image.LANCZOS)
    
    # Encode back to base64
    buf = io.BytesIO()
    final.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    
    return {"raw_mean": float(raw_mean), "preprocessed_image": b64}
