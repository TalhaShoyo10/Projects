import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


@dataclass(frozen=True)
class PreprocessResult:
    tensor: torch.Tensor  # (1,28,28) float32
    raw_mean: float
    inverted: bool
    cropped: bool
    bbox: tuple[int, int, int, int] | None  # (rmin, rmax, cmin, cmax) on the pre-invert array


_TO_TENSOR_NORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ]
)


def _ensure_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Accepts uint8 [0,255] or float [0,1]/[0,255] and returns uint8 [0,255].
    """
    if arr.dtype == np.uint8:
        return arr
    arr_f = arr.astype(np.float32)
    maxv = float(np.max(arr_f)) if arr_f.size else 0.0
    if maxv <= 1.0:
        arr_f = arr_f * 255.0
    arr_f = np.clip(arr_f, 0.0, 255.0)
    return arr_f.astype(np.uint8)


def preprocess_pil_for_mnist(img: Image.Image) -> PreprocessResult:
    """
    Preprocess an input image into the MNIST-like tensor the baseline model expects.

    Assumptions:
    - Incoming image is a single digit, possibly on a white background (common in canvas)
    - We invert if background appears white (mean > 127 on uint8)
    - We crop using a low threshold, pad to square, resize to 28x28
    - We apply MNIST normalization (mean/std)
    """
    img_l = img.convert("L")
    arr0 = _ensure_uint8(np.array(img_l))

    raw_mean = float(arr0.mean()) if arr0.size else 0.0

    inverted = False
    arr = arr0
    if raw_mean > 127.0:
        arr = 255 - arr
        inverted = True

    # Crop digit bbox (threshold chosen to ignore light noise but catch strokes)
    threshold = 20
    rows = np.any(arr > threshold, axis=1)
    cols = np.any(arr > threshold, axis=0)

    bbox = None
    cropped = False
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bbox = (int(rmin), int(rmax), int(cmin), int(cmax))
        arr = arr[rmin : rmax + 1, cmin : cmax + 1]
        cropped = True

    # Pad to square with margin (MNIST digits are centered with padding)
    h, w = arr.shape
    size = max(h, w)
    margin = max(2, size // 4)
    padded = np.zeros((size + margin * 2, size + margin * 2), dtype=np.uint8)
    y_off = margin + (size - h) // 2
    x_off = margin + (size - w) // 2
    padded[y_off : y_off + h, x_off : x_off + w] = arr

    # Resize to 28x28
    final = Image.fromarray(padded).resize((28, 28), Image.LANCZOS)
    tensor = _TO_TENSOR_NORM(final).to(torch.float32)

    return PreprocessResult(
        tensor=tensor,
        raw_mean=raw_mean,
        inverted=inverted,
        cropped=cropped,
        bbox=bbox,
    )


def preprocess_image_bytes(image_bytes: bytes) -> PreprocessResult:
    img = Image.open(io.BytesIO(image_bytes))
    return preprocess_pil_for_mnist(img)


def encode_png_base64(img: Image.Image) -> str:
    import base64

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def tensor_to_pil_unnormalized(t: torch.Tensor) -> Image.Image:
    """
    Convert a normalized (1,28,28) tensor back to a viewable uint8 PNG.
    """
    if t.ndim == 3:
        t = t.squeeze(0)
    x = t.detach().cpu().numpy().astype(np.float32)
    x = (x * MNIST_STD) + MNIST_MEAN
    x = np.clip(x, 0.0, 1.0)
    arr = (x * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def debug_summary(res: PreprocessResult) -> dict[str, Any]:
    return {
        "raw_mean": res.raw_mean,
        "inverted": res.inverted,
        "cropped": res.cropped,
        "bbox": res.bbox,
    }
