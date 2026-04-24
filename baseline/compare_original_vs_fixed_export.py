"""
Reproduces the 'everything predicts as 1' failure mode caused by exporting the Streamlit
canvas incorrectly (casting float [0,1] to uint8 without scaling).

This script does NOT depend on running the API. It loads your existing model weights
from `model/weights/best_model.bin` and shows how the same digit can become a blank-ish
image and collapse predictions.
"""

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms


def load_existing_model():
    import sys

    sys.path.append("model")
    from model import load_model  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_model("model/weights/best_model.bin", device), device


def original_ui_export_bug(img_data: np.ndarray) -> bytes:
    """
    Mirrors `ui/app.py`:
      arr = img_data[:, :, 0].astype(np.uint8)
      Image.fromarray(arr).save(...)

    If img_data is float32 in [0,1], this produces almost all 0/1 pixels.
    """
    arr = img_data[:, :, 0].astype(np.uint8)
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def fixed_export(img_data: np.ndarray) -> bytes:
    """
    A safe export: scale float [0,1] -> [0,255], keep RGBA, alpha composite on white.
    """
    arr = img_data
    if arr.dtype != np.uint8:
        arr_f = arr.astype(np.float32)
        if float(arr_f.max()) <= 1.0:
            arr_f = arr_f * 255.0
        arr = np.clip(arr_f, 0.0, 255.0).astype(np.uint8)

    if arr.shape[2] == 3:
        rgba = np.concatenate([arr, np.full((arr.shape[0], arr.shape[1], 1), 255, dtype=np.uint8)], axis=2)
    else:
        rgba = arr

    pil_rgba = Image.fromarray(rgba, mode="RGBA")
    white = Image.new("RGBA", pil_rgba.size, (255, 255, 255, 255))
    composed = Image.alpha_composite(white, pil_rgba).convert("L")

    buf = io.BytesIO()
    composed.save(buf, format="PNG")
    return buf.getvalue()


def api_like_preprocess(image_bytes: bytes) -> torch.Tensor:
    """
    Mirrors the preprocessing logic in `api/main.py` roughly (invert-if-white + crop + pad + resize),
    then ToTensor(). This uses no normalization because your existing model was trained without it.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    arr = np.array(img)

    if arr.mean() > 127:
        arr = 255 - arr

    rows = np.any(arr > 30, axis=1)
    cols = np.any(arr > 30, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        arr = arr[rmin : rmax + 1, cmin : cmax + 1]

    h, w = arr.shape
    size = max(h, w)
    margin = size // 4
    padded = np.zeros((size + margin * 2, size + margin * 2), dtype=np.uint8)
    y_off = margin + (size - h) // 2
    x_off = margin + (size - w) // 2
    padded[y_off : y_off + h, x_off : x_off + w] = arr

    final = Image.fromarray(padded).resize((28, 28), Image.LANCZOS)
    return transforms.ToTensor()(final)


def main():
    weights = Path("model/weights/best_model.bin")
    if not weights.exists():
        raise SystemExit("Missing `model/weights/best_model.bin`")

    model, device = load_existing_model()

    # Take a real MNIST digit, then convert it into a canvas-like RGBA float image in [0,1]
    mnist = datasets.MNIST(root="model/data", train=False, transform=transforms.ToTensor(), download=False)
    x, y = mnist[0]

    # Canvas-like: white background with black digit (invert), upscale to 140, pack into RGBA float
    base = (1.0 - x.squeeze().numpy()).astype(np.float32)  # 0..1, background ~1, digit ~0
    pil = Image.fromarray((base * 255).astype(np.uint8)).resize((140, 140), Image.NEAREST)
    gray = np.array(pil).astype(np.float32) / 255.0
    rgba = np.stack([gray, gray, gray, np.ones_like(gray)], axis=2)  # float RGBA

    bug_png = original_ui_export_bug(rgba)
    ok_png = fixed_export(rgba)

    bug_x = api_like_preprocess(bug_png).unsqueeze(0).to(device)
    ok_x = api_like_preprocess(ok_png).unsqueeze(0).to(device)

    with torch.no_grad():
        bug_pred = int(model(bug_x).argmax(1).item())
        ok_pred = int(model(ok_x).argmax(1).item())

    print("true_label:", int(y))
    print("bug_export: mean=", float(np.array(Image.open(io.BytesIO(bug_png))).mean()), "pred=", bug_pred)
    print("fixed_export: mean=", float(np.array(Image.open(io.BytesIO(ok_png))).mean()), "pred=", ok_pred)


if __name__ == "__main__":
    main()
