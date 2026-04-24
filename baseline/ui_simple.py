import base64
import io
from dataclasses import dataclass

import numpy as np
import requests
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas


@dataclass(frozen=True)
class ApiConfig:
    url: str


API = ApiConfig(url="http://localhost:8001")


def _canvas_rgba_to_png_bytes(image_data: np.ndarray) -> bytes:
    """
    streamlit-drawable-canvas may return float32 in [0,1] or uint8 in [0,255].
    It is usually RGBA. We alpha-composite onto white, then encode as PNG.
    """
    if image_data is None:
        raise ValueError("Missing canvas image_data")

    arr = image_data
    if arr.dtype != np.uint8:
        arr_f = arr.astype(np.float32)
        if float(arr_f.max()) <= 1.0:
            arr_f = arr_f * 255.0
        arr = np.clip(arr_f, 0.0, 255.0).astype(np.uint8)

    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Unexpected canvas shape: {arr.shape}")

    if arr.shape[2] == 3:
        rgba = np.concatenate([arr, np.full((arr.shape[0], arr.shape[1], 1), 255, dtype=np.uint8)], axis=2)
    else:
        rgba = arr

    pil_rgba = Image.fromarray(rgba, mode="RGBA")
    white_bg = Image.new("RGBA", pil_rgba.size, (255, 255, 255, 255))
    composed = Image.alpha_composite(white_bg, pil_rgba).convert("L")

    buf = io.BytesIO()
    composed.save(buf, format="PNG")
    return buf.getvalue()


def _b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def api_health() -> dict | None:
    # Called on every rerun; cache helps reduce UI "spazzing" when the canvas updates.
    # (Canvas can trigger frequent reruns if update_streamlit=True.)
    try:
        r = requests.get(f"{API.url}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_predict(png_bytes: bytes) -> dict:
    r = requests.post(
        f"{API.url}/predict",
        files={"file": ("digit.png", png_bytes, "image/png")},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def api_debug(png_bytes: bytes) -> dict:
    r = requests.post(
        f"{API.url}/debug",
        files={"file": ("digit.png", png_bytes, "image/png")},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


st.set_page_config(page_title="Baseline MNIST", layout="centered")
st.title("Baseline MNIST (simple)")

if "canvas_key" not in st.session_state:
    st.session_state["canvas_key"] = 0
if "canvas_image_data" not in st.session_state:
    st.session_state["canvas_image_data"] = None

@st.cache_data(ttl=10)
def _cached_health(url: str) -> dict | None:
    try:
        r = requests.get(f"{url}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


health = _cached_health(API.url)
if health is None:
    st.warning("API not reachable. Start it with: `uvicorn baseline.api_simple:app --reload --port 8001`")
else:
    st.success(f"API OK ({health.get('device')})")

st.markdown("---")
st.write("Draw a digit (black ink on white background).")

canvas = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_width=12,
    stroke_color="black",
    background_color="white",
    height=180,
    width=180,
    drawing_mode="freedraw",
    # Must be True to reliably capture the latest drawing pixels.
    # We keep reruns lightweight (cached health; no API calls unless you click buttons).
    update_streamlit=True,
    key=f"canvas_{st.session_state['canvas_key']}",
)

if canvas.image_data is not None:
    # Some reruns may briefly report a blank white canvas; don't overwrite a real drawing with blank.
    img = canvas.image_data
    try:
        rgb = img[:, :, :3].astype(np.float32)
        mean_rgb = float(rgb.mean())
    except Exception:
        mean_rgb = 255.0

    if st.session_state["canvas_image_data"] is None or mean_rgb < 254.9:
        st.session_state["canvas_image_data"] = img

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    do_predict = st.button("Predict", type="primary")
with col2:
    do_debug = st.button("Debug (show preproc)")
with col3:
    do_clear = st.button("Clear")

if do_clear:
    st.session_state["canvas_image_data"] = None
    st.session_state["canvas_key"] += 1
    st.rerun()

if do_predict or do_debug:
    img_data = st.session_state.get("canvas_image_data")
    if img_data is None:
        st.error("No drawing yet.")
    else:
        try:
            png_bytes = _canvas_rgba_to_png_bytes(img_data)
        except Exception as e:
            st.error(f"Failed to convert canvas: {e}")
            st.stop()

        if do_predict:
            try:
                out = api_predict(png_bytes)
            except Exception as e:
                st.error(f"API /predict failed: {e}")
                st.stop()

            st.metric("Predicted digit", out["predicted_digit"])
            st.metric("Confidence", f"{out['confidence']:.1%}")
            st.bar_chart({str(i): out["probabilities"][i] for i in range(10)})

        if do_debug:
            try:
                out = api_debug(png_bytes)
            except Exception as e:
                st.error(f"API /debug failed: {e}")
                st.stop()

            st.markdown("**Preprocess summary**")
            st.json(out["summary"])
            st.markdown("**What the model sees (28x28)**")
            st.image(_b64_to_pil(out["preprocessed_image_base64"]), width=112)
            st.markdown("**Prediction**")
            st.json(out["prediction"])
