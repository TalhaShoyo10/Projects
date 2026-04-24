import base64
import io
import requests
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ── Config ────────────────────────────────────────────────────────────────────

API_URL = "http://localhost:8000"


@st.cache_data(ttl=10)
def _cached_health(url: str) -> dict:
    r = requests.get(f"{url}/health", timeout=3)
    r.raise_for_status()
    return r.json()

st.set_page_config(
    page_title="MNIST CAPTCHA",
    page_icon="🔢",
    layout="centered"
)

# ── API helpers ───────────────────────────────────────────────────────────────

def api_get_captcha(noise_type: str, noise_level: float, length: int) -> dict | None:
    try:
        r = requests.get(f"{API_URL}/captcha", params={
            "noise_type": noise_type,
            "noise_level": noise_level,
            "display_size": 196,
            "length": length,
        }, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach the API. Make sure FastAPI is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_verify(captcha_id: str, answer: int | str) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/verify",
                          json={"captcha_id": captcha_id, "answer": answer},
                          timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Verify error: {e}")
        return None


def api_predict_image(image_bytes: bytes) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/predict",
                          files={"file": ("digit.png", image_bytes, "image/png")},
                          timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Predict error: {e}")
        return None


def b64_to_pil(b64_string: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_string)))


def canvas_image_to_png_bytes(image_data: np.ndarray) -> bytes:
    """
    Convert st_canvas image_data to PNG bytes.

    Handles float [0,1] vs uint8 [0,255] and alpha by compositing onto white.
    """
    arr = image_data
    if arr.dtype != np.uint8:
        arr_f = arr.astype(np.float32)
        if float(arr_f.max()) <= 1.0:
            arr_f = arr_f * 255.0
        arr = np.clip(arr_f, 0.0, 255.0).astype(np.uint8)

    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Unexpected canvas image shape: {arr.shape}")

    if arr.shape[2] == 3:
        rgba = np.concatenate([arr, np.full((arr.shape[0], arr.shape[1], 1), 255, dtype=np.uint8)], axis=2)
    else:
        rgba = arr

    pil_rgba = Image.fromarray(rgba, mode="RGBA")
    # Composite onto black so the saved PNG matches the drawing canvas theme
    # (black background + white strokes).
    black_bg = Image.new("RGBA", pil_rgba.size, (0, 0, 0, 255))
    composed = Image.alpha_composite(black_bg, pil_rgba).convert("L")

    buf = io.BytesIO()
    composed.save(buf, format="PNG")
    return buf.getvalue()


def reset_challenge():
    st.session_state.captcha_data   = None
    st.session_state.captcha_used   = False
    st.session_state.captcha_result = None
    st.session_state.cnn_result     = None
    st.session_state.digit_images   = {}
    st.session_state.canvas_key     = st.session_state.get("canvas_key", 0) + 1
    st.session_state.pop("canvas_image_data", None)


# ── Session state defaults ────────────────────────────────────────────────────

for key, default in {
    "captcha_data":   None,
    "captcha_used":   False,
    "captcha_result": None,
    "cnn_result":     None,
    "canvas_key":     0,
    "canvas_image_data": None,
    "captcha_length": 4,
    "digit_images":   {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🔢 MNIST CAPTCHA")

# Sidebar
st.sidebar.header("⚙️ Noise Settings")
noise_type  = st.sidebar.selectbox("Noise Type", ["gaussian", "salt_pepper"],
                                   format_func=lambda x: x.replace("_", " ").title())
noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.3, 0.05)
captcha_length = st.sidebar.slider(
    "CAPTCHA Length",
    min_value=1,
    max_value=6,
    value=int(st.session_state.get("captcha_length", 4)),
    step=1,
)
st.session_state["captcha_length"] = int(captcha_length)

st.sidebar.markdown("---")
st.sidebar.markdown("**API status**")
try:
    health = _cached_health(API_URL)
    st.sidebar.success(f"✅ API online — {health['status']}")
except Exception:
    st.sidebar.error("❌ API offline")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**How it works**
1. Click **New Challenge** to get a noisy digit
2. Draw the digit you think you see on the canvas
3. Click **Submit Drawing** — the CNN reads your drawing and submits it as your CAPTCHA answer
4. See if the CNN understood you!
""")

# Tabs
tab_challenge, tab_docs = st.tabs([
    "🎲 CAPTCHA Challenge",
    "📄 Integration Docs",
])


# ── Tab 1 : CAPTCHA Challenge (merged draw + verify flow) ─────────────────────
with tab_challenge:

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("🔄 New Challenge"):
            reset_challenge()
            st.session_state.captcha_data = api_get_captcha(
                noise_type,
                noise_level,
                st.session_state["captcha_length"],
            )

    if st.session_state.captcha_data is None:
        st.info("Click **New Challenge** to start.")

    else:
        data        = st.session_state.captcha_data
        captcha_pil = b64_to_pil(data["image_base64"])
        captcha_len = int(data.get("length", st.session_state.get("captcha_length", 4)))
        st.session_state["captcha_length"] = captcha_len

        # ── Before submission: show noisy digit + canvas side by side ────────
        if not st.session_state.captcha_used:
            st.markdown("**Step 1 — Look at the noisy digit on the left**")
            st.markdown("**Step 2 — Draw each digit (left → right) on the canvases**")
            st.markdown("**Step 3 — Hit Submit Drawing**")
            st.markdown("---")

            col_img, col_canvas = st.columns([1, 1])

            with col_img:
                st.markdown("#### Noisy CAPTCHA")
                st.image(captcha_pil, width=min(196 * captcha_len, 900))
                st.caption(f"{data['noise_type']} noise @ {data['noise_level']}")

            with col_canvas:
                # Keep the drawing canvas visually aligned with the CAPTCHA image by
                # giving the right column the same heading height as the left.
                st.markdown("#### Your Drawings")
                st.caption("White ink on black canvas — draw each digit separately")

                digit_images: dict[int, np.ndarray] = dict(st.session_state.get("digit_images", {}))
                per_row = 3 if captcha_len > 3 else captcha_len

                for start in range(0, captcha_len, per_row):
                    cols = st.columns(per_row)
                    for j in range(per_row):
                        i = start + j
                        if i >= captcha_len:
                            continue
                        with cols[j]:
                            st.caption(f"Digit {i+1}")
                            canvas_result = st_canvas(
                                fill_color="black",
                                stroke_width=12,
                                stroke_color="white",
                                background_color="black",
                                height=140,
                                width=140,
                                drawing_mode="freedraw",
                                update_streamlit=True,
                                key=f"canvas_{st.session_state['canvas_key']}_{i}",
                            )

                            if canvas_result.image_data is None:
                                continue

                            img = canvas_result.image_data
                            rgb = img[:, :, :3].astype(np.float32)
                            if float(rgb.max()) <= 1.0:
                                rgb = rgb * 255.0

                            gray = rgb.mean(axis=2)
                            corners = np.array(
                                [gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]],
                                dtype=np.float32,
                            )
                            bg_is_white = float(corners.mean()) > 127.0
                            non_bg = (gray < 235.0) if bg_is_white else (gray > 20.0)
                            has_stroke = int(non_bg.sum()) > 25
                            if has_stroke:
                                digit_images[i] = img

                st.session_state["digit_images"] = digit_images

            col_submit, col_clear = st.columns([1, 1])

            with col_clear:
                if st.button("🗑️ Clear Drawing"):
                    st.session_state["digit_images"] = {}
                    st.session_state["canvas_key"] += 1
                    st.rerun()

            with col_submit:
                if st.button("✅ Submit Drawing", type="primary"):
                    digit_images: dict[int, np.ndarray] = dict(st.session_state.get("digit_images", {}))
                    missing = [str(i + 1) for i in range(captcha_len) if i not in digit_images]
                    if missing:
                        st.warning(f"Draw digit(s): {', '.join(missing)}")
                    else:
                        predictions = []
                        predicted_digits = []

                        for i in range(captcha_len):
                            image_bytes = canvas_image_to_png_bytes(digit_images[i])
                            cnn = api_predict_image(image_bytes)
                            if not cnn:
                                st.stop()
                            predictions.append(cnn)
                            predicted_digits.append(int(cnn["predicted_digit"]))

                        predicted_answer = "".join(str(d) for d in predicted_digits)
                        verify = api_verify(data["captcha_id"], predicted_answer)

                        st.session_state.cnn_result = {
                            "predicted_answer": predicted_answer,
                            "digits": predictions,
                        }
                        st.session_state.captcha_result = verify
                        st.session_state.captcha_used = True
                        st.rerun()

        # ── After submission: show results ───────────────────────────────────
        else:
            cnn    = st.session_state.cnn_result
            result = st.session_state.captcha_result

            if cnn and result:
                st.markdown("---")

                # Outcome banner
                if result["success"]:
                    st.success("✅ CAPTCHA passed! Your drawing was correct.")
                else:
                    st.error(
                        f"❌ CAPTCHA failed. "
                        f"Your drawing was read as **{cnn['predicted_answer']}** "
                        f"but the CAPTCHA was **{result['true_label']}**."
                    )

                st.markdown("---")

                # Side by side: CAPTCHA image vs CNN result
                col_img, col_result = st.columns([1, 1])

                with col_img:
                    st.markdown("#### CAPTCHA digits")
                    st.image(captcha_pil, width=min(196 * captcha_len, 900))
                    st.caption(f"{data['noise_type']} noise @ {data['noise_level']}")

                with col_result:
                    st.markdown("#### CNN read your drawings as...")
                    st.metric("Predicted CAPTCHA", cnn["predicted_answer"])
                    digits = cnn.get("digits", [])
                    if digits:
                        rows = [
                            {
                                "position": i + 1,
                                "predicted_digit": d.get("predicted_digit"),
                                "confidence": d.get("confidence"),
                            }
                            for i, d in enumerate(digits)
                        ]
                        st.dataframe(rows, hide_index=True, width="stretch")

                st.info("Click **New Challenge** to try again.")


# ── Tab 2 : Integration Docs ──────────────────────────────────────────────────
with tab_docs:
    st.markdown("""
## Integrating this CAPTCHA into your service

Your service makes HTTP requests to the FastAPI backend.
No Python required — any language works.

---

### Step 1 — Get a CAPTCHA image

```http
GET /captcha?noise_type=gaussian&noise_level=0.3&length=4
```

**Response:**
```json
{
  "captcha_id": "3f7a2b1c-...",
  "image_base64": "<base64 PNG string>",
  "noise_type": "gaussian",
  "noise_level": 0.3,
  "length": 4
}
```

Decode `image_base64` and display it to your user.
Store `captcha_id` — you'll need it in Step 2.

---

### Step 2 — Verify the user's answer

```http
POST /verify
Content-Type: application/json

{
  "captcha_id": "3f7a2b1c-...",
  "answer": "5301"
}
```

**Response (correct):**
```json
{ "success": true, "true_label": null }
```

**Response (wrong):**
```json
{ "success": false, "true_label": 7 }
```

Use `success` to allow or block the user action.
Each `captcha_id` works **once only**.

---

### Step 3 (optional) — Direct image classification

```http
POST /predict
Content-Type: multipart/form-data

file: <image file>
```

**Response:**
```json
{
  "predicted_digit": 3,
  "confidence": 0.9821,
  "probabilities": [0.0, 0.0, 0.0, 0.98, ...]
}
```

---

### Example in Python

```python
import requests, base64
from PIL import Image
import io

# Step 1 — get CAPTCHA
r    = requests.get("http://your-domain/captcha", params={"noise_level": 0.3})
data = r.json()
captcha_id = data["captcha_id"]

# Decode and show image
img = Image.open(io.BytesIO(base64.b64decode(data["image_base64"])))
img.show()

# Step 2 — verify answer
answer = int(input("Enter digit: "))
result = requests.post("http://your-domain/verify",
                       json={"captcha_id": captcha_id, "answer": answer})
print(result.json())
```

---

### Interactive API docs

Visit **http://localhost:8000/docs** for the full auto-generated
documentation where you can test every endpoint live in your browser.
""")
