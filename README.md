# MNIST CAPTCHA

A CAPTCHA-inspired digit recognition service powered by a Convolutional Neural Network (CNN) trained on MNIST. Provides an interactive Streamlit UI for direct use and a FastAPI backend for external integration.

---

## Project Structure

```
mnist-captcha/
├── model/
│   ├── model.py          # CNN + FCNN architecture, count_parameters, load_model
│   ├── train.py          # Data loading, augmentation, training loop, best model saving
│   ├── evaluate.py       # Accuracy, F1, recall, confusion matrix, ROC, visualise_incorrect
│   └── weights/          # Created automatically after training — stores best_model.bin
├── api/
│   └── main.py           # FastAPI — GET /captcha, POST /verify, POST /predict
├── ui/
│   └── app.py            # Streamlit UI — calls FastAPI, no model loading
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
cd model
python train.py
```
Saves best weights to `model/weights/best_model.bin`.
If you pulled code changes that modify the CNN or preprocessing, retrain to regenerate compatible weights.

### 3. Run the API
```bash
cd api
uvicorn main:app --reload --port 8000
```
Interactive API docs available at `http://localhost:8000/docs`

### 4. Run the UI
```bash
cd ui
streamlit run app.py
```
UI available at `http://localhost:8501`

Run steps 3 and 4 in separate terminals simultaneously.

---

## How to Integrate

External services can integrate this CAPTCHA via HTTP — no Python required.

### Step 1 — Get a CAPTCHA image
```http
GET /captcha?noise_type=gaussian&noise_level=0.3&length=4
```

**Response:**
```json
{
  "captcha_id": "3f7a2b1c-9d4e-...",
  "image_base64": "<base64 PNG string>",
  "noise_type": "gaussian",
  "noise_level": 0.3,
  "length": 4
}
```
Decode `image_base64` and display it to your user. Store `captcha_id` for Step 2.

### Step 2 — Verify the user's answer
```http
POST /verify
Content-Type: application/json

{"captcha_id": "3f7a2b1c-9d4e-...", "answer": "5301"}
```

**Response (correct):**
```json
{"success": true, "true_label": null}
```

**Response (wrong):**
```json
{"success": false, "true_label": 7}
```

Each `captcha_id` is **single-use** — deleted after verification.

### Step 3 (optional) — Direct digit classification
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

### Python example
```python
import requests, base64
from PIL import Image
import io

# Step 1 — get CAPTCHA
r    = requests.get("http://your-domain/captcha", params={"noise_level": 0.3})
data = r.json()
captcha_id = data["captcha_id"]

img = Image.open(io.BytesIO(base64.b64decode(data["image_base64"])))
img.show()

# Step 2 — verify answer
answer = int(input("Enter digit: "))
result = requests.post("http://your-domain/verify",
                       json={"captcha_id": captcha_id, "answer": answer})
print(result.json())
```

---

## Model

| Property | Value |
|---|---|
| Architecture | CNN — Conv→ReLU→Pool × 2, then Linear×2 |
| Flatten size | 64 × 7 × 7 = 3,136 |
| Input | 28×28 grayscale images |
| Output | 10 class logits (digits 0–9) |
| Parameters | 225,034 |
| Loss | CrossEntropyLoss |
| Optimizer | Adam (lr=1e-3) |
| Target accuracy | ≥ 95% on MNIST test set |

### Augmentations (training only)
- Random rotation ±10°
- Random affine translation (up to 10% shift in x and y)

Validation and test sets receive no augmentation — clean `ToTensor()` only.

Note: Random erasing was considered during development (mentioned in code comments) but is not present in the final `train_transform` pipeline.

---

## Design Note

It is worth acknowledging that CAPTCHA verification does not inherently require a neural network — traditional approaches use deterministic checks (known fonts, fixed distortions, rule-based validation) and are simpler and more robust in production. The CNN here is a deliberate design choice to explore an applied use case for deep learning: the goal was to find an interesting way to incorporate a trained model into a working system, rather than to propose CNNs as the optimal CAPTCHA solution.

---

## ⚠️ Known Limitations & Security Considerations

This project is currently configured for **development and demonstration use only.**
The following limitations must be addressed before any production deployment.

### 1. No API Key Authentication

**Current state:** The API is completely open — any person or service that knows the URL can call any endpoint without identifying themselves.

**The risk:** There is no way to distinguish between a legitimate integrated service and a malicious one. Anyone can generate CAPTCHAs, flood the verify endpoint, or abuse the predict endpoint without restriction.

**Production fix:** Implement API key authentication. Each integrating service should be issued a unique secret key that must be sent with every request via the `Authorization` header:

```http
GET /captcha HTTP/1.1
Authorization: Bearer sk-your-secret-key
```

FastAPI's `Depends()` system makes this straightforward to add — a key verification function runs before every endpoint and rejects requests with invalid or missing keys.

### 2. CORS Is Set to Allow All Origins

**Current state:**
```python
allow_origins=["*"]
```
This allows any website — including malicious ones — to make browser-based requests to the API.

**The risk:** A malicious website could embed your API calls in their JavaScript and trick users' browsers into interacting with your service.

**Production fix:** Replace `"*"` with an explicit list of allowed origins:
```python
allow_origins=[
    "https://your-streamlit-app.com",
    "https://trusted-partner-site.com",
]
```

### 3. No Rate Limiting

**Current state:** There is no limit on how many requests a single client can make.

**The risk:** A single user or bot can send thousands of requests per second, crashing the server or running up compute costs. Since there is no authentication, there is also no way to identify or block the source.

**Production fix:** Add rate limiting — for example, 100 requests per minute per API key or IP address. Libraries like `slowapi` integrate directly with FastAPI:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/captcha")
@limiter.limit("100/minute")
def get_captcha(...):
    ...
```

### 4. CAPTCHA Store Is In-Memory Only

**Current state:**
```python
captcha_store: dict[str, int] = {}
```
Active CAPTCHAs are stored in a plain Python dictionary in the server's RAM.

**The risks:**
- If the server restarts, all active CAPTCHAs are lost — users mid-flow get a 404 on verify
- If you scale to multiple server instances, each has its own separate store — a CAPTCHA generated on instance A cannot be verified on instance B
- There is no expiry — an unused `captcha_id` lives in memory forever, which leaks memory over time

**Production fix:** Replace with Redis — a fast in-memory key-value store that is shared across all server instances and supports automatic key expiry:

```python
import redis
r = redis.Redis(host="localhost", port=6379)

# Store with 5 minute expiry
r.setex(captcha_id, 300, true_label)

# Retrieve and delete atomically
true_label = r.getdel(captcha_id)
```

### 5. Model Is Trained on MNIST Only

**Current state:** The model has only ever seen clean, centered, 28×28 MNIST digits.

**The risks:**
- Real-world handwriting (different sizes, angles, stroke widths, backgrounds) will produce unreliable predictions
- The model has a known tendency to predict digits with curves (like 2, 3, 8) incorrectly on out-of-distribution images
- A sufficiently motivated attacker could train their own model on MNIST to automatically solve your CAPTCHAs, since the digit set is small and well-known

**Mitigations:**
- Increase noise level to make automated solving harder
- Consider adding distortion augmentations at CAPTCHA generation time
- For a production CAPTCHA system, a more diverse and private training set is strongly recommended

### Summary Table

| Limitation | Risk Level | Fix |
|---|---|---|
| No API key auth | 🔴 High | Add `Depends()` key verification |
| CORS allows all origins | 🟡 Medium | Restrict to known domains |
| No rate limiting | 🔴 High | Add `slowapi` or similar |
| In-memory CAPTCHA store | 🟡 Medium | Replace with Redis |
| MNIST-only model | 🟡 Medium | Diversify training data |

---

## 🔭 Future Additions

### Dual-Model Mixed CAPTCHA (Digits + Letters)

Currently every CAPTCHA slot is a digit (0–9) recognised by a single CNN trained on MNIST. A planned extension is to introduce a second CNN trained on **EMNIST letters** (A–Z) and combine both models to generate mixed alphanumeric CAPTCHAs — more varied and harder to auto-solve.

#### How it would work

Two specialist models are loaded at startup, each responsible only for what it was trained on:

```python
DIGIT_MODEL  = load_model("weights/best_model_digits.bin",  DEVICE)  # MNIST  — digits 0-9
LETTER_MODEL = load_model("weights/best_model_letters.bin", DEVICE)  # EMNIST — letters A-Z
```

Each CAPTCHA slot is randomly assigned a type, with the hard constraint that **every CAPTCHA must contain at least one digit and at least one letter**:

```python
def generate_slot_types(length: int) -> list[str]:
    slots = ["digit", "letter"]               # guarantee one of each
    for _ in range(length - 2):
        slots.append(random.choice(["digit", "letter"]))
    random.shuffle(slots)
    return slots
# e.g. length=4 → ["letter", "digit", "letter", "digit"]
```

Each slot then draws from the appropriate dataset and routes through its specialist model:

```
slot type:   letter      digit      letter      digit
dataset:     EMNIST      MNIST      EMNIST      MNIST
label:         "A"        "5"        "G"        "2"
answer:              "A5G2"
```

Verification requires no changes — the answer is still a plain string comparison (`"A5G2" == "A5G2"`). The only meaningful change is that the answer string now contains both digit characters and uppercase letter characters.

#### Why this is interesting from a CNN perspective

Each model specialises on its own class distribution. Routing each tile to the correct specialist at inference time is a simple form of **mixture-of-experts** — rather than training one model to handle 36 classes (0–9 + A–Z) and accepting the accuracy trade-off, two focused models each handle what they know best. The mixing happens at the generation level, not inside any single model.

#### EMNIST is a drop-in replacement

EMNIST uses the same 28×28 grayscale format as MNIST and is available directly through torchvision with minimal code change:

```python
EMNIST_TEST = datasets.EMNIST(
    root="./data", split="letters", train=False,
    transform=transforms.ToTensor(), download=True
)
```

The existing noise functions, tile stitching logic, and base64 encoding pipeline all carry over unchanged.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Status message |
| GET | `/health` | Health check |
| GET | `/captcha` | Generate a noisy CAPTCHA image |
| POST | `/verify` | Verify a user's answer |
| POST | `/predict` | Direct digit classification |
| GET | `/docs` | Interactive API documentation |