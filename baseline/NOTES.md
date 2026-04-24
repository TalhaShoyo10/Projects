# Baseline vs current project

This folder adds a minimal, end-to-end MNIST classifier pipeline **without modifying any existing repo files**.

## Why this exists

Your current model weights are OK, but the Streamlit canvas export in `ui/app.py` can produce near-blank images if the canvas returns floats in `[0, 1]` and you cast directly to `uint8`. That makes the API preprocessing fail to detect/crop the digit, and the model collapses to a constant class (often `1`).

## What’s simpler here

- **Training**: standard MNIST normalization + Adam, no heavy augmentation.
- **Inference**: preprocessing + normalization matches training exactly.
- **UI export**: handles float vs uint8 and alpha-composites onto white.

## How to run

1. Train baseline weights:
   - `python -m baseline.train_simple`

2. Run baseline API:
   - `python -m uvicorn baseline.api_simple:app --reload --port 8001`

3. Run baseline UI:
   - `streamlit run baseline/ui_simple.py`

## Compare the original failure mode

- `python baseline/compare_original_vs_fixed_export.py`
