# Leaf Classifier Live (MPA-MLR Exercise 8)

Live on-device leaf classification running in the browser via WebAssembly (ONNX Runtime Web) from a trained PyTorch model.

🌐 **Live GitHub Pages URL:** [https://ejdamadam.github.io/mlr-ex08/](https://ejdamadam.github.io/mlr-ex08/)

---

## Overview

This repository contains a lightweight single-page web application (`index.html`) that classifies leaves live from a mobile phone's rear camera or from uploaded photos.

### Architecture & Pipeline
- **Model:** `leaf_model.onnx` (exported from PyTorch in Exercise 8, Task 2)
- **Runtime:** `onnxruntime-web` from CDN running locally in the browser via WebAssembly (WASM).
- **Privacy:** 100% on-device processing. No video frames, photographs, or telemetry are transmitted to any external server.
- **Classes:** `apple`, `cherry`, `chestnut`, `maple`.

### Preprocessing Pipeline (`model_info.json`)
The application implements the exact PyTorch preprocessing pipeline defined in the training notebook:
1. **Central Crop:** Extracts the central square (`cropSize = min(width, height)`) of the video frame or photo.
2. **Resize:** Rescales the cropped square to $96 \times 96$ pixels.
3. **Scale to $[0, 1]$:** Divides each pixel channel by $255.0$.
4. **Standardization:** Normalizes each channel with the dataset statistics:
   - Mean: $[0.485, 0.456, 0.406]$
   - Std: $[0.229, 0.224, 0.225]$
5. **Tensor Shape:** Formats the tensor into NCHW shape $1 \times 3 \times 96 \times 96$.
6. **Softmax:** Applies softmax to the raw logits to produce live class probabilities.

---

## Verification Control (`check_image.jpg`)
The application includes a built-in verification mechanism comparing predictions against `check_expected.json`:
- **apple:** $0.2922$ (Expected: $0.2922$)
- **cherry:** $0.0003$ (Expected: $0.0003$)
- **chestnut:** $0.7075$ (Expected: $0.7075$)
- **maple:** $0.0000$ (Expected: $0.0000$)

All probabilities match `check_expected.json` to 3+ decimal places.

---

## Mobile & Safari Compatibility
- Includes `playsinline` and `webkit-playsinline` attributes to allow inline video playback on iOS Safari without fullscreen takeover.
- Requests the rear camera via `facingMode: { ideal: "environment" }` with a manual flip button.
- Single-threaded WASM (`ort.env.wasm.numThreads = 1`) for guaranteed compatibility without requiring custom COOP/COEP isolation headers.
- QR code generated client-side directly in JavaScript (no third-party image tracking or services).
