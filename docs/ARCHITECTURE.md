# Architecture Overview

FireScope Edge is a small FastAPI service that runs a camera + inference loop in a background thread and exposes the current state over HTTP.

## High-Level Flow

1. **FastAPI startup**
   - `app/main.py` loads `configs/firescope.yaml`.
   - Creates `EdgeRuntime(...)`.
   - Starts the runtime thread on application startup.

2. **Frame capture**
   - `app/runtime.py` opens the camera using OpenCV:
     - `cv2.VideoCapture(camera_source, cv2.CAP_V4L2)`
   - `camera_source` can be an integer index (USB) or a URL (RTSP).

3. **Model inference**
   - The runtime loads weights via Ultralytics:
     - `YOLO(weights_path)`
   - Inference is executed with:
     - `model.predict(frame, imgsz=..., iou=..., device=..., verbose=False)`
   - The default configuration targets CPU inference (`device: "cpu"`) and `.onnx` weights.

4. **Filtering and state**
   - Per detection:
     - Per-class confidence thresholds are applied (`conf_smoke`, `conf_fire`).
     - A minimum-area ratio filter is applied (`min_area`).
   - Counts are computed:
     - `smoke_count`, `fire_count`
   - A temporal filter reduces flicker/false positives:
     - `TemporalFilter(m, n_fire, n_smoke, cooldown_s)`

5. **Alerts**
   - When the temporal filter triggers, a job is queued to `AlertManager`.
   - The alert writer thread:
     - stores an annotated snapshot (`.jpg`)
     - stores a JSON metadata file with detections and runtime info
   - If Telegram is enabled, a notification is queued and sent asynchronously.

## Concurrency Model

- **Runtime loop thread**: reads frames and runs inference.
- **Alert writer thread**: persists snapshots + metadata (best-effort; queue drops if overloaded).
- **Optional Telegram thread**: sends message/photo with retries and exponential backoff.

The runtime loop never blocks on alert writing or Telegram sending.

## Key Files

- `app/main.py` – FastAPI app + endpoints
- `app/runtime.py` – camera + inference loop and live runtime state
- `app/alerts_manager.py` – alert persistence + recent/latest cache
- `app/notifiers/telegram.py` – Telegram sender (stdlib-only)
- `configs/firescope.yaml` – configuration
