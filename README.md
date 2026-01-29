# FireScope Edge (`firescope-edge`)

Edge (on-device) computer-vision service for **smoke and fire classification** designed to run on a **Raspberry Pi** (or any Linux box with a camera).

It exposes a small **FastAPI** HTTP API for health/telemetry, provides an optional **MJPEG debug stream**, and can persist **alert snapshots + metadata** (and optionally send Telegram notifications) when smoke/fire is detected consistently over time.

## What It Does

- Captures frames from a camera (USB `/dev/video*` index or an RTSP URL).
- Runs a YOLO model (default: `models/best.onnx`) on each frame.
- Applies per-class confidence thresholds and a minimum-area filter.
- Uses a temporal filter (hits over a sliding window) to reduce false alarms.
- When an alert triggers, writes:
  - An annotated snapshot image (`.jpg`)
  - A JSON metadata file with detections and runtime info
- Optionally sends a Telegram message/photo.

## Repository Layout

- `app/` – FastAPI app + runtime loop + alert/notification logic
- `configs/firescope.yaml` – main configuration (camera, thresholds, temporal filter)
- `models/` – model weights (ignored by git; default path: `models/best.onnx`)
- `data/` – runtime output (ignored by git)
- `docs/` – additional documentation

## Quick Start (Development / Raspberry Pi)

Prereqs:
- Python 3.10+ recommended
- A camera accessible via V4L2 (USB cam) or an RTSP stream

Install:
- `python -m venv .venv`
- `. .venv/bin/activate`
- `pip install -r requirements.txt`

Configure:
- Edit `configs/firescope.yaml` (camera source, thresholds, output folder).
- Ensure your model weights exist at `runtime.weights_path` (default: `models/best.onnx`).
- (Optional) create `.env` for Telegram; see `.env.example`.

Run:
- `python -m app` (uses `configs/firescope.yaml` for host/port)
- or: `uvicorn app.main:app --host 0.0.0.0 --port 8000`

Open:
- Health: `http://<device-ip>:8000/health`
- Readiness/telemetry: `http://<device-ip>:8000/ready`
- Debug MJPEG stream: `http://<device-ip>:8000/debug/stream`

## Configuration

All settings live in `configs/firescope.yaml`.

Key options:
- `camera.source`
  - `0`, `1`, `2`, ... for local USB cameras
  - or an RTSP URL string (example: `"rtsp://user:pass@host:554/..."`)
- `runtime.weights_path` – path to `.onnx` (or `.pt`) weights
- `runtime.imgsz` – model input size (lower is faster)
- `runtime.backend` – currently informational; Ultralytics auto-selects based on the weights file
- `thresholds.conf_smoke` / `thresholds.conf_fire` – per-class confidence thresholds
- `thresholds.min_area` – minimum box area ratio (relative to frame area)
- `temporal_filter.m` – window size (frames)
- `temporal_filter.n_fire` / `temporal_filter.n_smoke` – required hits inside the window to trigger
- `temporal_filter.cooldown_s` – cooldown after any alert
- `storage.alerts_dir` – base directory where alerts are stored (snapshots + metadata)

Notes:
- Class IDs are expected to match `app/annotate.py`:
  - `0 = smoke`
  - `1 = fire`

## Alerts Output

When an alert is triggered, files are created under `storage.alerts_dir`:
- `snapshots/<alert_id>.jpg` – annotated image
- `metadata/<alert_id>.json` – metadata + detections

The API exposes:
- `GET /alerts/latest` – latest alert metadata (or `null`)
- `GET /alerts` – recent alerts (most recent first)

## Telegram Notifications (Optional)

Create a `.env` (do not commit it) with:
- `TELEGRAM_ENABLED=1`
- `TELEGRAM_BOT_TOKEN=...`
- `TELEGRAM_CHAT_ID=...`

Template: `.env.example`

Security note: if you ever committed a real token, rotate it immediately.

## Troubleshooting

- `Camera open failed` / `Camera read failed`:
  - Verify a camera exists (`/dev/video0`, `/dev/video1`, ...)
  - Set `camera.source` to the correct index in `configs/firescope.yaml`
  - Check permissions (your user may need to be in the `video` group)
- `Model load failed`:
  - Confirm `runtime.weights_path` exists (default: `models/best.onnx`)
  - Ensure the model matches the expected class IDs (`0=smoke`, `1=fire`)
- Low FPS:
  - Reduce `runtime.imgsz` and `camera.width`/`camera.height`
  - Increase thresholds to reduce post-processing load

## More Docs

- `docs/ARCHITECTURE.md` – how the runtime is structured
- `docs/API.md` – HTTP endpoints and example responses
- `docs/DEPLOY_RPI.md` – Raspberry Pi deployment notes
