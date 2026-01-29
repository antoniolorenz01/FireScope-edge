# HTTP API

Default server: `http://<device-ip>:8000`

## `GET /health`

Basic liveness probe.

Example response:
```json
{"status":"ok"}
```

## `GET /ready`

Readiness + runtime telemetry.

Example response:
```json
{
  "ready": true,
  "running": true,
  "model_loaded": true,
  "camera_ok": true,
  "fps": 9.7,
  "last_error": null,
  "smoke_count": 0,
  "fire_count": 0,
  "fire_hits": 0,
  "smoke_hits": 0,
  "fire_triggered": false,
  "smoke_triggered": false,
  "cooldown_remaining_s": 0.0
}
```

Notes:
- `ready` becomes `true` only when the model is loaded and the camera is working.
- If something fails, `last_error` contains a short error string.

## `GET /alerts/latest`

Returns the most recent alert metadata or `null` if there are no alerts yet.

## `GET /alerts`

Returns a list of recent alerts (most recent first).

## `GET /debug/stream`

MJPEG multipart stream for debugging. It draws bounding boxes on the latest frame.

Open in a browser:
- `http://<device-ip>:8000/debug/stream`

Or with `curl`:
- `curl -v http://<device-ip>:8000/debug/stream`
