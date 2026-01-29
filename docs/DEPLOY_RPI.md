# Raspberry Pi Deployment Notes

This project is designed to run on a Raspberry Pi-class device, but it is still “just” a Python service with OpenCV + Ultralytics + ONNXRuntime.

## Recommended Baseline

- Raspberry Pi OS (64-bit) or another Debian-based 64-bit distro
- Python 3.10+ recommended
- USB camera that appears as `/dev/video0` (V4L2)

## Install

From the repo root:
- `python -m venv .venv`
- `. .venv/bin/activate`
- `pip install -r requirements.txt`

## Camera Notes

### USB camera (recommended)

Use an integer index:
- `camera.source: 0`

If the service can’t open the camera:
- Check permissions (`ls -la /dev/video0`)
- Ensure your user is in the `video` group (log out/in after changes)
- Verify a camera is present: `v4l2-ctl --list-devices` (package `v4l-utils`)

### Raspberry Pi Camera Module

If you are using the CSI camera module, you typically need a V4L2-compatible path to feed OpenCV.
Common approaches:
- Enable the V4L2 compatibility layer for `libcamera`
- Or provide frames via another process that exposes `/dev/video*`

Exact steps depend on your OS image and camera stack.

## Performance Tips

- Keep `runtime.imgsz` small (default is `320`).
- Keep capture resolution small (`camera.width`/`camera.height`) for lower latency.
- If you see dropped frames, increase thresholds or the temporal filter window.

## Run on Boot (systemd)

This repo includes an empty `systemd/` folder; you can create a unit that runs:
- `uvicorn app.main:app --host 0.0.0.0 --port 8000`

Example outline (paths will depend on where you clone the repo):
1. Create a venv and install requirements.
2. Create a systemd unit pointing to the venv’s `uvicorn`.
3. Set `WorkingDirectory` to the repo root.
4. Set `EnvironmentFile` to a local `.env` if you enable Telegram.

If you want, I can add a ready-to-use `systemd` unit file tailored to your target path/user.
