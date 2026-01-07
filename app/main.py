from fastapi import FastAPI

from app.config import load_settings
from app.runtime import EdgeRuntime

app = FastAPI(title="FireScope Edge", version="0.1.0")

settings = load_settings()
runtime = EdgeRuntime(
    camera_source=settings.camera.source,
    width=settings.camera.width,
    height=settings.camera.height,
    mjpeg=settings.camera.mjpeg,
)


@app.on_event("startup")
def on_startup() -> None:
    runtime.start()


@app.on_event("shutdown")
def on_shutdown() -> None:
    runtime.stop()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    s = runtime.state
    ready_flag = s.model_loaded and s.camera_ok
    return {
        "ready": ready_flag,
        "running": s.running,
        "model_loaded": s.model_loaded,
        "camera_ok": s.camera_ok,
        "fps": s.fps,
        "last_error": s.last_error,
    }