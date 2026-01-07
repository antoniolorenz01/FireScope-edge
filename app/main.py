from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.config import load_settings
from app.runtime import EdgeRuntime
from app.streaming import mjpeg_generator

app = FastAPI(title="FireScope Edge", version="0.1.0")

settings = load_settings()
runtime = EdgeRuntime(
    camera_source=settings.camera.source,
    width=settings.camera.width,
    height=settings.camera.height,
    mjpeg=settings.camera.mjpeg,
    weights_path=settings.runtime.weights_path,
    device=settings.runtime.device,
    imgsz=settings.runtime.imgsz,
    conf_smoke=settings.thresholds.conf_smoke,
    conf_fire=settings.thresholds.conf_fire,
    iou=settings.thresholds.iou,
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


# Endpoint para debug de stream MJPEG
@app.get("/debug/stream")
def debug_stream():
    def get_latest():
        frame = runtime.state.last_frame
        dets = runtime.state.last_detections or []
        return frame, dets

    return StreamingResponse(
        mjpeg_generator(get_latest, fps=10.0),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )