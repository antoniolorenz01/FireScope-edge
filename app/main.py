from fastapi import FastAPI

from app.config import load_settings

app = FastAPI(title="FireScope Edge", version="0.1.0")

settings = load_settings()

_state = {"model_loaded": False, "camera_ok": False}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    ready_flag = _state["model_loaded"] and _state["camera_ok"]
    return {"ready": ready_flag, **_state}


@app.get("/config")
def get_config():
    return {
        "app": settings.app.__dict__,
        "runtime": settings.runtime.__dict__,
        "camera": {
            **settings.camera.__dict__,
            "source_type": type(settings.camera.source).__name__,
        },
        "thresholds": settings.thresholds.__dict__,
        "temporal_filter": settings.temporal_filter.__dict__,
        "storage": settings.storage.__dict__,
    }