from fastapi import FastAPI

app = FastAPI(title="FireScope Edge", version="0.1.0")

_state = {"model_loaded": False, "camera_ok": False}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    ready = _state["model_loaded"] and _state["camera_ok"]
    return {"ready": ready, **_state}
