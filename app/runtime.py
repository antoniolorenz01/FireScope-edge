from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Optional
from app.onnx_model import OnnxModel, load_onnx_model
import cv2
import numpy as np
from app.vision import preprocess_bgr_to_nchw_float
from app.postprocess import parse_ultralytics_onnx_outputs, Detection

@dataclass
class RuntimeState:
    running: bool = False
    model_loaded: bool = False
    camera_ok: bool = False
    last_error: Optional[str] = None
    fps: float = 0.0
    last_infer_ms: float = 0.0
    last_outputs: int = 0
    smoke_count: int = 0
    fire_count: int = 0
    last_frame: np.ndarray | None = None
    last_detections: list[Detection] = None


class EdgeRuntime:
    def __init__(
        self,
        camera_source: Any,
        width: int,
        height: int,
        mjpeg: bool,
        weights_path: str,
        device: str,
        imgsz: int,
        conf_smoke: float,
        conf_fire: float,
        iou: float,
    ) -> None:
        self._camera_source = camera_source
        self._width = width
        self._height = height
        self._mjpeg = mjpeg

        self.state = RuntimeState()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._cap: Optional[cv2.VideoCapture] = None
        self._weights_path = weights_path
        self._device = device
        self._imgsz = imgsz
        self._conf_smoke = conf_smoke
        self._conf_fire = conf_fire
        self._iou = iou

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="firescope-runtime", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

        self._release_camera()
        self.state.running = False

    def _open_camera(self) -> bool:
        self._release_camera()

        cap = cv2.VideoCapture(self._camera_source, cv2.CAP_V4L2)
        if not cap.isOpened():
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._height))

        if self._mjpeg:
            # MJPG can reduce USB camera latency on some devices
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        self._cap = cap
        return True

    def _release_camera(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None

    def _load_model(self) -> bool:
        self._model = load_onnx_model(self._weights_path, prefer_gpu=self._device.lower() == "cuda")
        return True

    def _run(self) -> None:
        self.state.running = True
        self.state.last_error = None

        try:
            self.state.model_loaded = self._load_model()
        except Exception as e:
            self.state.model_loaded = False
            self.state.last_error = f"Model load failed: {e!r}"

        if not self.state.model_loaded:
            self.state.running = False
            return

        # Camera loop with basic FPS tracking and reconnection
        if not self._open_camera():
            self.state.camera_ok = False
            self.state.last_error = "Camera open failed"
        else:
            self.state.camera_ok = True

        frames = 0
        t0 = time.perf_counter()

        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                self.state.camera_ok = False
                if self._open_camera():
                    self.state.camera_ok = True
                    self.state.last_error = None
                else:
                    self.state.last_error = "Camera open failed (retrying)"
                    time.sleep(1.0)
                    continue

            ok, _frame = self._cap.read()
            if ok:
                self.state.last_frame = _frame
            if not ok:
                self.state.camera_ok = False
                self.state.last_error = "Camera read failed (reconnecting)"
                self._release_camera()
                time.sleep(0.5)
                continue

            # Real forward pass (no post-processing yet)
            if self._model is not None:
                try:
                    t_infer0 = time.perf_counter()
                    outputs = self._model.session.run(
                        None,
                        {self._model.input_name: preprocess_bgr_to_nchw_float(_frame, self._imgsz).blob},
                    )
                    self.state.last_infer_ms = (time.perf_counter() - t_infer0) * 1000.0
                    self.state.last_outputs = len(outputs)
                    self.state.last_error = None

                    detections = parse_ultralytics_onnx_outputs(
                        outputs=outputs,
                        conf_smoke=self._conf_smoke,
                        conf_fire=self._conf_fire,
                        iou_th=self._iou,
                    )
                    self.state.smoke_count = sum(1 for d in detections if d.cls == 0)
                    self.state.fire_count = sum(1 for d in detections if d.cls == 1)
                    self.state.last_detections = detections
                except Exception as e:
                    self.state.last_error = f"Inference failed: {e!r}"

            frames += 1
            dt = time.perf_counter() - t0
            if dt >= 1.0:
                self.state.fps = frames / dt
                frames = 0
                t0 = time.perf_counter()

        self._release_camera()
        self.state.running = False
