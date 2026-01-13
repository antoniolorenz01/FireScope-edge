from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from app.alerts_manager import AlertManager, _AlertJob
from app.postprocess import Detection
from app.temporal_filter import TemporalFilter


@dataclass
class RuntimeState:
    running: bool = False
    model_loaded: bool = False
    camera_ok: bool = False
    last_error: Optional[str] = None

    fps: float = 0.0
    last_infer_ms: float = 0.0
    last_outputs: int = 0  # number of detections kept after filtering

    smoke_count: int = 0
    fire_count: int = 0

    last_frame: np.ndarray | None = None
    last_detections: list[Detection] | None = None

    fire_hits: int = 0
    smoke_hits: int = 0
    fire_triggered: bool = False
    smoke_triggered: bool = False
    cooldown_remaining_s: float = 0.0

    last_alert_id: str | None = None
    last_alert_kind: str | None = None


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
        min_area: float,
        m: int,
        n_fire: int,
        n_smoke: int,
        cooldown_s: float,
        alerts_dir: str,
        telegram: Optional[object] = None,
    ) -> None:
        self._camera_source = camera_source
        self._width = int(width)
        self._height = int(height)
        self._mjpeg = bool(mjpeg)

        self._weights_path = str(weights_path)
        self._device = str(device)
        self._imgsz = int(imgsz)

        self._conf_smoke = float(conf_smoke)
        self._conf_fire = float(conf_fire)
        self._iou = float(iou)
        self._min_area_ratio = float(min_area)

        self._temporal = TemporalFilter(m=m, n_fire=n_fire, n_smoke=n_smoke, cooldown_s=cooldown_s)

        self._alerts = AlertManager(base_dir=alerts_dir, keep_last=50, telegram=telegram)
        self._cooldown_s = float(cooldown_s)
        self._model_path = str(weights_path)

        self.state = RuntimeState()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._cap: Optional[cv2.VideoCapture] = None
        self._model: Optional[YOLO] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._alerts.start()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="firescope-runtime", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

        self._alerts.stop()
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
        # Ultralytics can load .onnx directly and will handle preprocessing + postprocessing.
        self._model = YOLO(self._weights_path)
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

            ok, frame = self._cap.read()
            if ok:
                self.state.last_frame = frame
            else:
                self.state.camera_ok = False
                self.state.last_error = "Camera read failed (reconnecting)"
                self._release_camera()
                time.sleep(0.5)
                continue

            if self._model is None:
                self.state.last_error = "Model not loaded"
                continue

            try:
                t_infer0 = time.perf_counter()

                # Ultralytics returns boxes already mapped to the original frame coordinates.
                res = self._model.predict(
                    frame,
                    imgsz=self._imgsz,
                    iou=self._iou,
                    device=self._device,
                    verbose=False,
                )[0]

                infer_ms = (time.perf_counter() - t_infer0) * 1000.0
                self.state.last_infer_ms = infer_ms
                self.state.last_error = None

                H, W = frame.shape[:2]
                min_area_px = self._min_area_ratio * float(W * H)

                detections: list[Detection] = []
                if res.boxes is not None and len(res.boxes) > 0:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    conf = res.boxes.conf.cpu().numpy()
                    cls = res.boxes.cls.cpu().numpy().astype(int)

                    for (x1, y1, x2, y2), score, c in zip(xyxy, conf, cls):
                        score_f = float(score)
                        c_i = int(c)

                        # Per-class confidence thresholds
                        if c_i == 0 and score_f < self._conf_smoke:
                            continue
                        if c_i == 1 and score_f < self._conf_fire:
                            continue

                        # Minimum area filter in original frame pixel space
                        area_px = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
                        if area_px < min_area_px:
                            continue

                        detections.append(
                            Detection(
                                cls=c_i,
                                conf=score_f,
                                xyxy=(float(x1), float(y1), float(x2), float(y2)),
                            )
                        )

                self.state.last_outputs = len(detections)
                self.state.last_detections = detections
                self.state.smoke_count = sum(1 for d in detections if d.cls == 0)
                self.state.fire_count = sum(1 for d in detections if d.cls == 1)

                fire_present = self.state.fire_count > 0
                smoke_present = self.state.smoke_count > 0

                t_status = self._temporal.update(fire_present=fire_present, smoke_present=smoke_present)
                self.state.fire_hits = t_status.fire_hits
                self.state.smoke_hits = t_status.smoke_hits
                self.state.fire_triggered = t_status.fire_triggered
                self.state.smoke_triggered = t_status.smoke_triggered
                self.state.cooldown_remaining_s = t_status.cooldown_remaining_s

                if self.state.fire_triggered or self.state.smoke_triggered:
                    kind = "fire" if self.state.fire_triggered else "smoke"
                    ts_unix = time.time()

                    # Copy frame and detections for the writer thread
                    frame_copy = frame.copy()
                    dets_copy = list(detections)

                    self._alerts.enqueue(
                        _AlertJob(
                            kind=kind,
                            ts_unix=ts_unix,
                            frame_bgr=frame_copy,
                            detections=dets_copy,
                            fire_hits=self.state.fire_hits,
                            smoke_hits=self.state.smoke_hits,
                            cooldown_s=self._cooldown_s,
                            smoke_count=self.state.smoke_count,
                            fire_count=self.state.fire_count,
                            model_path=self._model_path,
                            imgsz=self._imgsz,
                        )
                    )

                    latest = self._alerts.latest()
                    if latest is not None:
                        self.state.last_alert_id = latest.alert_id
                        self.state.last_alert_kind = latest.kind

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
