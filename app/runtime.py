from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2


@dataclass
class RuntimeState:
    running: bool = False
    model_loaded: bool = False
    camera_ok: bool = False
    last_error: Optional[str] = None
    fps: float = 0.0


class EdgeRuntime:
    def __init__(self, camera_source: Any, width: int, height: int, mjpeg: bool) -> None:
        self._camera_source = camera_source
        self._width = width
        self._height = height
        self._mjpeg = mjpeg

        self.state = RuntimeState()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._cap: Optional[cv2.VideoCapture] = None

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

        cap = cv2.VideoCapture(self._camera_source)
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
        # Placeholder: next step we will load ONNXRuntime here
        # For now, we just mark it as loaded so we can validate the lifecycle.
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
            if not ok:
                self.state.camera_ok = False
                self.state.last_error = "Camera read failed (reconnecting)"
                self._release_camera()
                time.sleep(0.5)
                continue

            # Placeholder: inference will go here in the next step
            frames += 1
            dt = time.perf_counter() - t0
            if dt >= 1.0:
                self.state.fps = frames / dt
                frames = 0
                t0 = time.perf_counter()

        self._release_camera()
        self.state.running = False
