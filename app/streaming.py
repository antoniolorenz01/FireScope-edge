from __future__ import annotations

import time
from typing import Generator, Optional

import numpy as np

from app.annotate import draw_detections, encode_jpeg
from app.postprocess import Detection


def mjpeg_generator(
    get_frame_fn,
    fps: float = 10.0,
) -> Generator[bytes, None, None]:
    """
    MJPEG multipart stream generator.
    `get_frame_fn` must return (frame_bgr: np.ndarray | None, detections: list[Detection]).
    """
    boundary = b"--frame"
    frame_interval_s = 1.0 / max(1e-6, fps)

    while True:
        t0 = time.perf_counter()

        frame, dets = get_frame_fn()
        if frame is None:
            time.sleep(0.1)
            continue

        annotated = draw_detections(frame, dets)
        jpg = encode_jpeg(annotated, quality=80)

        yield boundary + b"\r\n"
        yield b"Content-Type: image/jpeg\r\n"
        yield f"Content-Length: {len(jpg)}\r\n\r\n".encode("utf-8")
        yield jpg + b"\r\n"

        dt = time.perf_counter() - t0
        sleep_s = frame_interval_s - dt
        if sleep_s > 0:
            time.sleep(sleep_s)
