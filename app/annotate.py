from __future__ import annotations

from typing import List

import cv2
import numpy as np

from app.postprocess import Detection


CLASS_NAMES = {0: "smoke", 1: "fire"}


def draw_detections(img_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
    out = img_bgr.copy()

    COLORS = {
        0: (255, 102, 0),   # smoke: orange-blue (BGR)
        1: (0, 0, 255),     # fire: red (BGR)
    }

    for d in detections:
        x1, y1, x2, y2 = d.xyxy
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))

        label = f"{CLASS_NAMES.get(d.cls, str(d.cls))} {d.conf:.2f}"
        color = COLORS.get(d.cls, (0, 255, 0))

        cv2.rectangle(out, p1, p2, color, 2)
        cv2.putText(
            out,
            label,
            (p1[0], max(0, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return out


def encode_jpeg(img_bgr: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return buf.tobytes()
