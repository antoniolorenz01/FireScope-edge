from __future__ import annotations

from typing import List, Tuple

import numpy as np

from app.postprocess import Detection


def _clip_xyxy(xyxy: Tuple[float, float, float, float], w: int, h: int) -> Tuple[float, float, float, float]:
    # Clip box coordinates to image boundaries
    x1, y1, x2, y2 = xyxy
    x1 = float(np.clip(x1, 0.0, w - 1.0))
    y1 = float(np.clip(y1, 0.0, h - 1.0))
    x2 = float(np.clip(x2, 0.0, w - 1.0))
    y2 = float(np.clip(y2, 0.0, h - 1.0))

    # Ensure proper ordering after clipping
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return x1, y1, x2, y2


def map_letterbox_xyxy_to_original(
    detections: List[Detection],
    scale: float,
    pad: Tuple[int, int],
    orig_shape_hw: Tuple[int, int],
) -> List[Detection]:
    """
    Maps boxes from letterboxed model input space back to the original frame space.

    Assumptions:
    - The frame was resized with ratio `scale` and padded with `pad=(pad_x, pad_y)`.
    - Detection.xyxy is in model input pixel coordinates (e.g., 320x320).
    """
    pad_x, pad_y = pad
    orig_h, orig_w = orig_shape_hw

    mapped: List[Detection] = []
    inv = 1.0 / max(scale, 1e-9)

    for d in detections:
        x1, y1, x2, y2 = d.xyxy

        # Remove padding (letterbox) and unscale to original coordinates
        x1 = (x1 - pad_x) * inv
        y1 = (y1 - pad_y) * inv
        x2 = (x2 - pad_x) * inv
        y2 = (y2 - pad_y) * inv

        xyxy = _clip_xyxy((x1, y1, x2, y2), w=orig_w, h=orig_h)
        mapped.append(Detection(cls=d.cls, conf=d.conf, xyxy=xyxy))

    return mapped


def filter_by_min_area_ratio(
    detections: List[Detection],
    orig_shape_hw: Tuple[int, int],
    min_area_ratio: float,
) -> List[Detection]:
    """
    Filters detections by area ratio relative to the original frame area.
    Example: min_area_ratio=0.002 means 0.2% of the frame area.
    """
    orig_h, orig_w = orig_shape_hw
    frame_area = float(orig_w * orig_h)
    if frame_area <= 0:
        return []

    kept: List[Detection] = []
    for d in detections:
        x1, y1, x2, y2 = d.xyxy
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        area_ratio = (w * h) / frame_area

        if area_ratio >= float(min_area_ratio):
            kept.append(d)

    return kept
