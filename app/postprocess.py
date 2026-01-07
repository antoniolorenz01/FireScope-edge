from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class Detection:
    cls: int
    conf: float
    xyxy: Tuple[float, float, float, float]


def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    # xywh: [x_center, y_center, w, h]
    x, y, w, h = xywh.T
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return np.stack([x1, y1, x2, y2], axis=1)


def _iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    # box: (4,), boxes: (N,4)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_box = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area_boxes = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])

    union = area_box + area_boxes - inter + 1e-9
    return inter / union


def nms_xyxy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_th: float,
) -> List[int]:
    # Classic NMS, returns kept indices
    if boxes.size == 0:
        return []

    idxs = np.argsort(scores)[::-1]
    keep: List[int] = []

    while idxs.size > 0:
        i = int(idxs[0])
        keep.append(i)

        if idxs.size == 1:
            break

        rest = idxs[1:]
        ious = _iou_xyxy(boxes[i], boxes[rest])
        idxs = rest[ious < iou_th]

    return keep


def parse_ultralytics_onnx_outputs(
    outputs: List[np.ndarray],
    conf_smoke: float,
    conf_fire: float,
    iou_th: float,
) -> List[Detection]:
    """
    Attempts to parse common Ultralytics-style ONNX output shapes.

    Supported patterns (common):
    - (1, 6, N): [x, y, w, h, conf, cls]
    - (1, N, 6): same fields

    Coordinates are expected in the model input pixel space (imgsz x imgsz).
    """
    if not outputs:
        return []

    out = outputs[0]
    out = np.asarray(out)

    # Squeeze batch dimension if present
    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]

    # Normalize to (N,6)
    if out.ndim == 2 and out.shape[0] == 6:
        out = out.T  # (6,N) -> (N,6)
    elif out.ndim == 2 and out.shape[1] == 6:
        pass
    else:
        # Unknown format
        return []

    xywh = out[:, 0:4].astype(np.float32)
    conf = out[:, 4].astype(np.float32)
    cls = out[:, 5].astype(np.int32)

    # Per-class confidence thresholds
    per_det_th = np.where(cls == 0, conf_smoke, conf_fire)
    mask = conf >= per_det_th
    xywh = xywh[mask]
    conf = conf[mask]
    cls = cls[mask]

    if xywh.size == 0:
        return []

    boxes = _xywh_to_xyxy(xywh)

    # Apply NMS per class
    detections: List[Detection] = []
    for k in (0, 1):
        m = cls == k
        if not np.any(m):
            continue

        b = boxes[m]
        s = conf[m]
        keep = nms_xyxy(b, s, iou_th=iou_th)

        for j in keep:
            detections.append(
                Detection(
                    cls=int(k),
                    conf=float(s[j]),
                    xyxy=(float(b[j, 0]), float(b[j, 1]), float(b[j, 2]), float(b[j, 3])),
                )
            )

    return detections
