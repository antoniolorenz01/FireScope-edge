from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class PreprocessResult:
    blob: np.ndarray
    scale: float
    pad: Tuple[int, int]


def letterbox(
    img: np.ndarray,
    new_size: int,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)

    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_w = new_size - nw
    pad_h = new_size - nh

    left = pad_w // 2
    top = pad_h // 2
    right = pad_w - left
    bottom = pad_h - top

    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, r, (left, top)


def preprocess_bgr_to_nchw_float(img_bgr: np.ndarray, imgsz: int) -> PreprocessResult:
    t0 = time.perf_counter()

    img, scale, pad = letterbox(img_bgr, imgsz)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = img_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
    x = np.expand_dims(x, axis=0)   # CHW -> NCHW

    _ = time.perf_counter() - t0
    return PreprocessResult(blob=x, scale=scale, pad=pad)