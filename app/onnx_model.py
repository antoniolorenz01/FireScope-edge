from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import onnxruntime as ort


@dataclass
class OnnxModel:
    session: ort.InferenceSession
    input_name: str
    input_shape: tuple


def load_onnx_model(weights_path: str, prefer_gpu: bool = False) -> OnnxModel:
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX weights not found: {path.resolve()}")

    providers = ["CPUExecutionProvider"]
    if prefer_gpu:
        # If CUDA provider is available in your build, it will be used.
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    sess = ort.InferenceSession(str(path), providers=providers)

    inp = sess.get_inputs()[0]
    input_name = inp.name
    input_shape = tuple(inp.shape)

    return OnnxModel(session=sess, input_name=input_name, input_shape=input_shape)
