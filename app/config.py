from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AppConfig:
    host: str
    port: int


@dataclass(frozen=True)
class RuntimeConfig:
    backend: str
    weights_path: str
    imgsz: int
    device: str


@dataclass(frozen=True)
class CameraConfig:
    source: Any  # int (USB index) or str (RTSP URL)
    width: int
    height: int
    mjpeg: bool


@dataclass(frozen=True)
class ThresholdsConfig:
    conf_smoke: float
    conf_fire: float
    iou: float
    min_area: float


@dataclass(frozen=True)
class TemporalFilterConfig:
    m: int
    n_fire: int
    n_smoke: int
    cooldown_s: float


@dataclass(frozen=True)
class StorageConfig:
    snapshots_dir: str


@dataclass(frozen=True)
class Settings:
    app: AppConfig
    runtime: RuntimeConfig
    camera: CameraConfig
    thresholds: ThresholdsConfig
    temporal_filter: TemporalFilterConfig
    storage: StorageConfig


def load_settings(path: str = "configs/firescope.yaml") -> Settings:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path.resolve()}")

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Settings(
        app=AppConfig(**raw["app"]),
        runtime=RuntimeConfig(**raw["runtime"]),
        camera=CameraConfig(**raw["camera"]),
        thresholds=ThresholdsConfig(**raw["thresholds"]),
        temporal_filter=TemporalFilterConfig(**raw["temporal_filter"]),
        storage=StorageConfig(**raw["storage"]),
    )
