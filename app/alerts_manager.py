from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, List, Optional
from app.notifiers.telegram import TelegramNotifier, TelegramJob

import numpy as np

from app.annotate import draw_detections, encode_jpeg
from app.postprocess import Detection


@dataclass(frozen=True)
class AlertMeta:
    alert_id: str
    kind: str  # "fire" | "smoke"
    ts_unix: float

    fire_hits: int
    smoke_hits: int
    cooldown_s: float

    smoke_count: int
    fire_count: int

    model_path: str
    imgsz: int

    snapshot_path: str
    metadata_path: str


@dataclass
class _AlertJob:
    kind: str
    ts_unix: float
    frame_bgr: np.ndarray
    detections: List[Detection]
    fire_hits: int
    smoke_hits: int
    cooldown_s: float
    smoke_count: int
    fire_count: int
    model_path: str
    imgsz: int


class AlertManager:
    def __init__(self, base_dir: str, keep_last: int = 50, telegram: Optional[TelegramNotifier] = None) -> None:
        self._base_dir = Path(base_dir)
        self._snap_dir = self._base_dir / "snapshots"
        self._meta_dir = self._base_dir / "metadata"

        self._snap_dir.mkdir(parents=True, exist_ok=True)
        self._meta_dir.mkdir(parents=True, exist_ok=True)

        self._q: "queue.Queue[_AlertJob]" = queue.Queue(maxsize=100)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._last: Optional[AlertMeta] = None
        self._recent: Deque[AlertMeta] = __import__("collections").deque(maxlen=keep_last)
        self._telegram = telegram

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        if self._telegram:
            self._telegram.start()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, name="firescope-alert-writer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._telegram:
            self._telegram.stop()

    def enqueue(self, job: _AlertJob) -> None:
        # Best-effort: if the queue is full, we drop the alert to protect realtime inference.
        try:
            self._q.put_nowait(job)
        except queue.Full:
            return

    def latest(self) -> Optional[AlertMeta]:
        with self._lock:
            return self._last

    def recent(self) -> List[AlertMeta]:
        with self._lock:
            return list(self._recent)

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._q.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                meta = self._persist(job)
                with self._lock:
                    self._last = meta
                    self._recent.appendleft(meta)
                if self._telegram:
                    msg = _format_telegram_message(meta, job)
                    self._telegram.enqueue(
                        TelegramJob(
                            kind=meta.kind,
                            ts_unix=meta.ts_unix,
                            message=msg,
                            photo_path=meta.snapshot_path,
                        )
                    )
            finally:
                self._q.task_done()


# Utilidad para formatear mensajes de Telegram
def _format_telegram_message(meta: AlertMeta, job: _AlertJob) -> str:
    kind_emoji = "🔥" if meta.kind == "fire" else "💨"
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(meta.ts_unix))
    return (
        f"{kind_emoji} {meta.kind.upper()} detected\n"
        f"Time: {ts}\n"
        f"Hits (fire/smoke): {meta.fire_hits}/{meta.smoke_hits}\n"
        f"Detections (fire/smoke): {meta.fire_count}/{meta.smoke_count}\n"
        f"Cooldown: {meta.cooldown_s:.0f}s\n"
        f"Alert ID: {meta.alert_id}"
    )


# Método persistente de AlertManager
def _persist(self, job: _AlertJob) -> AlertMeta:
    alert_id = f"{int(job.ts_unix)}_{job.kind}_{uuid.uuid4().hex[:8]}"

    snap_path = self._snap_dir / f"{alert_id}.jpg"
    meta_path = self._meta_dir / f"{alert_id}.json"

    # Store an annotated snapshot for human review
    annotated = draw_detections(job.frame_bgr, job.detections)
    snap_bytes = encode_jpeg(annotated, quality=85)
    snap_path.write_bytes(snap_bytes)

    meta = AlertMeta(
        alert_id=alert_id,
        kind=job.kind,
        ts_unix=float(job.ts_unix),
        fire_hits=int(job.fire_hits),
        smoke_hits=int(job.smoke_hits),
        cooldown_s=float(job.cooldown_s),
        smoke_count=int(job.smoke_count),
        fire_count=int(job.fire_count),
        model_path=str(job.model_path),
        imgsz=int(job.imgsz),
        snapshot_path=str(snap_path),
        metadata_path=str(meta_path),
    )

    payload = asdict(meta)
    payload["detections"] = [
        {"cls": d.cls, "conf": d.conf, "xyxy": list(d.xyxy)} for d in job.detections
    ]

    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return meta

# Asignar el método a la clase
AlertManager._persist = _persist
