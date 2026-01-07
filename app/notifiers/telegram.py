from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class TelegramConfig:
    token: str
    chat_id: str
    send_photo: bool = True
    timeout_s: float = 7.0
    max_retries: int = 6
    base_backoff_s: float = 1.0


@dataclass(frozen=True)
class TelegramJob:
    kind: str
    ts_unix: float
    message: str
    photo_path: Optional[str] = None


class TelegramNotifier:
    def __init__(self, cfg: TelegramConfig) -> None:
        self._cfg = cfg
        self._q: "queue.Queue[TelegramJob]" = queue.Queue(maxsize=200)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.last_error: Optional[str] = None
        self.sent_count: int = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, name="firescope-telegram", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def enqueue(self, job: TelegramJob) -> None:
        # Best-effort: never block realtime code.
        try:
            self._q.put_nowait(job)
        except queue.Full:
            return

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._q.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._send_with_retries(job)
            finally:
                self._q.task_done()

    def _send_with_retries(self, job: TelegramJob) -> None:
        backoff = float(self._cfg.base_backoff_s)

        for attempt in range(1, int(self._cfg.max_retries) + 1):
            try:
                if self._cfg.send_photo and job.photo_path:
                    self._send_photo(job)
                else:
                    self._send_message(job.message)

                self.last_error = None
                self.sent_count += 1
                return
            except Exception as e:
                self.last_error = f"Telegram send failed (attempt {attempt}/{self._cfg.max_retries}): {e!r}"
                time.sleep(min(backoff, 30.0))
                backoff *= 2.0

    def _send_message(self, text: str) -> None:
        url = f"https://api.telegram.org/bot{self._cfg.token}/sendMessage"
        payload = {
            "chat_id": self._cfg.chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            url=url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self._cfg.timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            if resp.status < 200 or resp.status >= 300:
                raise RuntimeError(f"Telegram sendMessage HTTP {resp.status}: {body}")

    def _send_photo(self, job: TelegramJob) -> None:
        url = f"https://api.telegram.org/bot{self._cfg.token}/sendPhoto"

        photo_path = Path(job.photo_path)
        if not photo_path.exists():
            # Fallback to text if file is missing
            self._send_message(job.message)
            return

        boundary = f"----firescope-{int(time.time()*1000)}"
        content_type = f"multipart/form-data; boundary={boundary}"

        fields = {
            "chat_id": self._cfg.chat_id,
            "caption": job.message,
        }

        body = _encode_multipart_formdata(boundary, fields, file_field="photo", file_path=photo_path)

        req = urllib.request.Request(
            url=url,
            data=body,
            headers={"Content-Type": content_type, "Content-Length": str(len(body))},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self._cfg.timeout_s) as resp:
            resp_body = resp.read().decode("utf-8", errors="ignore")
            if resp.status < 200 or resp.status >= 300:
                raise RuntimeError(f"Telegram sendPhoto HTTP {resp.status}: {resp_body}")


def _encode_multipart_formdata(
    boundary: str,
    fields: dict,
    file_field: str,
    file_path: Path,
) -> bytes:
    # Minimal multipart encoder (stdlib-only)
    crlf = "\r\n"
    lines: list[bytes] = []

    for k, v in fields.items():
        lines.append(f"--{boundary}{crlf}".encode("utf-8"))
        lines.append(f'Content-Disposition: form-data; name="{k}"{crlf}{crlf}'.encode("utf-8"))
        lines.append(f"{v}{crlf}".encode("utf-8"))

    filename = file_path.name
    file_bytes = file_path.read_bytes()

    lines.append(f"--{boundary}{crlf}".encode("utf-8"))
    lines.append(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"{crlf}'.encode("utf-8")
    )
    lines.append(f"Content-Type: image/jpeg{crlf}{crlf}".encode("utf-8"))
    lines.append(file_bytes)
    lines.append(crlf.encode("utf-8"))

    lines.append(f"--{boundary}--{crlf}".encode("utf-8"))

    return b"".join(lines)


def telegram_config_from_env(
    *,
    chat_id: Optional[str] = None,
    send_photo: bool = True,
    timeout_s: float = 7.0,
    max_retries: int = 6,
) -> TelegramConfig:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    cid = (chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")).strip()

    if not token:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN env var")
    if not cid:
        raise RuntimeError("Missing TELEGRAM_CHAT_ID env var")

    return TelegramConfig(
        token=token,
        chat_id=cid,
        send_photo=send_photo,
        timeout_s=timeout_s,
        max_retries=max_retries,
    )
