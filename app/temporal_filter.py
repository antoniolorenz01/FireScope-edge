from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple


@dataclass
class TemporalStatus:
    fire_hits: int
    smoke_hits: int
    fire_triggered: bool
    smoke_triggered: bool
    cooldown_remaining_s: float


class TemporalFilter:
    def __init__(self, m: int, n_fire: int, n_smoke: int, cooldown_s: float) -> None:
        self.m = int(m)
        self.n_fire = int(n_fire)
        self.n_smoke = int(n_smoke)
        self.cooldown_s = float(cooldown_s)

        self._fire_hist: Deque[bool] = deque(maxlen=self.m)
        self._smoke_hist: Deque[bool] = deque(maxlen=self.m)

        self._last_alert_ts: float = 0.0

    def reset(self) -> None:
        self._fire_hist.clear()
        self._smoke_hist.clear()
        self._last_alert_ts = 0.0

    def update(self, fire_present: bool, smoke_present: bool, now_ts: Optional[float] = None) -> TemporalStatus:
        now = time.time() if now_ts is None else float(now_ts)

        self._fire_hist.append(bool(fire_present))
        self._smoke_hist.append(bool(smoke_present))

        fire_hits = int(sum(self._fire_hist))
        smoke_hits = int(sum(self._smoke_hist))

        cooldown_remaining = max(0.0, (self._last_alert_ts + self.cooldown_s) - now)
        in_cooldown = cooldown_remaining > 0.0

        fire_triggered = (fire_hits >= self.n_fire) and not in_cooldown
        smoke_triggered = (smoke_hits >= self.n_smoke) and not in_cooldown

        # Single cooldown for any alert
        if fire_triggered or smoke_triggered:
            self._last_alert_ts = now
            cooldown_remaining = self.cooldown_s

        return TemporalStatus(
            fire_hits=fire_hits,
            smoke_hits=smoke_hits,
            fire_triggered=fire_triggered,
            smoke_triggered=smoke_triggered,
            cooldown_remaining_s=cooldown_remaining,
        )