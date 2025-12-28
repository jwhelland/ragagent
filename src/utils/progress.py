import threading
import time
from typing import Any, Dict

from loguru import logger


class ExtractionProgress:
    """Lightweight extraction progress tracker with heartbeat logging."""

    def __init__(self, total_chunks: int, heartbeat_seconds: float = 15.0) -> None:
        self.total_chunks = max(0, total_chunks)
        self.heartbeat_seconds = heartbeat_seconds
        self.stages: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def enable_stage(self, name: str, label: str) -> None:
        self.stages[name] = {
            "label": label,
            "total": self.total_chunks,
            "done": 0,
            "start": time.time(),
            "last_log": 0.0,
        }

    def update(self, name: str, *, increment: int = 1) -> None:
        stage = self.stages.get(name)
        if not stage:
            return
        with self._lock:
            stage["done"] = min(stage["done"] + increment, stage["total"])
            now = time.time()
            elapsed = max(now - stage["start"], 1e-6)
            rate = stage["done"] / elapsed
            remaining = max(stage["total"] - stage["done"], 0)
            eta_seconds = remaining / rate if rate > 0 else float("inf")

            should_log = (
                stage["done"] == stage["total"]
                or stage["last_log"] == 0
                or (now - stage["last_log"]) >= self.heartbeat_seconds
            )
            if should_log:
                percent = (stage["done"] / max(stage["total"], 1)) * 100
                logger.info(
                    "Extraction {}: {}/{} ({:.0f}%), {:.2f} cps, ETA {}",
                    stage["label"],
                    stage["done"],
                    stage["total"],
                    percent,
                    rate,
                    self._format_eta(eta_seconds),
                )
                stage["last_log"] = now

    def _format_eta(self, seconds: float) -> str:
        if seconds == float("inf"):
            return "unknown"
        minutes, secs = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h{minutes:02d}m"
        if minutes:
            return f"{minutes}m{secs:02d}s"
        return f"{secs}s"
