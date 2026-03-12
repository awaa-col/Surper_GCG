from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def default_log_path(experiment_name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return str(Path("results") / "logs" / f"{experiment_name}_{stamp}.jsonl")


@dataclass
class JsonlRunLogger:
    path: str

    def __post_init__(self) -> None:
        target = Path(self.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._fp = target.open("a", encoding="utf-8")

    def log(self, event: str, **payload: Any) -> None:
        row: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
        }
        row.update(payload)
        self._fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fp.flush()

    def close(self) -> None:
        if hasattr(self, "_fp") and self._fp and not self._fp.closed:
            self._fp.close()

