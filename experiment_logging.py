"""CSV event logging for Vision-Reactive Pupper experiments."""

from __future__ import annotations

import csv
import os
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "data" / "experiments"

CSV_FIELDS = [
    "timestamp_iso",
    "timestamp_unix_s",
    "monotonic_s",
    "session_id",
    "trial_id",
    "side",
    "component",
    "event",
    "phase",
    "sequence",
    "frame_index",
    "peer",
    "stream_url",
    "robot_host",
    "robot_port",
    "model",
    "device",
    "image_width",
    "image_height",
    "detections_count",
    "target_detected",
    "target_confidence",
    "target_class_id",
    "target_x1",
    "target_y1",
    "target_x2",
    "target_y2",
    "target_center_x",
    "target_center_y",
    "target_area_px2",
    "target_area_ratio",
    "horizontal_error",
    "yaw_error_rad",
    "inference_time_s",
    "frame_age_s",
    "packet_bytes",
    "mode",
    "stale",
    "linear_x",
    "angular_z",
    "tail_amplitude_us",
    "tail_hw_available",
    "condition_distance_m",
    "ground_truth_person_present",
    "final_distance_m",
    "approach_success",
    "approach_duration_s",
    "notes",
]


def make_session_id(prefix: str = "session") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{socket.gethostname()}_{stamp}_{os.getpid()}"


class CsvEventLogger:
    """Append-only CSV logger with a stable schema."""

    def __init__(
        self,
        path: Path,
        *,
        session_id: str,
        side: str,
        component: str,
        trial_id: str = "",
    ):
        self.path = Path(path)
        self.session_id = session_id
        self.side = side
        self.component = component
        self.trial_id = trial_id
        self._lock = threading.Lock()
        self._file = None
        self._writer = None
        self._sequence = 0

        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists() and self.path.stat().st_size > 0
        self._file = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDS)
        if not exists:
            self._writer.writeheader()
            self._file.flush()

    def log(self, event: str, **fields: Any) -> None:
        now = time.time()
        row = {name: "" for name in CSV_FIELDS}
        row.update(
            {
                "timestamp_iso": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                "timestamp_unix_s": f"{now:.6f}",
                "monotonic_s": f"{time.monotonic():.6f}",
                "session_id": self.session_id,
                "trial_id": fields.pop("trial_id", self.trial_id),
                "side": self.side,
                "component": self.component,
                "event": event,
            }
        )

        with self._lock:
            self._sequence += 1
            row["sequence"] = self._sequence
            unknown = {}
            for key, value in fields.items():
                if key in row:
                    row[key] = _format_value(value)
                else:
                    unknown[key] = value
            if unknown:
                notes = row.get("notes") or ""
                extra = "; ".join(f"{key}={_format_value(value)}" for key, value in sorted(unknown.items()))
                row["notes"] = f"{notes}; {extra}" if notes else extra
            assert self._writer is not None
            self._writer.writerow(row)
            assert self._file is not None
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.flush()
                self._file.close()
                self._file = None
                self._writer = None


def build_logger(
    *,
    side: str,
    component: str,
    session_id: Optional[str] = None,
    trial_id: str = "",
    log_dir: Optional[Path] = None,
    csv_log: Optional[Path] = None,
) -> CsvEventLogger:
    session = session_id or make_session_id(component)
    if csv_log is None:
        directory = Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR
        csv_log = directory / f"{session}_{side}_{component}.csv"
    return CsvEventLogger(Path(csv_log), session_id=session, side=side, component=component, trial_id=trial_id)


def target_fields(person: Optional[Mapping[str, Any]], image_width: float = 0.0, image_height: float = 0.0) -> dict[str, Any]:
    if not person:
        return {"target_detected": False}

    box = person.get("box") or ("", "", "", "")
    x1, y1, x2, y2 = list(box)[:4]
    width = float(x2) - float(x1)
    height = float(y2) - float(y1)
    area = max(0.0, width * height)
    frame_area = max(1.0, float(image_width) * float(image_height))
    center = person.get("center") or ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)

    return {
        "target_detected": True,
        "target_confidence": person.get("confidence", ""),
        "target_class_id": person.get("class_id", 0),
        "target_x1": x1,
        "target_y1": y1,
        "target_x2": x2,
        "target_y2": y2,
        "target_center_x": center[0],
        "target_center_y": center[1],
        "target_area_px2": area,
        "target_area_ratio": person.get("area_ratio", area / frame_area),
        "horizontal_error": person.get("horizontal_error", ""),
    }


def _format_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return value
