"""
Tail actuation for the vision-reactive Pupper.

The ESP32-S3 tail firmware generates the wag motion locally and accepts simple
USB-serial commands from the robot:

    CMD <amplitude_us> <wag_frequency_hz> <envelope_frequency_hz> <timeout_ms>

The robot side only needs to map person box area to wag amplitude and keep
refreshing that command. If commands stop arriving, the firmware recenters the
tail after the watchdog timeout.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class TailConfig:
    serial_port: str = "/dev/ttyACM0"
    baudrate: int = 115200
    startup_delay_s: float = 2.0
    write_interval_s: float = 0.1
    watchdog_timeout_s: float = 0.5
    wag_frequency_hz: float = 3.0
    envelope_frequency_hz: float = 0.5
    amplitude_epsilon_us: float = 8.0
    wag_start_area: float = 240000.0
    wag_full_area: float = 360000.0
    amplitude_min_us: float = 80.0
    amplitude_max_us: float = 250.0


class TailController:
    """USB-serial bridge for the ESP32-S3 tail controller."""

    def __init__(self, config: Optional[TailConfig] = None, enable_hardware: bool = True):
        self.cfg = config or TailConfig()
        self._serial = None
        self._hw_available = False
        self._last_write_time = 0.0
        self._last_amplitude_us: Optional[float] = None

        if enable_hardware:
            self._hw_available = self._init_hardware()

    def _init_hardware(self) -> bool:
        try:
            import serial  # type: ignore
        except Exception as e:
            print(f"[tail] pyserial unavailable ({e}); running in dry-run mode.")
            return False

        try:
            self._serial = serial.Serial(
                self.cfg.serial_port,
                self.cfg.baudrate,
                timeout=0.2,
                write_timeout=0.2,
            )
            time.sleep(max(0.0, self.cfg.startup_delay_s))
            self._write_command(0.0, force=True)
            return True
        except Exception as e:
            print(f"[tail] Serial tail init failed ({e}); running in dry-run mode.")
            self._serial = None
            return False

    def start(self) -> None:
        if self._hw_available:
            self._write_command(0.0, force=True)

    def stop(self) -> None:
        if self._serial is not None:
            try:
                self._write_command(0.0, force=True)
                self._serial.close()
            except Exception:
                pass
        self._serial = None

    def set_from_area(self, area_px2: float) -> None:
        """Set wag amplitude from person box area."""
        start = self.cfg.wag_start_area
        full = max(start, self.cfg.wag_full_area)

        if area_px2 <= start:
            amplitude_us = 0.0
        elif full <= start:
            amplitude_us = self.cfg.amplitude_max_us
        else:
            ratio = min(1.0, (area_px2 - start) / (full - start))
            span = self.cfg.amplitude_max_us - self.cfg.amplitude_min_us
            amplitude_us = self.cfg.amplitude_min_us + span * ratio

        self._write_command(amplitude_us)

    def set_idle(self) -> None:
        self._write_command(0.0)

    @property
    def hardware_available(self) -> bool:
        return self._hw_available

    def _write_command(self, amplitude_us: float, force: bool = False) -> None:
        amplitude_us = max(0.0, min(self.cfg.amplitude_max_us, float(amplitude_us)))
        now = time.monotonic()
        changed = (
            self._last_amplitude_us is None
            or abs(amplitude_us - self._last_amplitude_us) >= self.cfg.amplitude_epsilon_us
        )
        due = (now - self._last_write_time) >= max(0.0, self.cfg.write_interval_s)

        if not force and not changed and not due:
            return

        self._last_amplitude_us = amplitude_us
        self._last_write_time = now

        if self._serial is None:
            return

        timeout_ms = int(max(0.0, self.cfg.watchdog_timeout_s) * 1000.0)
        line = (
            f"CMD {amplitude_us:.1f} "
            f"{self.cfg.wag_frequency_hz:.3f} "
            f"{self.cfg.envelope_frequency_hz:.3f} "
            f"{timeout_ms}\n"
        )

        try:
            self._serial.write(line.encode("ascii"))
            self._serial.flush()
        except Exception:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None
            self._hw_available = False
