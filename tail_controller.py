"""
Tail actuation for the vision-reactive Pupper.

A 9 g hobby servo on a PCA9685 channel drives a cable-pull tail. A background
thread generates a sinusoidal angle sweep whose frequency is set by the main
loop from bounding-box area:

    f_wag = f_max * min(A / A_stop, 1)

If no person is being tracked the target frequency is 0, and the servo holds
at the idle (center) angle.

The PCA9685 import is optional — on a dev laptop the controller runs headless
(no actual PWM output) so the rest of the pipeline can still be exercised.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class TailConfig:
    channel: int = 15             # PCA9685 channel for the tail servo
    f_max_hz: float = 2.0         # wag rate at closest range (proposal: 2 Hz)
    a_stop: float = 60000.0       # bbox area matching max wag (should match ApproachConfig)
    angle_min_deg: float = 60.0   # servo sweep lower bound
    angle_max_deg: float = 120.0  # servo sweep upper bound
    update_hz: float = 50.0       # servo command update rate
    pwm_freq_hz: int = 50         # PCA9685 frame rate (standard 20 ms hobby servo)
    freq_smoothing: float = 0.2   # EMA coefficient on target frequency (0 = no smoothing)


class TailController:
    """Background servo driver with a thread-safe setpoint."""

    def __init__(self, config: Optional[TailConfig] = None, enable_hardware: bool = True):
        self.cfg = config or TailConfig()
        self._lock = threading.Lock()
        self._target_freq = 0.0       # Hz, commanded by set_from_area / set_idle
        self._current_freq = 0.0      # Hz, smoothed, used inside the thread
        self._phase = 0.0             # rad, accumulated so freq changes don't cause jumps
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._servo = None
        self._hw_available = False

        if enable_hardware:
            self._hw_available = self._init_hardware()

    # -- hardware --------------------------------------------------------
    def _init_hardware(self) -> bool:
        try:
            from adafruit_servokit import ServoKit  # type: ignore
        except Exception as e:  # ImportError or board-level failure
            print(f"[tail] PCA9685/servokit unavailable ({e}); running in dry-run mode.")
            return False
        try:
            kit = ServoKit(channels=16, frequency=self.cfg.pwm_freq_hz)
            self._servo = kit.servo[self.cfg.channel]
            # Center at startup so we don't jolt the tail.
            self._servo.angle = (self.cfg.angle_min_deg + self.cfg.angle_max_deg) / 2.0
            return True
        except Exception as e:
            print(f"[tail] ServoKit init failed ({e}); running in dry-run mode.")
            self._servo = None
            return False

    # -- public API ------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="tail-wagger", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        t = self._thread
        self._thread = None
        if t is not None:
            t.join(timeout=1.0)
        # Return to idle on shutdown.
        if self._servo is not None:
            try:
                self._servo.angle = (self.cfg.angle_min_deg + self.cfg.angle_max_deg) / 2.0
            except Exception:
                pass

    def set_from_area(self, area_px2: float) -> None:
        """Set target wag frequency from a bounding-box area."""
        ratio = max(0.0, min(1.0, area_px2 / self.cfg.a_stop)) if self.cfg.a_stop > 0 else 0.0
        with self._lock:
            self._target_freq = self.cfg.f_max_hz * ratio

    def set_idle(self) -> None:
        """No target visible — stop wagging."""
        with self._lock:
            self._target_freq = 0.0

    @property
    def hardware_available(self) -> bool:
        return self._hw_available

    # -- background loop -------------------------------------------------
    def _run(self) -> None:
        dt = 1.0 / max(1.0, self.cfg.update_hz)
        center = (self.cfg.angle_min_deg + self.cfg.angle_max_deg) / 2.0
        half = (self.cfg.angle_max_deg - self.cfg.angle_min_deg) / 2.0
        alpha = max(0.0, min(1.0, self.cfg.freq_smoothing))

        next_tick = time.monotonic()
        while self._running:
            with self._lock:
                target = self._target_freq

            # EMA smoothing avoids abrupt frequency jumps when A changes fast.
            if alpha > 0.0:
                self._current_freq += alpha * (target - self._current_freq)
            else:
                self._current_freq = target

            f = self._current_freq
            if f <= 1e-3:
                angle = center
                self._phase = 0.0
            else:
                self._phase += 2.0 * math.pi * f * dt
                if self._phase > 2.0 * math.pi:
                    self._phase -= 2.0 * math.pi
                angle = center + half * math.sin(self._phase)

            if self._servo is not None:
                try:
                    self._servo.angle = float(angle)
                except Exception:
                    # Don't let a transient PWM error take down the loop.
                    pass

            next_tick += dt
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()
