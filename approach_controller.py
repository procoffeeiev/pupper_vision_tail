"""
Visual-servoing approach controller.

Turns a single Detection (bounding box + confidence) into a body-frame
velocity command. Follows the mapping described in the project proposal:

    yaw rate:  psi_dot  = -k_yaw * yaw_error_rad
    forward :  v_x = v_max                         when A < A_slow
               v_x = v_max * (A_stop - A) /       when A_slow <= A < A_stop
                            (A_stop - A_slow)
               v_x = 0                             when A >= A_stop

A small deadband on yaw error prevents twitching when the target is near-centered.
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Protocol


class _DetectionLike(Protocol):
    class_id: int
    confidence: float
    size_x: float
    size_y: float
    @property
    def normalized_x(self) -> float: ...
    @property
    def yaw_error_rad(self) -> float: ...


@dataclass
class ApproachConfig:
    person_class_id: int = 0      # COCO: 0 = person
    min_confidence: float = 0.5   # tau in the proposal
    k_yaw: float = 2.0            # rad/s per rad of yaw error
    yaw_deadband_rad: float = 0.05 # ignore |yaw_error| smaller than this
    search_angular_z: float = 0.6 # rad/s in-place search turn when no person is visible
    v_max: float = 0.35           # m/s forward cap
    a_slow: float = 20000.0       # bbox area (px^2) at which we start decelerating
    a_stop: float = 60000.0       # bbox area at which we fully stop
    max_angular_z: float = 1.5    # rad/s safety cap


class ApproachController:
    def __init__(self, config: Optional[ApproachConfig] = None):
        self.cfg = config or ApproachConfig()
        self._search_direction = 1.0

    def pick_target(self, detections: Iterable[_DetectionLike]) -> Optional[_DetectionLike]:
        """Return the highest-confidence person detection above threshold, or None."""
        best = None
        for d in detections:
            if d.class_id != self.cfg.person_class_id:
                continue
            if d.confidence < self.cfg.min_confidence:
                continue
            if best is None or d.confidence > best.confidence:
                best = d
        return best

    def step(self, target: Optional[_DetectionLike]) -> tuple[float, float]:
        """Compute (linear_x, angular_z) for the current target. None target → stop."""
        if target is None:
            return 0.0, 0.0

        yaw_error = target.yaw_error_rad
        if abs(yaw_error) < self.cfg.yaw_deadband_rad:
            angular_z = 0.0
        else:
            angular_z = -self.cfg.k_yaw * yaw_error
            # clamp for safety
            cap = self.cfg.max_angular_z
            angular_z = max(-cap, min(cap, angular_z))
            self._search_direction = 1.0 if angular_z >= 0.0 else -1.0

        area = float(target.size_x) * float(target.size_y)
        if area < self.cfg.a_slow:
            linear_x = self.cfg.v_max
        elif area < self.cfg.a_stop:
            span = self.cfg.a_stop - self.cfg.a_slow
            linear_x = self.cfg.v_max * (self.cfg.a_stop - area) / span
        else:
            linear_x = 0.0

        return linear_x, angular_z

    def search_step(self) -> tuple[float, float]:
        """Rotate in place to reacquire a person while the detector is still alive."""
        angular_z = abs(float(self.cfg.search_angular_z)) * self._search_direction
        cap = self.cfg.max_angular_z
        angular_z = max(-cap, min(cap, angular_z))
        return 0.0, angular_z
