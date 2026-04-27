#!/usr/bin/env python3
"""
Vision-Reactive Pupper main loop.

Subscribes to /detections (published by whichever detector bridge is active),
picks the highest-confidence person, and closes two loops:

  * locomotion — ApproachController turns the bounding box into a Twist on
    /cmd_vel (yaw proportional to image-frame dx, forward speed decreasing
    with bbox area).

  * tail — once the target is close enough that forward motion has stopped,
    TailController sends amplitude commands to the included serial servo
    firmware. Larger boxes mean larger wag amplitude.

Hardware missing (no serial tail controller attached, dev laptop, etc.) is
fine — the tail controller drops into dry-run mode and the rest of the
pipeline still runs.
"""

from __future__ import annotations

import os
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import rclpy
import yaml
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time

from geometry_msgs.msg import Twist
from vision_msgs.msg import Detection2DArray

from approach_controller import ApproachConfig, ApproachController
from tail_controller import TailConfig, TailController


DEFAULT_LABELS_PATH = os.path.join(os.path.dirname(__file__), "coco.txt")
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

DETECTION_WATCHDOG_S = 1.0  # stop if no frames for this long


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    center_x: float
    center_y: float
    size_x: float
    size_y: float
    image_width: float

    @property
    def normalized_x(self) -> float:
        width = max(1.0, float(self.image_width))
        return (self.center_x / width) - 0.5


class PupperInterface(Node):
    """ROS 2 node: subscribes to /detections, publishes /cmd_vel."""

    def __init__(self, labels_path: str = DEFAULT_LABELS_PATH, detection_image_width: float = 700.0):
        super().__init__("pupper_vision_tail_node")

        self._class_names = self._load_labels(labels_path)
        self._detection_image_width = float(detection_image_width)

        self._lock = threading.Lock()
        self._latest: List[Detection] = []
        self._last_time: Optional[Time] = None

        self._cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.create_subscription(
            Detection2DArray, "/detections", self._detection_cb, 10
        )

    def _load_labels(self, path: str) -> List[str]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().splitlines()
        except FileNotFoundError:
            self.get_logger().warning(
                f"Labels file not found at {path}; class_name will fall back to id."
            )
            return []

    def _name(self, class_id: int) -> str:
        if 0 <= class_id < len(self._class_names):
            return self._class_names[class_id]
        return str(class_id)

    def _detection_cb(self, msg: Detection2DArray) -> None:
        dets: List[Detection] = []
        for d in msg.detections:
            if not d.results:
                continue
            hyp = d.results[0]
            try:
                class_id = int(hyp.hypothesis.class_id)
            except ValueError:
                class_id = -1
            dets.append(
                Detection(
                    class_id=class_id,
                    class_name=self._name(class_id),
                    confidence=float(hyp.hypothesis.score),
                    center_x=float(d.bbox.center.position.x),
                    center_y=float(d.bbox.center.position.y),
                    size_x=float(d.bbox.size_x),
                    size_y=float(d.bbox.size_y),
                    image_width=self._detection_image_width,
                )
            )
        with self._lock:
            self._latest = dets
            self._last_time = self.get_clock().now()

    def get_detections(self) -> List[Detection]:
        with self._lock:
            return list(self._latest)

    def seconds_since_last_detection(self) -> float:
        with self._lock:
            t = self._last_time
        if t is None:
            return float("inf")
        return (self.get_clock().now() - t).nanoseconds / 1e9

    def set_velocity(self, linear_x: float = 0.0, linear_y: float = 0.0,
                     angular_z: float = 0.0) -> None:
        cmd = Twist()
        cmd.linear.x = float(linear_x)
        cmd.linear.y = float(linear_y)
        cmd.angular.z = float(angular_z)
        self._cmd_pub.publish(cmd)


def load_configs(path: str) -> tuple[ApproachConfig, TailConfig, dict]:
    """Parse config.yaml into controller configs. Missing file → defaults."""
    approach = ApproachConfig()
    tail = TailConfig()
    runtime = {
        "loop_hz": 10.0,
        "tail_enable_hardware": True,
        "detection_image_width": 700.0,
    }

    if not Path(path).is_file():
        return approach, tail, runtime

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    for key, val in (data.get("approach") or {}).items():
        if hasattr(approach, key):
            setattr(approach, key, val)
    for key, val in (data.get("tail") or {}).items():
        if hasattr(tail, key):
            setattr(tail, key, val)
    for key in ("loop_hz", "tail_enable_hardware", "detection_image_width"):
        if key in (data.get("runtime") or {}):
            runtime[key] = data["runtime"][key]

    return approach, tail, runtime


def main():
    approach_cfg, tail_cfg, runtime = load_configs(DEFAULT_CONFIG_PATH)
    approach = ApproachController(approach_cfg)
    tail = TailController(tail_cfg, enable_hardware=runtime["tail_enable_hardware"])

    # Disable rclpy's default SIGINT so we can publish a zero-velocity stop
    # before tearing the context down.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = PupperInterface(detection_image_width=runtime["detection_image_width"])

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    tail.start()
    node.get_logger().info(
        f"Started. Tail hardware: {'enabled' if tail.hardware_available else 'dry-run'}. "
        f"Loop {runtime['loop_hz']:.1f} Hz."
    )

    stop_requested = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_requested.set())

    loop_dt = 1.0 / max(1.0, runtime["loop_hz"])

    try:
        while rclpy.ok() and not stop_requested.is_set():
            # Watchdog: if detection feed has gone silent, stop and idle the tail.
            stale = node.seconds_since_last_detection() > DETECTION_WATCHDOG_S

            detections = node.get_detections()
            target = None if stale else approach.pick_target(detections)

            linear_x, angular_z = approach.step(target)

            if target is None:
                tail.set_idle()
            else:
                target_area = target.size_x * target.size_y
                if linear_x <= 1e-3:
                    tail.set_from_area(target_area)
                else:
                    tail.set_idle()

            node.set_velocity(linear_x=linear_x, angular_z=angular_z)
            time.sleep(loop_dt)

    except Exception as e:
        node.get_logger().error(f"Main loop error: {e}")
    finally:
        # Flush zero velocity so the neural controller actually stops the robot.
        for _ in range(5):
            node.set_velocity(0.0, 0.0, 0.0)
            time.sleep(0.1)
        tail.stop()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
