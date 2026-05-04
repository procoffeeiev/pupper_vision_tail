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

import math
import os
import argparse
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
from experiment_logging import build_logger
from fisheye_converter import load_camera_model
from tail_controller import TailConfig, TailController


DEFAULT_LABELS_PATH = os.path.join(os.path.dirname(__file__), "coco.txt")
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
DEFAULT_CAMERA_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "camera_params.yaml")

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
    image_height: float
    yaw_error_rad: float

    @property
    def normalized_x(self) -> float:
        width = max(1.0, float(self.image_width))
        return (self.center_x / width) - 0.5


class PupperInterface(Node):
    """ROS 2 node: subscribes to /detections, publishes /cmd_vel."""

    def __init__(
        self,
        labels_path: str = DEFAULT_LABELS_PATH,
        detection_image_width: float = 700.0,
        detection_image_height: float = 525.0,
        camera_params_path: str = DEFAULT_CAMERA_PARAMS_PATH,
    ):
        super().__init__("pupper_vision_tail_node")

        self._class_names = self._load_labels(labels_path)
        self._detection_image_width = float(detection_image_width)
        self._detection_image_height = float(detection_image_height)
        self._camera_model = None
        if Path(camera_params_path).is_file():
            try:
                self._camera_model = load_camera_model(
                    camera_params_path,
                    int(self._detection_image_width),
                    int(self._detection_image_height),
                )
                self.get_logger().info("Using fisheye camera model for yaw-angle mapping.")
            except Exception as exc:
                self.get_logger().warning(f"Failed to load camera model; using linear yaw mapping ({exc})")

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

    def _yaw_error_from_pixel(self, center_x: float, center_y: float) -> float:
        if self._camera_model is None:
            width = max(1.0, self._detection_image_width)
            return ((center_x / width) - 0.5) * 2.0

        ray_x, _, ray_z, valid = self._camera_model.unproject(center_x, center_y)
        valid_value = bool(valid) if not hasattr(valid, "shape") else bool(valid.all())
        if not valid_value:
            width = max(1.0, self._detection_image_width)
            return ((center_x / width) - 0.5) * 2.0
        return float(math.atan2(float(ray_x), float(ray_z)))

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
            center_x = float(d.bbox.center.position.x)
            center_y = float(d.bbox.center.position.y)
            dets.append(
                Detection(
                    class_id=class_id,
                    class_name=self._name(class_id),
                    confidence=float(hyp.hypothesis.score),
                    center_x=center_x,
                    center_y=center_y,
                    size_x=float(d.bbox.size_x),
                    size_y=float(d.bbox.size_y),
                    image_width=self._detection_image_width,
                    image_height=self._detection_image_height,
                    yaw_error_rad=self._yaw_error_from_pixel(center_x, center_y),
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


def detection_log_fields(target: Optional[Detection]) -> dict:
    if target is None:
        return {"target_detected": False}

    area = float(target.size_x) * float(target.size_y)
    frame_area = max(1.0, float(target.image_width) * float(target.image_height))
    return {
        "target_detected": True,
        "target_confidence": target.confidence,
        "target_class_id": target.class_id,
        "target_center_x": target.center_x,
        "target_center_y": target.center_y,
        "target_area_px2": area,
        "target_area_ratio": area / frame_area,
        "horizontal_error": target.normalized_x * 2.0,
        "yaw_error_rad": target.yaw_error_rad,
    }


def load_configs(path: str) -> tuple[ApproachConfig, TailConfig, dict]:
    """Parse config.yaml into controller configs. Missing file → defaults."""
    approach = ApproachConfig()
    tail = TailConfig()
    runtime = {
        "loop_hz": 10.0,
        "tail_enable_hardware": True,
        "detection_image_width": 700.0,
        "detection_image_height": 525.0,
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
    for key in ("loop_hz", "tail_enable_hardware", "detection_image_width", "detection_image_height"):
        if key in (data.get("runtime") or {}):
            runtime[key] = data["runtime"][key]

    return approach, tail, runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", default=os.environ.get("SESSION_ID"))
    parser.add_argument("--trial-id", default=os.environ.get("TRIAL_ID", ""))
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(os.environ.get("EXPERIMENT_LOG_DIR", Path(__file__).resolve().parent / "data" / "experiments")),
    )
    parser.add_argument("--csv-log", type=Path, default=None)
    parser.add_argument("--condition-distance-m", default=os.environ.get("CONDITION_DISTANCE_M", ""))
    parser.add_argument("--final-distance-m", default=os.environ.get("FINAL_DISTANCE_M", ""))
    parser.add_argument(
        "--ground-truth-person-present",
        choices=["true", "false", "unknown"],
        default=os.environ.get("GROUND_TRUTH_PERSON_PRESENT", "unknown"),
    )
    args = parser.parse_args()

    logger = build_logger(
        side="pupper",
        component="control_loop",
        session_id=args.session_id,
        trial_id=args.trial_id,
        log_dir=args.log_dir,
        csv_log=args.csv_log,
    )
    approach_cfg, tail_cfg, runtime = load_configs(DEFAULT_CONFIG_PATH)
    approach = ApproachController(approach_cfg)
    tail = TailController(tail_cfg, enable_hardware=runtime["tail_enable_hardware"], event_logger=logger)

    # Disable rclpy's default SIGINT so we can publish a zero-velocity stop
    # before tearing the context down.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = PupperInterface(
        detection_image_width=runtime["detection_image_width"],
        detection_image_height=runtime["detection_image_height"],
    )

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    tail.start()
    node.get_logger().info(
        f"Started. Tail hardware: {'enabled' if tail.hardware_available else 'dry-run'}. "
        f"Loop {runtime['loop_hz']:.1f} Hz."
    )
    logger.log(
        "control_loop_start",
        image_width=runtime["detection_image_width"],
        image_height=runtime["detection_image_height"],
        tail_hw_available=tail.hardware_available,
        condition_distance_m=args.condition_distance_m,
        ground_truth_person_present=args.ground_truth_person_present,
        notes=f"loop_hz={runtime['loop_hz']}; csv_log={logger.path}",
    )

    stop_requested = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_requested.set())

    loop_dt = 1.0 / max(1.0, runtime["loop_hz"])
    last_mode = None
    tail_active = False
    approach_active = False
    approach_start_time = None
    loop_index = 0

    try:
        while rclpy.ok() and not stop_requested.is_set():
            # Watchdog: if detection feed has gone silent, stop and idle the tail.
            frame_age_s = node.seconds_since_last_detection()
            stale = frame_age_s > DETECTION_WATCHDOG_S

            detections = node.get_detections()
            target = None if stale else approach.pick_target(detections)
            target_area = 0.0
            tail_amplitude_us = 0.0

            if stale:
                mode = "stale"
                linear_x, angular_z = 0.0, 0.0
                tail_amplitude_us = tail.set_idle()
            elif target is None:
                mode = "search"
                linear_x, angular_z = approach.search_step()
                tail_amplitude_us = tail.set_idle()
            else:
                mode = "track"
                linear_x, angular_z = approach.step(target)
                target_area = target.size_x * target.size_y
                if linear_x <= 1e-3:
                    tail_amplitude_us = tail.set_from_area(target_area)
                else:
                    tail_amplitude_us = tail.set_idle()

            if mode != last_mode:
                if mode == "stale":
                    node.get_logger().warning("Detection feed stale; stopping in place.")
                elif mode == "search":
                    node.get_logger().info("No person detected; rotating in place to search.")
                else:
                    node.get_logger().info("Person detected; tracking target.")
                logger.log(
                    "mode_change",
                    mode=mode,
                    stale=stale,
                    frame_age_s=frame_age_s,
                    detections_count=len(detections),
                    tail_amplitude_us=tail_amplitude_us,
                    tail_hw_available=tail.hardware_available,
                    condition_distance_m=args.condition_distance_m,
                    ground_truth_person_present=args.ground_truth_person_present,
                    **detection_log_fields(target),
                )
                last_mode = mode

            if (
                mode == "track"
                and target is not None
                and linear_x > 1e-3
                and not approach_active
            ):
                approach_start_time = time.monotonic()
                approach_active = True
                logger.log(
                    "approach_start",
                    mode=mode,
                    stale=stale,
                    frame_age_s=frame_age_s,
                    detections_count=len(detections),
                    linear_x=linear_x,
                    angular_z=angular_z,
                    tail_amplitude_us=tail_amplitude_us,
                    tail_hw_available=tail.hardware_available,
                    condition_distance_m=args.condition_distance_m,
                    ground_truth_person_present=args.ground_truth_person_present,
                    **detection_log_fields(target),
                )

            if (
                mode == "track"
                and target is not None
                and linear_x <= 1e-3
                and approach_active
            ):
                duration = ""
                if approach_start_time is not None:
                    duration = time.monotonic() - approach_start_time
                logger.log(
                    "approach_stop",
                    mode=mode,
                    stale=stale,
                    frame_age_s=frame_age_s,
                    detections_count=len(detections),
                    linear_x=linear_x,
                    angular_z=angular_z,
                    tail_amplitude_us=tail_amplitude_us,
                    tail_hw_available=tail.hardware_available,
                    condition_distance_m=args.condition_distance_m,
                    ground_truth_person_present=args.ground_truth_person_present,
                    approach_duration_s=duration,
                    **detection_log_fields(target),
                )
                approach_active = False
                approach_start_time = None

            now_tail_active = tail_amplitude_us > 0.0
            if now_tail_active and not tail_active:
                logger.log(
                    "tail_wag_start",
                    mode=mode,
                    stale=stale,
                    frame_age_s=frame_age_s,
                    detections_count=len(detections),
                    linear_x=linear_x,
                    angular_z=angular_z,
                    tail_amplitude_us=tail_amplitude_us,
                    tail_hw_available=tail.hardware_available,
                    condition_distance_m=args.condition_distance_m,
                    ground_truth_person_present=args.ground_truth_person_present,
                    **detection_log_fields(target),
                )
            elif tail_active and not now_tail_active:
                logger.log(
                    "tail_wag_stop",
                    mode=mode,
                    stale=stale,
                    frame_age_s=frame_age_s,
                    detections_count=len(detections),
                    tail_amplitude_us=tail_amplitude_us,
                    tail_hw_available=tail.hardware_available,
                    condition_distance_m=args.condition_distance_m,
                    ground_truth_person_present=args.ground_truth_person_present,
                    **detection_log_fields(target),
                )
            tail_active = now_tail_active

            logger.log(
                "control_sample",
                frame_index=loop_index,
                mode=mode,
                stale=stale,
                frame_age_s=frame_age_s,
                detections_count=len(detections),
                linear_x=linear_x,
                angular_z=angular_z,
                tail_amplitude_us=tail_amplitude_us,
                tail_hw_available=tail.hardware_available,
                condition_distance_m=args.condition_distance_m,
                ground_truth_person_present=args.ground_truth_person_present,
                **detection_log_fields(target),
            )

            node.set_velocity(linear_x=linear_x, angular_z=angular_z)
            loop_index += 1
            time.sleep(loop_dt)

    except Exception as e:
        node.get_logger().error(f"Main loop error: {e}")
    finally:
        # Flush zero velocity so the neural controller actually stops the robot.
        for _ in range(5):
            node.set_velocity(0.0, 0.0, 0.0)
            time.sleep(0.1)
        tail.stop()
        logger.log(
            "control_loop_stop",
            frame_index=loop_index,
            final_distance_m=args.final_distance_m,
            tail_hw_available=tail.hardware_available,
        )
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)
        logger.close()


if __name__ == "__main__":
    main()
