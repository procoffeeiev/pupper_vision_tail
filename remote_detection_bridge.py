#!/usr/bin/env python3
"""
Receive person detections from a remote RT-DETR laptop client and republish
them as vision_msgs/Detection2DArray on /detections.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from experiment_logging import build_logger, target_fields


class RemoteDetectionBridge(Node):
    def __init__(self, host: str, port: int, event_logger=None, peer_timeout_s: float = 2.0):
        super().__init__("remote_detection_bridge")
        self._publisher = self.create_publisher(Detection2DArray, "/detections", 10)
        self._event_logger = event_logger
        self._peer_timeout_s = max(0.1, float(peer_timeout_s))
        self._peers: dict[str, float] = {}

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((host, port))
        self._sock.setblocking(False)

        self.get_logger().info(f"Listening for remote detections on udp://{host}:{port}")
        if self._event_logger is not None:
            self._event_logger.log("bridge_start", robot_host=host, robot_port=port)
        self.create_timer(0.01, self._poll_socket)
        self.create_timer(0.25, self._check_peer_timeouts)

    def destroy_node(self):
        if self._event_logger is not None:
            for peer in list(self._peers):
                self._event_logger.log("udp_peer_disconnect", peer=peer, notes="bridge_shutdown")
            self._event_logger.log("bridge_stop")
        try:
            self._sock.close()
        except Exception:
            pass
        super().destroy_node()

    def _poll_socket(self) -> None:
        while True:
            try:
                payload, addr = self._sock.recvfrom(65535)
            except BlockingIOError:
                return
            except Exception as e:
                self.get_logger().error(f"Socket read failed: {e}")
                return

            peer = f"{addr[0]}:{addr[1]}"
            self._note_peer(peer)
            try:
                message = json.loads(payload.decode("utf-8"))
            except Exception as e:
                self.get_logger().warning(f"Dropping invalid detection packet ({e})")
                if self._event_logger is not None:
                    self._event_logger.log("udp_invalid_packet", peer=peer, packet_bytes=len(payload), notes=str(e))
                continue

            detection_count = self._publish_detection_array(message)
            if self._event_logger is not None:
                target = self._select_logged_target(message)
                image_width = float(message.get("image_width", 0.0) or 0.0)
                image_height = float(message.get("image_height", 0.0) or 0.0)
                self._event_logger.log(
                    "udp_packet",
                    peer=peer,
                    packet_bytes=len(payload),
                    frame_index=message.get("frame_index", ""),
                    image_width=image_width,
                    image_height=image_height,
                    detections_count=detection_count,
                    model=message.get("model", ""),
                    device=message.get("device", ""),
                    inference_time_s=message.get("inference_time_s", ""),
                    notes=f"laptop_session_id={message.get('session_id', '')}; laptop_trial_id={message.get('trial_id', '')}",
                    **target_fields(target, image_width, image_height),
                )

    def _note_peer(self, peer: str) -> None:
        if peer not in self._peers and self._event_logger is not None:
            self._event_logger.log("udp_peer_connect", peer=peer)
        self._peers[peer] = time.monotonic()

    def _check_peer_timeouts(self) -> None:
        now = time.monotonic()
        for peer, last_seen in list(self._peers.items()):
            if now - last_seen > self._peer_timeout_s:
                if self._event_logger is not None:
                    self._event_logger.log("udp_peer_disconnect", peer=peer, notes="peer_timeout")
                del self._peers[peer]

    def _select_logged_target(self, message: dict[str, Any]) -> dict[str, Any] | None:
        best = None
        for det in message.get("detections", []):
            box = det.get("box")
            if not isinstance(box, list) or len(box) != 4:
                continue
            candidate = {
                "class_id": det.get("class_id", 0),
                "confidence": float(det.get("confidence", 0.0)),
                "box": box,
            }
            if best is None or candidate["confidence"] > best["confidence"]:
                best = candidate
        return best

    def _publish_detection_array(self, message: dict[str, Any]) -> int:
        detection_msg = Detection2DArray()
        detection_msg.header.stamp = self.get_clock().now().to_msg()
        detection_msg.header.frame_id = "camera"

        for det in message.get("detections", []):
            box = det.get("box")
            if not isinstance(box, list) or len(box) != 4:
                continue

            x1, y1, x2, y2 = [float(v) for v in box]
            size_x = x2 - x1
            size_y = y2 - y1
            if size_x <= 0.0 or size_y <= 0.0:
                continue

            detection = Detection2D()
            detection.bbox.center.position.x = (x1 + x2) / 2.0
            detection.bbox.center.position.y = (y1 + y2) / 2.0
            detection.bbox.size_x = size_x
            detection.bbox.size_y = size_y

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(int(det.get("class_id", 0)))
            hyp.hypothesis.score = float(det.get("confidence", 0.0))
            detection.results.append(hyp)

            detection_msg.detections.append(detection)

        self._publisher.publish(detection_msg)
        return len(detection_msg.detections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--peer-timeout-s", type=float, default=2.0)
    parser.add_argument("--session-id", default=os.environ.get("SESSION_ID"))
    parser.add_argument("--trial-id", default=os.environ.get("TRIAL_ID", ""))
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(os.environ.get("EXPERIMENT_LOG_DIR", Path(__file__).resolve().parent / "data" / "experiments")),
    )
    parser.add_argument("--csv-log", type=Path, default=None)
    args = parser.parse_args()

    logger = build_logger(
        side="pupper",
        component="detection_bridge",
        session_id=args.session_id,
        trial_id=args.trial_id,
        log_dir=args.log_dir,
        csv_log=args.csv_log,
    )

    rclpy.init()
    node = RemoteDetectionBridge(args.host, args.port, event_logger=logger, peer_timeout_s=args.peer_timeout_s)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        logger.close()


if __name__ == "__main__":
    main()
