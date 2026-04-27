#!/usr/bin/env python3
"""
Receive person detections from a remote RT-DETR laptop client and republish
them as vision_msgs/Detection2DArray on /detections.
"""

from __future__ import annotations

import argparse
import json
import socket
from typing import Any

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


class RemoteDetectionBridge(Node):
    def __init__(self, host: str, port: int):
        super().__init__("remote_detection_bridge")
        self._publisher = self.create_publisher(Detection2DArray, "/detections", 10)

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((host, port))
        self._sock.setblocking(False)

        self.get_logger().info(f"Listening for remote detections on udp://{host}:{port}")
        self.create_timer(0.01, self._poll_socket)

    def destroy_node(self):
        try:
            self._sock.close()
        except Exception:
            pass
        super().destroy_node()

    def _poll_socket(self) -> None:
        while True:
            try:
                payload, _ = self._sock.recvfrom(65535)
            except BlockingIOError:
                return
            except Exception as e:
                self.get_logger().error(f"Socket read failed: {e}")
                return

            try:
                message = json.loads(payload.decode("utf-8"))
            except Exception as e:
                self.get_logger().warning(f"Dropping invalid detection packet ({e})")
                continue

            self._publish_detection_array(message)

    def _publish_detection_array(self, message: dict[str, Any]) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9999)
    args = parser.parse_args()

    rclpy.init()
    node = RemoteDetectionBridge(args.host, args.port)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
