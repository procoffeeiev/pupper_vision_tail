import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image


PERSON_CLASS_ID = 0


def letterbox(image, size=640, color=(114, 114, 114)):
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    pad_x = size - new_width
    pad_y = size - new_height
    left = pad_x // 2
    right = pad_x - left
    top = pad_y // 2
    bottom = pad_y - top

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    return padded, scale, left, top


def detect_people(session, frame_rgb, input_size, confidence_threshold, nms_threshold):
    original_height, original_width = frame_rgb.shape[:2]
    model_input, scale, pad_left, pad_top = letterbox(frame_rgb, input_size)

    blob = np.transpose(model_input.astype(np.float32) / 255.0, (2, 0, 1))[None]
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: blob})[0]

    predictions = output[0]
    boxes = []
    confidences = []

    for pred in predictions:
        confidence = float(pred[4 + PERSON_CLASS_ID])
        if confidence < confidence_threshold:
            continue

        center_x, center_y, width, height = pred[:4] * input_size

        x1 = (center_x - width / 2 - pad_left) / scale
        y1 = (center_y - height / 2 - pad_top) / scale
        x2 = (center_x + width / 2 - pad_left) / scale
        y2 = (center_y + height / 2 - pad_top) / scale

        x1 = int(max(0, min(original_width - 1, x1)))
        y1 = int(max(0, min(original_height - 1, y1)))
        x2 = int(max(0, min(original_width - 1, x2)))
        y2 = int(max(0, min(original_height - 1, y2)))

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        confidences.append(confidence)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    people = []

    if len(indices) == 0:
        return people

    for index in np.array(indices).flatten():
        x, y, width, height = boxes[index]
        confidence = confidences[index]
        x1, y1, x2, y2 = x, y, x + width, y + height
        center_x = x1 + width / 2
        center_y = y1 + height / 2
        area_ratio = (width * height) / (original_width * original_height)
        horizontal_error = (center_x - original_width / 2) / (original_width / 2)

        people.append(
            {
                "confidence": confidence,
                "box": (x1, y1, x2, y2),
                "center": (center_x, center_y),
                "horizontal_error": horizontal_error,
                "area_ratio": area_ratio,
            }
        )

    people.sort(key=lambda item: item["confidence"], reverse=True)
    return people


def draw_people(frame_rgb, people):
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    for person in people:
        x1, y1, x2, y2 = person["box"]
        confidence = person["confidence"]
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (30, 120, 255), 3)
        cv2.putText(
            frame_bgr,
            f"RT-DETR person {confidence:.2f}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (30, 120, 255),
            2,
            cv2.LINE_AA,
        )
    return frame_bgr


class OneFrameSubscriber(Node):
    def __init__(self, topic):
        super().__init__("one_frame_rtdetr_subscriber")
        self.frame_rgb = None
        self.subscription = self.create_subscription(
            CompressedImage if topic.endswith("compressed") else Image,
            topic,
            self.on_image,
            1,
        )

    def on_image(self, msg):
        if isinstance(msg, CompressedImage):
            data = np.frombuffer(msg.data, dtype=np.uint8)
            frame_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                self.get_logger().warning("Could not decode compressed image.")
                return
        else:
            channels = 3 if msg.encoding in ("rgb8", "bgr8") else 1
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if msg.encoding == "rgb8" else frame

        self.frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def receive_one_frame(topic, timeout):
    rclpy.init()
    node = OneFrameSubscriber(topic)
    deadline = time.time() + timeout
    try:
        while rclpy.ok() and node.frame_rgb is None and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.frame_rgb is None:
            raise SystemExit(f"No image received from {topic} within {timeout:.1f}s")
        return node.frame_rgb
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main():
    default_dir = Path.home() / "pupper_human_detection"

    parser = argparse.ArgumentParser(description="Run RT-DETR on one ROS camera frame.")
    parser.add_argument("--topic", default="/camera/image_raw/compressed")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--model", type=Path, default=default_dir / "rtdetr-l.onnx")
    parser.add_argument("--output", type=Path, default=default_dir / "pupper_rtdetr_ros_detection.jpg")
    parser.add_argument("--raw-output", type=Path, default=default_dir / "pupper_rtdetr_ros_raw.jpg")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--nms", type=float, default=0.45)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Could not find ONNX model: {args.model}")

    frame_rgb = receive_one_frame(args.topic, args.timeout)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.raw_output), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    started = time.perf_counter()
    people = detect_people(session, frame_rgb, args.input_size, args.confidence, args.nms)
    elapsed = time.perf_counter() - started

    annotated = draw_people(frame_rgb, people)
    cv2.imwrite(str(args.output), annotated)

    for i, person in enumerate(people, start=1):
        x1, y1, x2, y2 = person["box"]
        center_x, center_y = person["center"]
        print(
            f"Person {i}: confidence={person['confidence']:.2f}, "
            f"box=({x1}, {y1}, {x2}, {y2}), "
            f"center=({center_x:.1f}, {center_y:.1f}), "
            f"horizontal_error={person['horizontal_error']:.3f}, "
            f"area_ratio={person['area_ratio']:.3f}"
        )

    print(f"People detected: {len(people)}")
    print(f"Inference time: {elapsed:.3f}s")
    print(f"Saved raw frame to: {args.raw_output}")
    print(f"Saved annotated frame to: {args.output}")


if __name__ == "__main__":
    main()
