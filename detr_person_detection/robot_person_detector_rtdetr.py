import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


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


def load_camera_frame(width, height, warmup_seconds):
    from picamera2 import Picamera2

    camera = Picamera2()
    config = camera.create_preview_configuration(
        main={"size": (width, height), "format": "RGB888"}
    )
    camera.configure(config)
    camera.start()
    time.sleep(warmup_seconds)
    frame_rgb = camera.capture_array()
    camera.stop()
    return frame_rgb


def load_image_frame(path):
    frame_bgr = cv2.imread(str(path))
    if frame_bgr is None:
        raise SystemExit(f"Could not read image: {path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


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
        class_scores = pred[4:]
        confidence = float(class_scores[PERSON_CLASS_ID])
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

    indices = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        confidence_threshold,
        nms_threshold,
    )

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
        label = f"RT-DETR person {confidence:.2f}"
        cv2.putText(
            frame_bgr,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (30, 120, 255),
            2,
            cv2.LINE_AA,
        )

    return frame_bgr


def main():
    default_dir = Path.home() / "pupper_human_detection"

    parser = argparse.ArgumentParser(description="Run RT-DETR person detection on Pupper camera frames.")
    parser.add_argument("--source", choices=["camera", "image"], default="camera")
    parser.add_argument("--input", type=Path, help="Input image path when --source image is used.")
    parser.add_argument("--model", type=Path, default=default_dir / "rtdetr-l.onnx")
    parser.add_argument("--output", type=Path, default=default_dir / "pupper_rtdetr_person_detection.jpg")
    parser.add_argument("--raw-output", type=Path, default=default_dir / "pupper_rtdetr_camera_frame.jpg")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--warmup", type=float, default=1.0)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Could not find ONNX model: {args.model}")

    if args.source == "camera":
        frame_rgb = load_camera_frame(args.width, args.height, args.warmup)
    else:
        if args.input is None:
            raise SystemExit("--input is required when --source image is used.")
        frame_rgb = load_image_frame(args.input)

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
