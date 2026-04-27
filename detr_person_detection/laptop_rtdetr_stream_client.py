import argparse
import json
import socket
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import cv2
import torch
from ultralytics import RTDETR


PROJECT_DIR = Path(__file__).resolve().parent
PERSON_CLASS_ID = 0


class LatestFrameReader:
    def __init__(self, stream_url):
        self.stream_url = stream_url
        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=2.0)

    def get(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def _run(self):
        cap = cv2.VideoCapture(self.stream_url)
        if not cap.isOpened():
            print(f"Could not open stream: {self.stream_url}")
            self.stopped = True
            return

        while not self.stopped:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            with self.lock:
                self.frame = frame

        cap.release()


def draw_label(image, text, x1, y1):
    cv2.rectangle(image, (x1, max(0, y1 - 30)), (x1 + 260, y1), (30, 120, 255), -1)
    cv2.putText(
        image,
        text,
        (x1 + 6, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def detect_people(model, frame_bgr, confidence):
    results = model.predict(frame_bgr, conf=confidence, verbose=False)
    people = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            score = float(box.conf[0])
            if class_id != PERSON_CLASS_ID or score < confidence:
                continue

            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                continue

            center_x = x1 + width / 2
            center_y = y1 + height / 2
            image_h, image_w = frame_bgr.shape[:2]
            horizontal_error = (center_x - image_w / 2) / (image_w / 2)
            area_ratio = (width * height) / (image_w * image_h)

            people.append(
                {
                    "confidence": score,
                    "box": (x1, y1, x2, y2),
                    "center": (center_x, center_y),
                    "horizontal_error": horizontal_error,
                    "area_ratio": area_ratio,
                }
            )

    people.sort(key=lambda item: item["confidence"], reverse=True)
    return people


def annotate(frame_bgr, people):
    output = frame_bgr.copy()
    for person in people:
        x1, y1, x2, y2 = person["box"]
        cv2.rectangle(output, (x1, y1), (x2, y2), (30, 120, 255), 3)
        draw_label(output, f"RT-DETR person {person['confidence']:.2f}", x1, y1)
    return output


def build_detection_packet(frame_bgr, people):
    image_h, image_w = frame_bgr.shape[:2]
    return {
        "image_width": image_w,
        "image_height": image_h,
        "detections": [
            {
                "class_id": PERSON_CLASS_ID,
                "confidence": person["confidence"],
                "box": list(person["box"]),
            }
            for person in people
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Run RT-DETR locally on Pupper's streamed camera video.")
    parser.add_argument("--stream-url", default="http://10.20.19.129:8080/stream.mjpg")
    parser.add_argument("--model", type=Path, default=PROJECT_DIR / "rtdetr-l.pt")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means run until interrupted.")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--robot-host", help="Robot IP/hostname for returning detections over UDP. Defaults to the stream host.")
    parser.add_argument("--robot-port", type=int, default=9999)
    parser.add_argument("--no-send", action="store_true", help="Preview only; do not send detections back to the robot.")
    parser.add_argument("--save-output", type=Path, default=PROJECT_DIR / "laptop_rtdetr_stream_detection.jpg")
    parser.add_argument("--save-raw", type=Path, default=PROJECT_DIR / "laptop_stream_latest_raw.jpg")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Could not find RT-DETR model: {args.model}")

    model = RTDETR(str(args.model))
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    print(f"Loaded RT-DETR model on {device}.")

    reader = LatestFrameReader(args.stream_url)
    reader.start()
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    robot_host = args.robot_host or urlparse(args.stream_url).hostname or "127.0.0.1"
    frame_index = 0
    last_annotated = None
    last_raw = None

    print(f"Reading stream: {args.stream_url}")
    if not args.no_send:
        print(f"Sending detections to udp://{robot_host}:{args.robot_port}")
    print("Press q in the preview window to quit." if args.preview else "Press Ctrl+C to stop.")

    try:
        while True:
            frame = reader.get()
            if frame is None:
                time.sleep(0.02)
                continue

            last_raw = frame
            started = time.perf_counter()
            people = detect_people(model, frame, args.confidence)
            elapsed = time.perf_counter() - started
            last_annotated = annotate(frame, people)

            if not args.no_send:
                packet = build_detection_packet(frame, people)
                sender.sendto(
                    json.dumps(packet, separators=(",", ":")).encode("utf-8"),
                    (robot_host, args.robot_port),
                )

            if people:
                target = people[0]
                print(
                    f"detected=True confidence={target['confidence']:.2f} "
                    f"horizontal_error={target['horizontal_error']:.3f} "
                    f"area_ratio={target['area_ratio']:.3f} "
                    f"inference_time={elapsed:.3f}s"
                )
            else:
                print(f"detected=False inference_time={elapsed:.3f}s")

            display = last_annotated if last_annotated is not None else frame

            if args.preview:
                cv2.imshow("Pupper stream RT-DETR", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
            if args.max_frames and frame_index >= args.max_frames:
                break

    except KeyboardInterrupt:
        print("Stopping.")
    finally:
        reader.stop()
        sender.close()
        if args.preview:
            cv2.destroyAllWindows()

    if last_raw is not None:
        args.save_raw.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_raw), last_raw)
        print(f"Saved latest raw frame to: {args.save_raw}")

    if last_annotated is not None:
        args.save_output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_output), last_annotated)
        print(f"Saved latest annotated frame to: {args.save_output}")


if __name__ == "__main__":
    main()
