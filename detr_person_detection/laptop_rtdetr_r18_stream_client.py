import argparse
import json
import socket
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import cv2
import torch
from kornia.contrib.models.rt_detr import DETRPostProcessor
from kornia.models.detection import ObjectDetector
from kornia.models.detection.rtdetr import RTDETRDetectorBuilder
from kornia.models.utils import ResizePreProcessor


PROJECT_DIR = Path(__file__).resolve().parent
PERSON_CLASS_ID = 0


def resolve_device(device_arg: str) -> str:
    if device_arg in {"gpu", "auto"}:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        raise SystemExit(
            "No GPU backend available for RT-DETR-R18. "
            "Use a CUDA/MPS-capable machine or explicitly pass --device cpu."
        )

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("Requested --device cuda, but CUDA is not available.")
        return "cuda"

    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("Requested --device mps, but Apple MPS is not available.")
        return "mps"

    return "cpu"


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


def build_model(image_size, confidence, device):
    model = RTDETRDetectorBuilder.build(
        model_name="rtdetr_r18vd",
        pretrained=True,
        image_size=image_size,
        confidence_threshold=confidence,
        confidence_filtering=True,
    )

    device = resolve_device(device)

    model = model.to(device).eval()
    return model, torch.device(device)


def detect_people(model, device, frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor.to(device)

    with torch.inference_mode():
        detections = model([tensor])[0].detach().cpu()

    image_h, image_w = frame_bgr.shape[:2]
    people = []

    for detection in detections:
        class_id = int(detection[0].item())
        confidence = float(detection[1].item())
        if class_id != PERSON_CLASS_ID:
            continue

        x, y, width, height = [float(v) for v in detection[2:6]]
        if width <= 0 or height <= 0:
            continue

        x1 = int(max(0, min(image_w - 1, x)))
        y1 = int(max(0, min(image_h - 1, y)))
        x2 = int(max(0, min(image_w - 1, x + width)))
        y2 = int(max(0, min(image_h - 1, y + height)))

        box_width = x2 - x1
        box_height = y2 - y1
        if box_width <= 0 or box_height <= 0:
            continue

        center_x = x1 + box_width / 2
        center_y = y1 + box_height / 2
        horizontal_error = (center_x - image_w / 2) / (image_w / 2)
        area_ratio = (box_width * box_height) / (image_w * image_h)

        people.append(
            {
                "confidence": confidence,
                "box": (x1, y1, x2, y2),
                "center": (center_x, center_y),
                "horizontal_error": horizontal_error,
                "area_ratio": area_ratio,
            }
        )

    # Prefer the most useful tracking target, not a tiny high-confidence speck.
    people.sort(key=lambda item: (item["area_ratio"], item["confidence"]), reverse=True)
    return people


def annotate(frame_bgr, people, inference_time, device):
    output = frame_bgr.copy()

    for person in people:
        x1, y1, x2, y2 = person["box"]
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 120, 30), 2)
        label = f"RT-DETR-R18 {person['confidence']:.2f}"
        cv2.putText(
            output,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 120, 30),
            2,
            cv2.LINE_AA,
        )

    status = f"R18 {device} {inference_time:.3f}s"
    cv2.putText(output, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 120, 30), 2, cv2.LINE_AA)
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
    parser = argparse.ArgumentParser(description="Run smaller RT-DETR-R18 locally on Pupper's streamed camera video.")
    parser.add_argument("--stream-url", default="http://10.20.19.129:8080/stream.mjpg")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--image-size", type=int, default=480)
    parser.add_argument("--device", choices=["gpu", "auto", "cuda", "mps", "cpu"], default="gpu")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--robot-host", help="Robot IP/hostname for returning detections over UDP. Defaults to the stream host.")
    parser.add_argument("--robot-port", type=int, default=9999)
    parser.add_argument("--no-send", action="store_true", help="Preview only; do not send detections back to the robot.")
    parser.add_argument("--save-output", type=Path, default=PROJECT_DIR / "laptop_rtdetr_r18_stream_detection.jpg")
    parser.add_argument("--save-raw", type=Path, default=PROJECT_DIR / "laptop_rtdetr_r18_stream_raw.jpg")
    args = parser.parse_args()

    print("Loading RT-DETR-R18...")
    model, device = build_model(args.image_size, args.confidence, args.device)
    print(f"Loaded RT-DETR-R18 on {device}.")

    reader = LatestFrameReader(args.stream_url)
    reader.start()
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    robot_host = args.robot_host or urlparse(args.stream_url).hostname or "127.0.0.1"

    last_raw = None
    last_annotated = None
    processed = 0

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
            people = detect_people(model, device, frame)
            elapsed = time.perf_counter() - started
            last_annotated = annotate(frame, people, elapsed, str(device))

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

            if args.preview:
                cv2.imshow("Pupper stream RT-DETR-R18", last_annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            processed += 1
            if args.max_frames and processed >= args.max_frames:
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
