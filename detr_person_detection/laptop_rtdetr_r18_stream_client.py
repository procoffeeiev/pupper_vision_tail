import argparse
import os
import json
import socket
import sys
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
from kornia.contrib.models.rt_detr import DETRPostProcessor
from kornia.models.detection import ObjectDetector
from kornia.models.detection.rtdetr import RTDETRDetectorBuilder
from kornia.models.utils import ResizePreProcessor


PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiment_logging import build_logger, target_fields  # noqa: E402

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
    def __init__(self, stream_url, event_logger=None):
        self.stream_url = stream_url
        self.event_logger = event_logger
        self.lock = threading.Lock()
        self.frame = None
        self.last_frame_time = 0.0
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

    def age_seconds(self):
        with self.lock:
            if self.last_frame_time <= 0.0:
                return float("inf")
            return time.time() - self.last_frame_time

    def _run(self):
        while not self.stopped:
            request = Request(self.stream_url, headers={"User-Agent": "pupper-vision-tail"})
            try:
                response = urlopen(request, timeout=5.0)
            except (URLError, TimeoutError, OSError) as exc:
                with self.lock:
                    self.frame = None
                    self.last_frame_time = 0.0
                if self.event_logger is not None:
                    self.event_logger.log("stream_connect_failed", stream_url=self.stream_url, notes=str(exc))
                print(f"Could not open stream: {self.stream_url} ({exc}); retrying...")
                time.sleep(1.0)
                continue

            print(f"Connected to stream: {self.stream_url}")
            if self.event_logger is not None:
                self.event_logger.log("stream_connect", stream_url=self.stream_url)
            buffer = bytearray()
            disconnect_reason = "reader_stop"
            try:
                while not self.stopped:
                    chunk = response.read(4096)
                    if not chunk:
                        with self.lock:
                            self.frame = None
                            self.last_frame_time = 0.0
                        disconnect_reason = "empty_chunk"
                        print("Stream stalled; reconnecting...")
                        break

                    buffer.extend(chunk)
                    while True:
                        start = buffer.find(b"\xff\xd8")
                        if start < 0:
                            if len(buffer) > 2:
                                del buffer[:-2]
                            break

                        end = buffer.find(b"\xff\xd9", start + 2)
                        if end < 0:
                            if start > 0:
                                del buffer[:start]
                            break

                        jpeg = bytes(buffer[start:end + 2])
                        del buffer[:end + 2]

                        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is None:
                            continue

                        with self.lock:
                            self.frame = frame
                            self.last_frame_time = time.time()
            except Exception as exc:
                if not self.stopped:
                    with self.lock:
                        self.frame = None
                        self.last_frame_time = 0.0
                    disconnect_reason = str(exc)
                    print(f"Stream read failed ({exc}); reconnecting...")
            finally:
                if self.event_logger is not None:
                    self.event_logger.log("stream_disconnect", stream_url=self.stream_url, notes=disconnect_reason)
                try:
                    response.close()
                except Exception:
                    pass
            if not self.stopped:
                time.sleep(0.5)


def draw_waiting_frame(width=960, height=540, text="Waiting for camera stream..."):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        image,
        text,
        (40, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


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
                "class_id": PERSON_CLASS_ID,
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


def build_detection_packet(
    frame_bgr,
    people,
    *,
    frame_index=None,
    session_id="",
    trial_id="",
    model_name="",
    device="",
    inference_time_s=None,
):
    image_h, image_w = frame_bgr.shape[:2]
    return {
        "session_id": session_id,
        "trial_id": trial_id,
        "frame_index": frame_index,
        "model": model_name,
        "device": device,
        "sent_unix_s": time.time(),
        "inference_time_s": inference_time_s,
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
    parser.add_argument("--session-id", default=os.environ.get("SESSION_ID"))
    parser.add_argument("--trial-id", default=os.environ.get("TRIAL_ID", ""))
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(os.environ.get("EXPERIMENT_LOG_DIR", ROOT_DIR / "data" / "experiments")),
    )
    parser.add_argument("--csv-log", type=Path, default=None)
    parser.add_argument(
        "--condition-distance-m",
        default=os.environ.get("CONDITION_DISTANCE_M", ""),
        help="Ground-truth person distance for detection trials, if known.",
    )
    parser.add_argument(
        "--ground-truth-person-present",
        choices=["true", "false", "unknown"],
        default=os.environ.get("GROUND_TRUTH_PERSON_PRESENT", "unknown"),
    )
    args = parser.parse_args()

    robot_host = args.robot_host or urlparse(args.stream_url).hostname or "127.0.0.1"
    logger = build_logger(
        side="laptop",
        component="rtdetr_r18",
        session_id=args.session_id,
        trial_id=args.trial_id,
        log_dir=args.log_dir,
        csv_log=args.csv_log,
    )
    logger.log(
        "laptop_client_start",
        stream_url=args.stream_url,
        robot_host=robot_host,
        robot_port=args.robot_port,
        model="rtdetr_r18vd",
        condition_distance_m=args.condition_distance_m,
        ground_truth_person_present=args.ground_truth_person_present,
        notes=f"csv_log={logger.path}",
    )

    print("Loading RT-DETR-R18...")
    model, device = build_model(args.image_size, args.confidence, args.device)
    print(f"Loaded RT-DETR-R18 on {device}.")
    logger.log("model_loaded", model="rtdetr_r18vd", device=str(device))

    reader = LatestFrameReader(args.stream_url, event_logger=logger)
    reader.start()
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    last_raw = None
    last_annotated = None
    processed = 0
    stop_reason = "normal"

    print(f"Reading stream: {args.stream_url}")
    if not args.no_send:
        print(f"Sending detections to udp://{robot_host}:{args.robot_port}")
    print("Press q in the preview window to quit." if args.preview else "Press Ctrl+C to stop.")

    try:
        while True:
            frame = reader.get()
            if frame is None or reader.age_seconds() > 2.0:
                if args.preview:
                    waiting = draw_waiting_frame(text="Reconnecting to Pupper camera stream...")
                    cv2.imshow("Pupper stream RT-DETR-R18", waiting)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                time.sleep(0.02)
                continue

            last_raw = frame
            started = time.perf_counter()
            people = detect_people(model, device, frame)
            elapsed = time.perf_counter() - started
            last_annotated = annotate(frame, people, elapsed, str(device))

            packet_bytes = ""
            if not args.no_send:
                packet = build_detection_packet(
                    frame,
                    people,
                    frame_index=processed,
                    session_id=logger.session_id,
                    trial_id=args.trial_id,
                    model_name="rtdetr_r18vd",
                    device=str(device),
                    inference_time_s=elapsed,
                )
                payload = json.dumps(packet, separators=(",", ":")).encode("utf-8")
                sender.sendto(
                    payload,
                    (robot_host, args.robot_port),
                )
                packet_bytes = len(payload)

            image_h, image_w = frame.shape[:2]
            log_fields = target_fields(people[0] if people else None, image_w, image_h)
            logger.log(
                "detection_frame",
                frame_index=processed,
                stream_url=args.stream_url,
                robot_host=robot_host,
                robot_port=args.robot_port,
                model="rtdetr_r18vd",
                device=str(device),
                image_width=image_w,
                image_height=image_h,
                detections_count=len(people),
                inference_time_s=elapsed,
                frame_age_s=reader.age_seconds(),
                packet_bytes=packet_bytes,
                condition_distance_m=args.condition_distance_m,
                ground_truth_person_present=args.ground_truth_person_present,
                **log_fields,
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
        stop_reason = "keyboard_interrupt"
        print("Stopping.")
    finally:
        reader.stop()
        sender.close()
        if args.preview:
            cv2.destroyAllWindows()
        logger.log("laptop_client_stop", frame_index=processed, notes=stop_reason)
        logger.close()

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
