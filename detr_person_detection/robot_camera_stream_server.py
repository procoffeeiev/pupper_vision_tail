import argparse
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiment_logging import build_logger  # noqa: E402


class FrameStore:
    def __init__(self):
        self.condition = threading.Condition()
        self.jpeg = None
        self.frame_count = 0
        self.updated_at = 0.0

    def set(self, jpeg):
        with self.condition:
            self.jpeg = bytes(jpeg)
            self.frame_count += 1
            self.updated_at = time.time()
            self.condition.notify_all()

    def get(self, timeout=2.0):
        deadline = time.time() + timeout
        with self.condition:
            while self.jpeg is None and time.time() < deadline:
                self.condition.wait(timeout=0.1)
            return self.jpeg, self.frame_count, self.updated_at

    def wait_next(self, previous_count, timeout=2.0):
        deadline = time.time() + timeout
        with self.condition:
            while self.frame_count == previous_count and time.time() < deadline:
                self.condition.wait(timeout=0.1)
            return self.jpeg, self.frame_count, self.updated_at


FRAME_STORE = FrameStore()
STOP_EVENT = threading.Event()
EVENT_LOGGER = None


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def run_picamera_source(width, height, fps, quality):
    import cv2
    from picamera2 import Picamera2

    camera = Picamera2()
    config = camera.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"}
    )
    camera.configure(config)
    camera.start()
    time.sleep(1.0)

    delay = 1.0 / max(fps, 1)
    try:
        while not STOP_EVENT.is_set():
            frame_rgb = camera.capture_array()
            ok, encoded = cv2.imencode(
                ".jpg",
                cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), quality],
            )
            if ok:
                FRAME_STORE.set(encoded)
            time.sleep(delay)
    finally:
        camera.stop()


def run_ros_compressed_source(topic):
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage

    class CameraSubscriber(Node):
        def __init__(self):
            super().__init__("camera_mjpeg_bridge")
            self.create_subscription(CompressedImage, topic, self.on_image, 1)

        def on_image(self, msg):
            FRAME_STORE.set(msg.data)

    rclpy.init()
    node = CameraSubscriber()
    try:
        while rclpy.ok() and not STOP_EVENT.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    except ExternalShutdownException:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"{self.address_string()} - {fmt % args}")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h1>Pupper Camera Stream</h1>"
                b"<img src='/stream.mjpg' style='max-width:100%;'>"
                b"</body></html>"
            )
            return

        if self.path == "/snapshot.jpg":
            jpeg, _, _ = FRAME_STORE.get()
            if jpeg is None:
                self.send_error(503, "No camera frame available yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.end_headers()
            self.wfile.write(jpeg)
            return

        if self.path == "/stream.mjpg":
            peer = f"{self.client_address[0]}:{self.client_address[1]}"
            if EVENT_LOGGER is not None:
                EVENT_LOGGER.log("stream_client_connect", peer=peer, stream_url=self.path)
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            count = -1
            disconnect_reason = "server_stop"
            while not STOP_EVENT.is_set():
                jpeg, count, _ = FRAME_STORE.wait_next(count)
                if jpeg is None:
                    continue
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    disconnect_reason = "client_disconnect"
                    break
            if EVENT_LOGGER is not None:
                EVENT_LOGGER.log(
                    "stream_client_disconnect",
                    peer=peer,
                    stream_url=self.path,
                    frame_index=count,
                    notes=disconnect_reason,
                )
            return

        self.send_error(404)


def main():
    global EVENT_LOGGER

    parser = argparse.ArgumentParser(description="Stream Pupper camera frames as MJPEG over HTTP.")
    parser.add_argument("--source", choices=["picamera", "ros-compressed"], default="picamera")
    parser.add_argument("--topic", default="/camera/image_raw/compressed")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--session-id", default=os.environ.get("SESSION_ID"))
    parser.add_argument("--trial-id", default=os.environ.get("TRIAL_ID", ""))
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(os.environ.get("EXPERIMENT_LOG_DIR", ROOT_DIR / "data" / "experiments")),
    )
    parser.add_argument("--csv-log", type=Path, default=None)
    args = parser.parse_args()

    EVENT_LOGGER = build_logger(
        side="pupper",
        component="camera_stream",
        session_id=args.session_id,
        trial_id=args.trial_id,
        log_dir=args.log_dir,
        csv_log=args.csv_log,
    )
    EVENT_LOGGER.log(
        "stream_server_start",
        stream_url=f"http://{args.host}:{args.port}/stream.mjpg",
        image_width=args.width,
        image_height=args.height,
        notes=f"source={args.source}; topic={args.topic}; csv_log={EVENT_LOGGER.path}",
    )

    if args.source == "picamera":
        source = threading.Thread(
            target=run_picamera_source,
            args=(args.width, args.height, args.fps, args.jpeg_quality),
            daemon=True,
        )
    else:
        source = threading.Thread(target=run_ros_compressed_source, args=(args.topic,), daemon=True)

    server = ReusableThreadingHTTPServer((args.host, args.port), StreamHandler)

    def request_shutdown(*_args):
        STOP_EVENT.set()
        server.shutdown()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    source.start()
    print(f"Serving Pupper camera stream at http://{args.host}:{args.port}/stream.mjpg")
    print(f"Snapshot endpoint: http://{args.host}:{args.port}/snapshot.jpg")
    print(f"Source: {args.source}")
    try:
        server.serve_forever()
    finally:
        STOP_EVENT.set()
        server.server_close()
        source.join(timeout=2.0)
        if EVENT_LOGGER is not None:
            EVENT_LOGGER.log("stream_server_stop", frame_index=FRAME_STORE.frame_count)
            EVENT_LOGGER.close()


if __name__ == "__main__":
    main()
