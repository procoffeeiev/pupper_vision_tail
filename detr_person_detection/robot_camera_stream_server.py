import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


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
        while True:
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
        rclpy.spin(node)
    finally:
        node.destroy_node()
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
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            count = -1
            while True:
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
                    break
            return

        self.send_error(404)


def main():
    parser = argparse.ArgumentParser(description="Stream Pupper camera frames as MJPEG over HTTP.")
    parser.add_argument("--source", choices=["picamera", "ros-compressed"], default="picamera")
    parser.add_argument("--topic", default="/camera/image_raw/compressed")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    args = parser.parse_args()

    if args.source == "picamera":
        source = threading.Thread(
            target=run_picamera_source,
            args=(args.width, args.height, args.fps, args.jpeg_quality),
            daemon=True,
        )
    else:
        source = threading.Thread(target=run_ros_compressed_source, args=(args.topic,), daemon=True)

    source.start()

    server = ThreadingHTTPServer((args.host, args.port), StreamHandler)
    print(f"Serving Pupper camera stream at http://{args.host}:{args.port}/stream.mjpg")
    print(f"Snapshot endpoint: http://{args.host}:{args.port}/snapshot.jpg")
    print(f"Source: {args.source}")
    server.serve_forever()


if __name__ == "__main__":
    main()
