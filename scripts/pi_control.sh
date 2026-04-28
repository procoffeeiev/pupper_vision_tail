#!/bin/bash
# Run the robot-side runtime in the foreground and stop cleanly on Ctrl+C.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
POLICY_PATH="$PROJECT_ROOT/models/policy.json"

source ~/.bashrc
cd "$PROJECT_ROOT"

if [[ ! -f "$POLICY_PATH" ]]; then
    echo "Missing locomotion policy: $POLICY_PATH"
    exit 1
fi

ROS2_PID=""
STREAM_PID=""
BRIDGE_PID=""
MAIN_PID=""

cleanup() {
    trap - INT TERM EXIT

    if [[ -n "$MAIN_PID" ]] && kill -0 "$MAIN_PID" 2>/dev/null; then
        kill -INT "$MAIN_PID" 2>/dev/null || true
    fi
    if [[ -n "$BRIDGE_PID" ]] && kill -0 "$BRIDGE_PID" 2>/dev/null; then
        kill -TERM "$BRIDGE_PID" 2>/dev/null || true
    fi
    if [[ -n "$STREAM_PID" ]] && kill -0 "$STREAM_PID" 2>/dev/null; then
        kill -TERM "$STREAM_PID" 2>/dev/null || true
    fi
    if [[ -n "$ROS2_PID" ]] && kill -0 "$ROS2_PID" 2>/dev/null; then
        kill -TERM "$ROS2_PID" 2>/dev/null || true
    fi

    wait "$MAIN_PID" 2>/dev/null || true
    wait "$BRIDGE_PID" 2>/dev/null || true
    wait "$STREAM_PID" 2>/dev/null || true
    wait "$ROS2_PID" 2>/dev/null || true
}

trap cleanup INT TERM EXIT

ros2 launch "$PROJECT_ROOT/robot.launch.py" &
ROS2_PID=$!

sleep 2

python3 "$PROJECT_ROOT/detr_person_detection/robot_camera_stream_server.py" \
    --source ros-compressed \
    --topic /camera/image_raw/compressed \
    --host 0.0.0.0 \
    --port 8080 &
STREAM_PID=$!

python3 "$PROJECT_ROOT/remote_detection_bridge.py" --host 0.0.0.0 --port 9999 &
BRIDGE_PID=$!

python3 "$PROJECT_ROOT/main.py" &
MAIN_PID=$!

echo "robot-side stack running"
echo "Ctrl+C stops ros2 launch, camera stream, detection bridge, and main.py cleanly"

wait "$MAIN_PID"
