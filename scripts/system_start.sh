#!/bin/bash
# Start the robot base stack, camera stream server, and remote detection bridge.

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Project root is one level up from scripts directory
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project directory
cd "$PROJECT_ROOT"

source ~/.bashrc

POLICY_PATH="$PROJECT_ROOT/models/policy_latest.json"

if [[ "${SKIP_NEURAL_CONTROLLER_POLICY_CHECK:-0}" != "1" ]] && [[ ! -f "$POLICY_PATH" ]]; then
    echo "Missing locomotion policy: $POLICY_PATH"
    echo "Put your neural_controller policy JSON there or set SKIP_NEURAL_CONTROLLER_POLICY_CHECK=1"
    exit 1
fi

ROS2_PID=""
STREAM_PID=""
BRIDGE_PID=""

cleanup() {
    trap - INT TERM EXIT

    if [[ -n "${BRIDGE_PID:-}" ]] && kill -0 "$BRIDGE_PID" 2>/dev/null; then
        kill -TERM "$BRIDGE_PID" 2>/dev/null || true
    fi

    if [[ -n "${STREAM_PID:-}" ]] && kill -0 "$STREAM_PID" 2>/dev/null; then
        kill -TERM "$STREAM_PID" 2>/dev/null || true
    fi

    if [[ -n "${ROS2_PID:-}" ]] && kill -0 "$ROS2_PID" 2>/dev/null; then
        kill -TERM "$ROS2_PID" 2>/dev/null || true
    fi

    wait "$BRIDGE_PID" 2>/dev/null || true
    wait "$STREAM_PID" 2>/dev/null || true
    wait "$ROS2_PID" 2>/dev/null || true
}

trap cleanup INT TERM EXIT

ros2 launch "$PROJECT_ROOT/robot.launch.py" &
ROS2_PID=$!

python3 "$PROJECT_ROOT/detr_person_detection/robot_camera_stream_server.py" \
    --source ros-compressed \
    --topic /camera/image_raw/compressed \
    --host 0.0.0.0 \
    --port 8080 &
STREAM_PID=$!

python3 "$PROJECT_ROOT/remote_detection_bridge.py" --host 0.0.0.0 --port 9999 &
BRIDGE_PID=$!

wait "$BRIDGE_PID"
