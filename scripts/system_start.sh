#!/bin/bash
# Start the robot policy, camera node, and object detector.

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Project root is one level up from scripts directory
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project directory
cd "$PROJECT_ROOT"

source ~/.bashrc

ROS2_PID=""
DETECTOR_PID=""

cleanup() {
    trap - INT TERM EXIT

    if [[ -n "${DETECTOR_PID:-}" ]] && kill -0 "$DETECTOR_PID" 2>/dev/null; then
        kill -TERM "$DETECTOR_PID" 2>/dev/null || true
    fi

    if [[ -n "${ROS2_PID:-}" ]] && kill -0 "$ROS2_PID" 2>/dev/null; then
        kill -TERM "$ROS2_PID" 2>/dev/null || true
    fi

    wait "$DETECTOR_PID" 2>/dev/null || true
    wait "$ROS2_PID" 2>/dev/null || true
}

trap cleanup INT TERM EXIT

ros2 launch "$PROJECT_ROOT/robot.launch.py" &
ROS2_PID=$!

python "$PROJECT_ROOT/hailo_detection.py" &
DETECTOR_PID=$!

wait "$DETECTOR_PID"
