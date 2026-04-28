#!/bin/bash
# Run the robot-side runtime in the foreground and stop cleanly on Ctrl+C.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
POLICY_PATH="$PROJECT_ROOT/models/policy.json"

set +u
if [[ -f /opt/ros/jazzy/setup.bash ]]; then
    source /opt/ros/jazzy/setup.bash
fi
if [[ -f "$HOME/pupperv3-monorepo/ros2_ws/install/setup.bash" ]]; then
    source "$HOME/pupperv3-monorepo/ros2_ws/install/setup.bash"
fi
source ~/.bashrc
set -u
cd "$PROJECT_ROOT"

if [[ ! -f "$POLICY_PATH" ]]; then
    echo "Missing locomotion policy: $POLICY_PATH"
    exit 1
fi

wait_for_url() {
    local url="$1"
    local timeout_s="$2"
    local start_ts
    start_ts="$(date +%s)"

    while true; do
        if python3 -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2).read(1)' "$url" >/dev/null 2>&1; then
            return 0
        fi

        if (( "$(date +%s)" - start_ts >= timeout_s )); then
            return 1
        fi
        sleep 0.5
    done
}

kill_stale() {
    pkill -f "$PROJECT_ROOT/main.py" 2>/dev/null || true
    pkill -f "$PROJECT_ROOT/remote_detection_bridge.py" 2>/dev/null || true
    pkill -f "$PROJECT_ROOT/detr_person_detection/robot_camera_stream_server.py" 2>/dev/null || true
    pkill -f "ros2 launch $PROJECT_ROOT/robot.launch.py" 2>/dev/null || true
    fuser -k 8080/tcp 2>/dev/null || true
    fuser -k 9999/udp 2>/dev/null || true
    sleep 1
}

require_running() {
    local pid="$1"
    local name="$2"
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "$name failed to start"
        exit 1
    fi
}

ROS2_PID=""
STREAM_PID=""
BRIDGE_PID=""
MAIN_PID=""

terminate_process() {
    local pid="$1"
    local signal="${2:-TERM}"

    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi

    kill "-$signal" "$pid" 2>/dev/null || true
    for _ in {1..50}; do
        if ! kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        sleep 0.1
    done

    kill -KILL "$pid" 2>/dev/null || true
}

cleanup() {
    trap - INT TERM EXIT

    terminate_process "$MAIN_PID" INT
    terminate_process "$BRIDGE_PID" TERM
    terminate_process "$STREAM_PID" TERM
    terminate_process "$ROS2_PID" TERM

    wait "$MAIN_PID" 2>/dev/null || true
    wait "$BRIDGE_PID" 2>/dev/null || true
    wait "$STREAM_PID" 2>/dev/null || true
    wait "$ROS2_PID" 2>/dev/null || true

    kill_stale
}

trap cleanup INT TERM EXIT

kill_stale

ros2 launch "$PROJECT_ROOT/robot.launch.py" &
ROS2_PID=$!
sleep 2
require_running "$ROS2_PID" "ros2 launch"

python3 "$PROJECT_ROOT/detr_person_detection/robot_camera_stream_server.py" \
    --source ros-compressed \
    --topic /camera/image_raw/compressed \
    --host 0.0.0.0 \
    --port 8080 &
STREAM_PID=$!
sleep 1
require_running "$STREAM_PID" "camera stream server"

python3 "$PROJECT_ROOT/remote_detection_bridge.py" --host 0.0.0.0 --port 9999 &
BRIDGE_PID=$!
sleep 1
require_running "$BRIDGE_PID" "detection bridge"

python3 "$PROJECT_ROOT/main.py" &
MAIN_PID=$!
sleep 1
require_running "$MAIN_PID" "main loop"

if ! wait_for_url "http://127.0.0.1:8080/snapshot.jpg" 30; then
    echo "camera stream did not become ready in time"
    exit 1
fi

echo "robot-side stack running"
echo "Ctrl+C stops ros2 launch, camera stream, detection bridge, and main.py cleanly"
echo "camera stream ready at http://127.0.0.1:8080/stream.mjpg"

wait "$MAIN_PID"
