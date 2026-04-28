#!/bin/bash
# Run the laptop-side RT-DETR preview in the foreground and stop cleanly on Ctrl+C.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

STREAM_URL="${STREAM_URL:-http://10.20.19.129:8080/stream.mjpg}"
ROBOT_HOST="${ROBOT_HOST:-10.20.19.129}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/detr_person_detection/rtdetr-l.pt}"
DEVICE="${DEVICE:-gpu}"

cd "$PROJECT_ROOT"

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Missing RT-DETR model: $MODEL_PATH"
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

PYTHON_BIN="python3"
if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
fi

echo "waiting for robot camera stream..."
if ! wait_for_url "$STREAM_URL" 60; then
    echo "camera stream did not become reachable in time: $STREAM_URL"
    exit 1
fi

echo "laptop RT-DETR preview running"
echo "stream: $STREAM_URL"
echo "robot: $ROBOT_HOST"
echo "device: $DEVICE"
echo "Ctrl+C stops the detector cleanly"

exec "$PYTHON_BIN" "$PROJECT_ROOT/detr_person_detection/laptop_rtdetr_stream_client.py" \
    --stream-url "$STREAM_URL" \
    --robot-host "$ROBOT_HOST" \
    --model "$MODEL_PATH" \
    --device "$DEVICE" \
    --preview
