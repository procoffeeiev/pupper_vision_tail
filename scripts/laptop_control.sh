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

PYTHON_BIN="python3"
if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
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
