# Vision-Reactive Pupper

This project makes Pupper detect a person, turn to face them, walk forward,
stop at a configured distance, and wag its tail. The active detection path is
remote RT-DETR: the robot streams camera frames to a laptop, the laptop runs
RT-DETR, and the robot receives the detections back over UDP.

The robot still handles locomotion and tail behavior locally:

- `main.py` converts detections into `/cmd_vel`
- the external `neural_controller` package consumes `/cmd_vel` and runs the
  learned locomotion policy
- `tail_controller.py` commands the ESP32-S3 tail controller over USB serial

## Active Pipeline

```text
robot camera -> robot_camera_stream_server.py -> MJPEG stream -> laptop RT-DETR
                                                           |
                                                           v
                                       UDP person detections back to robot
                                                           |
                                                           v
                                              remote_detection_bridge.py
                                                           |
                                                           v
                                                       /detections
                                                           |
                                                           v
                                                        main.py
                                      +--------------------+-------------------+
                                      |                                        |
                                      v                                        v
                                   /cmd_vel                               USB serial
                                      |                                        |
                                      v                                        v
                           neural_controller policy                    ESP32-S3 tail servo
```

## Detection

The active detector is not the local Hailo node. It is the laptop RT-DETR
stream client in `detr_person_detection/`.

Current detection flow:

1. `camera_ros` publishes `/camera/image_raw/compressed` on the robot.
2. `detr_person_detection/robot_camera_stream_server.py` exposes that feed as
   MJPEG over HTTP.
3. A laptop runs either:
   - `detr_person_detection/laptop_rtdetr_stream_client.py` for RT-DETR-L
   - `detr_person_detection/laptop_rtdetr_r18_stream_client.py` for RT-DETR-R18
4. The laptop script sends person detections back to the robot over UDP.
5. `remote_detection_bridge.py` republishes those packets as
   `vision_msgs/Detection2DArray` on `/detections`.
6. `main.py` picks the best `person` detection and drives approach + tail.

The older local Hailo YOLOv5m path is still in the repo as `hailo_detection.py`,
but it is no longer the active bring-up path.

## Locomotion Policy

The low-level locomotion model is loaded by the external `neural_controller`
package, but this repo now defines the active policy file location.

Put your policy JSON here:

```text
models/policy.json
```

`robot.launch.py` overrides `neural_controller.model_path` to that file.

## Tail Control

The active tail firmware is the PlatformIO project in `tailmovement/`, updated
for the ESP32-S3 + `ESP32Servo` setup.

Communication is programmatic USB CDC serial, not a human serial monitor
workflow. The robot sends commands of the form:

```text
CMD <amplitude_us> <wag_frequency_hz> <envelope_frequency_hz> <timeout_ms>
```

That is a more reasonable control path than manually typing into a serial
monitor because:

- the robot can refresh commands continuously
- the ESP32 has a watchdog and centers the tail if commands stop
- the wag waveform stays local to the MCU, so the robot only sends small
  setpoint updates

If you want a wireless option later, UDP over Wi-Fi from the robot to the
ESP32 would be the next reasonable step. For a USB-tethered tail controller,
USB CDC serial is still the most practical choice.

## Repository Layout

- `main.py`: robot-side control loop
- `approach_controller.py`: maps person box offset/area to `Twist`
- `tail_controller.py`: robot-side USB serial client for the ESP32 tail board
- `remote_detection_bridge.py`: UDP laptop detections -> `/detections`
- `robot.launch.py`: robot base stack + `neural_controller`
- `config.yaml`: approach, tail, and runtime tuning
- `scripts/pi_control.sh`: robot-side foreground runner with clean Ctrl+C shutdown
- `scripts/laptop_control.sh`: laptop-side foreground RT-DETR preview runner
- `models/`: place `policy.json` here
- `tailmovement/`: ESP32-S3 PlatformIO tail firmware
- `detr_person_detection/`: laptop RT-DETR scripts and related utilities

## Prerequisites

- ROS 2 with the robot base stack available
- `vision_msgs` installed in the robot ROS 2 workspace
- a working `neural_controller` package in the robot workspace
- `models/policy.json` placed in this repo
- an ESP32-S3 flashed with the firmware in `tailmovement/`
- a laptop with the repo checked out to run RT-DETR

Install Python dependencies on the robot:

```bash
pip install -r requirements.txt
```

If `vision_msgs` is missing:

```bash
cd /home/pi/pupperv3-monorepo/ros2_ws/src
git clone -b jazzy https://github.com/ros-perception/vision_msgs.git
cd /home/pi/pupperv3-monorepo/ros2_ws
rosdep install --from-paths src/vision_msgs --ignore-src -r -y
source /opt/ros/jazzy/setup.bash
colcon build --packages-select vision_msgs vision_msgs_py --symlink-install
```

## How To Use

### 1. Put the locomotion policy in this repo

Place the policy file at:

```text
models/policy.json
```

### 2. Flash the ESP32-S3 tail firmware

Use the PlatformIO project in `tailmovement/`.

Servo wiring used by the firmware:

- signal -> GPIO 4
- feedback -> GPIO 1
- power -> 5 V
- ground -> GND

Do not power the servo directly from the Pi USB port alone.

### 3. Start the robot side

On the robot:

```bash
./scripts/pi_control.sh
```

This starts:

- `ros2 launch robot.launch.py`
- the MJPEG camera stream server on port `8080`
- the UDP detection bridge on port `9999`
- `main.py`

Press `Ctrl+C` in that terminal to stop the whole robot-side stack cleanly.
The script also kills stale leftover processes from previous runs before it
launches, and it waits until the camera stream is actually serving frames
before declaring the stack ready.

### 4. Start the laptop RT-DETR client

On the laptop, from the same repo:

RT-DETR-L client dependencies:

```bash
pip install ultralytics torch opencv-python
```

RT-DETR-L:

```bash
STREAM_URL=http://<robot-ip>:8080/stream.mjpg \
ROBOT_HOST=<robot-ip> \
./scripts/laptop_control.sh
```

This opens a live laptop window with the Pupper camera feed, the frame
centerline, and RT-DETR person boxes in real time.

By default the RT-DETR laptop client now requires a GPU backend and will pick
`cuda` first, then Apple `mps`. It will not silently fall back to CPU unless
you explicitly pass `--device cpu`.

Press `Ctrl+C` in that terminal to stop the laptop detector cleanly.
The laptop script waits for the robot MJPEG stream to become reachable before
starting RT-DETR, so it is safe to launch the two scripts back-to-back in
separate terminals. The RT-DETR client also reconnects automatically if the
MJPEG stream stalls during a restart.

If you want to run the client directly instead of the wrapper:

```bash
python3 detr_person_detection/laptop_rtdetr_stream_client.py \
  --stream-url http://<robot-ip>:8080/stream.mjpg \
  --robot-host <robot-ip> \
  --preview
```

### 5. Behavior

Once detections are flowing:

- the robot yaws to center the person
- it walks forward while the person box is small
- if the detector is alive but no person is visible, it rotates in place to search
- it slows and stops as the box area approaches `approach.a_stop`
- once stopped, the robot sends tail wag commands to the ESP32-S3

## Tuning

All tuning is in `config.yaml`.

Important parameters:

- `approach.k_yaw`: yaw gain from horizontal image error
- `approach.search_angular_z`: in-place search turn rate when no person is detected
- `approach.a_slow`: area where forward speed starts decreasing
- `approach.a_stop`: area where forward motion stops
- `runtime.detection_image_width`: width used to normalize horizontal box error
- `tail.wag_start_area`: area where tail wagging begins
- `tail.wag_full_area`: area where tail reaches max wag amplitude
- `tail.amplitude_min_us`, `tail.amplitude_max_us`: ESP32 wag amplitude limits
- `tail.wag_frequency_hz`: wag rate generated on the ESP32
- `tail.envelope_frequency_hz`: slower amplitude envelope on the ESP32

The current defaults assume the remote RT-DETR path is using full-width camera
frames around 1400 px wide.

## Topics and Ports

ROS topics:

- `/detections`: subscribed by `main.py`
- `/cmd_vel`: published by `main.py`
- `/camera/image_raw/compressed`: consumed by the MJPEG stream server

Network ports:

- `8080`: MJPEG stream from robot to laptop
- `9999`: UDP detections from laptop back to robot
