# Vision-Reactive Pupper with Motion Tail

Closed-loop pupper v3 behavior: detect a person, walk toward them, and wag a
servo-driven tail faster as they get closer. Runs on a Raspberry Pi 5 with a
Hailo NPU for detection and a PCA9685 driving a 9 g servo for the tail.

Project proposal: *Vision-Reactive Pupper with Motion Tail*, ROB-UY 2004
(Wang, Lin, Liang).

---

## Pipeline

```
 camera ─▶ hailo_detection.py ─▶ /detections ─┐
                                              ▼
                                        main.py
                                     (ApproachController + TailController)
                                              │
                    /cmd_vel ◀─────────────────┼───▶ PCA9685 servo (tail)
```

- `hailo_detection.py` — Hailo NPU YOLOv5m → `vision_msgs/Detection2DArray`
  on `/detections`. *(Proposal specifies RT-DETR-R18; this initial pipeline
  uses the proven Hailo YOLOv5m. RT-DETR swap is a follow-up — same output
  contract, only this file changes.)*
- `approach_controller.py` — visual servo: yaw from bbox x-offset, forward
  speed that decreases linearly between `a_slow` and `a_stop`.
- `tail_controller.py` — background thread that sinusoidally sweeps a servo
  at a frequency scaled from bbox area.
- `main.py` — ROS 2 node wiring everything together; publishes Twist on
  `/cmd_vel`.
- `config.yaml` — all tunables (gains, areas, servo limits, loop rate).

---

## Running on the robot

Two terminals. First one brings up the base stack and detector:

```bash
./scripts/system_start.sh
```

Leave it running. In a second terminal:

```bash
python main.py
```

`Ctrl+C` in the `main.py` terminal publishes several zero-velocity commands
before tearing down, and re-centers the tail servo.

### Topics

| Topic | Direction | Type | Purpose |
| --- | --- | --- | --- |
| `/detections` | subscribed | `vision_msgs/Detection2DArray` | Per-frame detections from Hailo. |
| `/cmd_vel` | published | `geometry_msgs/Twist` | Body-frame velocity to the neural controller. |

### First-time setup

`vision_msgs` must be installed into the robot's ROS 2 workspace:

```bash
cd /home/pi/pupperv3-monorepo/ros2_ws/src
git clone -b jazzy https://github.com/ros-perception/vision_msgs.git
cd /home/pi/pupperv3-monorepo/ros2_ws
sudo rosdep init  # skip if already initialized
rosdep update
rosdep install --from-paths src/vision_msgs --ignore-src -r -y
source /opt/ros/jazzy/setup.bash
colcon build --packages-select vision_msgs vision_msgs_py --symlink-install
```

Python extras for the tail servo:

```bash
pip install -r requirements.txt
```

---

## Hardware wiring (tail)

- 9 g hobby servo → PCA9685 channel **15** (override with `tail.channel` in
  `config.yaml`).
- PCA9685 on the Pi's I²C bus (default `/dev/i2c-1`).
- External 5–6 V supply on the PCA9685 V+ rail — do **not** run the servo off
  the Pi 5 V.

If the PCA9685 isn't present (dev laptop, unit-testing the control loop), the
tail controller logs a warning and drops to dry-run mode — locomotion still
works.

---

## Tuning

All tunables live in `config.yaml`. The parameters most worth touching on
first-run on the real robot:

| Param | Start | What it controls |
| --- | --- | --- |
| `approach.k_yaw` | 2.0 | How hard yaw pulls toward the target. Too high → oscillation. |
| `approach.a_slow`, `approach.a_stop` | 20k / 60k px² | Distance-proxy thresholds. Measure bbox area at 1 m and at 0.3 m and set accordingly. |
| `approach.v_max` | 0.35 m/s | Forward speed cap. Lower while tuning. |
| `tail.angle_min_deg`, `tail.angle_max_deg` | 60° / 120° | Tail sweep limits. Tight cable → narrow; loose → wide. |
| `runtime.loop_hz` | 10 | Control rate. Detector runs ~5 Hz, so >10 just repeats frames. |

---

## Performance targets (from the proposal)

- Detection rate at 1 m: **DR > 0.80**
- Approach success (2 m → <0.30 m stop): **AS > 0.70**
- Tail response latency: **L̄ < 500 ms**

The watchdog in `main.py` idles both locomotion and tail when no detection
frame has arrived for >1 s.

---

## Layout

```
pupper_vision_tail/
├── main.py                   # control loop (this project's novel code)
├── approach_controller.py    # visual-servo approach
├── tail_controller.py        # PCA9685 wag driver
├── config.yaml               # tunable params
├── requirements.txt
├── hailo_detection.py        # Hailo NPU detector ROS node
├── utils.py                  # HailoAsyncInference helpers
├── fisheye_converter.py      # double-sphere → equirectangular unwarp
├── camera_params.yaml        # fisheye intrinsics
├── coco.txt                  # class labels
├── robot.launch.py           # base-stack launch file
├── yolov5m_wo_spp_60p.hef    # Hailo model (~35 MB)
└── scripts/
    └── system_start.sh       # brings up base stack + detector
```

---

## Follow-ups

- **RT-DETR-R18 swap** — replace the Hailo YOLOv5m backbone with RT-DETR in
  INT8. Benchmark DR & latency against the existing pipeline (Week 1 of the
  proposal).
- **Stretch goal** — HSV ball-tracking mode. Add a second target-picker path
  driving the same `ApproachController`.
