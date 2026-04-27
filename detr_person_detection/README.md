## RT-DETR Detection Utilities

This folder contains the RT-DETR scripts used for the active remote-detection
pipeline and a few related experiments.

### Active remote-detection path

- `robot_camera_stream_server.py`
  - Runs on the robot.
  - Streams `/camera/image_raw/compressed` as MJPEG over HTTP.
- `laptop_rtdetr_stream_client.py`
  - Runs on the laptop.
  - Uses RT-DETR-L from `rtdetr-l.pt`.
  - Sends detections back to the robot over UDP.
- `laptop_rtdetr_r18_stream_client.py`
  - Runs on the laptop.
  - Uses RT-DETR-R18.
  - Sends detections back to the robot over UDP.

The robot receives those UDP detections through `remote_detection_bridge.py`
in the repo root, which republishes them as `/detections`.

### Other utilities

- `detect_person_rtdetr.py`
  - Run RT-DETR on a saved image and draw person boxes.
- `robot_person_detector_rtdetr.py`
  - Standalone robot-side direct-camera RT-DETR script.
- `robot_person_detector_rtdetr_ros.py`
  - Standalone robot-side RT-DETR script for one ROS camera frame.

### Model files

- `rtdetr-l.pt`
  - Local RT-DETR-L checkpoint used by the laptop stream client.
- `rtdetr_r18vd_dec3_6x_coco_from_paddle.pth`
  - RT-DETR-R18 checkpoint for the R18 client path.
