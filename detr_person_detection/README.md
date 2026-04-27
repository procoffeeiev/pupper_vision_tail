## RT-DETR Person Detection

This folder contains the RT-DETR-based person detection variants developed for the
Pupper vision-following project.

### Main files

- `detect_person_rtdetr.py`
  - Run RT-DETR on a saved image and draw person boxes.
- `laptop_rtdetr_stream_client.py`
  - Laptop-side live stream client using the larger RT-DETR-L model.
- `laptop_rtdetr_r18_stream_client.py`
  - Laptop-side live stream client using the smaller RT-DETR-R18 model.
- `robot_camera_stream_server.py`
  - Robot-side MJPEG camera stream server for laptop inference.
- `robot_person_detector_rtdetr.py`
  - Robot-side direct-camera RT-DETR inference script.
- `robot_person_detector_rtdetr_ros.py`
  - Robot-side RT-DETR inference from ROS camera topics.

### Model files

- `rtdetr-l.pt`
  - Larger RT-DETR model used for higher-accuracy laptop inference.
- `rtdetr_r18vd_dec3_6x_coco_from_paddle.pth`
  - Smaller RT-DETR-R18 checkpoint used for smoother laptop inference.

### Notes

- `rtdetr-l.onnx` is intentionally not committed here because it exceeds GitHub's
  100 MB file size limit.
- Current recommended architecture:
  - Pupper streams camera frames
  - Laptop runs RT-DETR locally
  - Laptop computes `horizontal_error` and `area_ratio`
