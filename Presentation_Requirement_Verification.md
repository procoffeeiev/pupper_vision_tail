# Final Presentation Requirement Verification

Source requirements: `ROB-UY-2004 - Final Project Presentation.docx`

Deck: `Vision_Reactive_Pupper_Final_Presentation.pptx`

## Content Requirements

| Requirement | Deck coverage | Status |
| --- | --- | --- |
| No longer than 5 minutes | 10 concise slides, built for roughly 25-30 seconds per slide. | Satisfied |
| Introduction to the technical approach to locomotion/manipulation | Slides 2-3 introduce the person-reactive approach scenario and system architecture. | Satisfied |
| Describe the scenario where the state estimator would be used | Slide 2 describes the camera-to-approach-and-tail scenario. | Satisfied |
| Images encouraged | Slide 7 uses the saved RT-DETR detector output from the repo. | Satisfied |
| Problem definition | Slide 4 defines the state-estimation problem. | Satisfied |
| Clearly define input measurements | Slide 3 lists camera frames, detection boxes, scores, and frame age. | Satisfied |
| Clearly define state variables being estimated | Slides 3-4 define `target_valid`, bearing `theta`, bbox area `A`, mode, and tail amplitude. | Satisfied |
| Clearly define intermediate variables | Slide 4 defines bbox center, unprojected ray, confidence threshold, and watchdog age. | Satisfied |
| Coordinate-frame diagram | Slide 4 shows camera frame `C`, robot base `B`, person target `P`, optical axis, and bearing `theta`. | Satisfied |
| Method of state estimation | Slide 5 describes the deterministic image-based estimator and measurement mapping. | Satisfied |
| Cite prior work | Slides 5 and 10 cite visual servo control, Double Sphere camera model, RT-DETR, and COCO. | Satisfied |
| General class of filter | Slide 5 identifies the estimator as deterministic image-based state estimation for visual servo control. | Satisfied |
| System block diagram showing sensor inputs and estimated states | Slide 3 shows sensor measurements, the estimator, estimated state signals, and actuation consumers. | Satisfied |
| Highlight similarity/difference from standard filters | Slide 5 compares it to standard filters and notes the absence of prediction/covariance/particle updates. | Satisfied |
| Experiment and results description | Slide 8 defines the intended detection, approach, and tail-latency experiments. Slide 9 states current implementation results and evidence limitations. | Partially satisfied |
| Equation used to quantify performance | Slide 8 provides equations for detection rate, approach success, and tail latency. | Satisfied |
| Performance plots with labeled axes | No aggregate metric data or result logs were found in the repo, so the revised deck does not include fake or placeholder plots. | Not satisfied due to missing data |
| Conclusions with 2-4 key takeaways | Slide 9 lists three takeaways. | Satisfied |
| IEEE-format references | Slide 10 uses IEEE-style references. | Satisfied |

## Metric Data Source

The proposal defines the metric targets:

- Detection rate: true-positive percentage across 50 frames at 0.5 m, 1 m, and 2 m. Target: >80% at 1 m.
- Approach success: trials where Pupper reaches within 30 cm from a 2 m start. Target: >70% over 10 trials.
- Tail response latency: person detection to tail wag onset. Target: <500 ms.

Repository search found no CSV, log, video, or summary file containing completed aggregate metric trials. The only result artifact present is:

- `detr_person_detection/laptop_rtdetr_stream_detection.jpg`, showing one RT-DETR-L detection sample with confidence 0.68, horizontal error -0.221, and area ratio 0.001.

Therefore, the revised deck treats the proposal numbers as targets and explicitly marks aggregate detection rate, approach success, and tail latency data as not found in this repo snapshot.
