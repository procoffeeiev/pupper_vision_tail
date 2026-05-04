# One-Hour Experiment Plan

Core project claim:

```text
Pupper can detect a person, face them, approach, and stop near them.
```

Use only the metrics that test that claim.

## Metrics

1. **Detection rate**

```text
detection_rate = detected_person_frames / total_sampled_person_frames
```

For the one-hour run, one detection trial means `50` sampled frames. If the
person appears in `47` of those frames, detection rate is `47/50 = 94%`.

2. **False-positive rate**

```text
false_positive_rate = false_person_detections / no_person_frames
```

3. **End-to-end approach success**

```text
success if final_distance_m <= 0.30
approach_success_rate = successful_trials / total_trials
```

4. **Final stopping distance**

```text
final_distance_m = distance value recorded at the stop point
```

5. **Stop centering error**

```text
stop_centering_error = abs(yaw_error_rad at approach_stop)
```

Approach duration and mean inference time are secondary numbers only.

## One-Hour Workflow

### 0-10 min: connect and start logs

Pick one session id:

```bash
export SESSION_ID=hw_eval_001
```

Start Pupper:

```bash
SESSION_ID=$SESSION_ID ./scripts/pi_control.sh
```

All CSV logs go to:

```text
data/experiments/
```

### 10-15 min: detection sanity trial

Put a person about `1.0 m` in front of Pupper and collect 50 frames:

```bash
SESSION_ID=$SESSION_ID \
TRIAL_ID=det_person_1m \
CONDITION_DISTANCE_M=1.0 \
GROUND_TRUTH_PERSON_PRESENT=true \
MAX_FRAMES=50 \
./scripts/laptop_control.sh
```

Remove the person and collect 30 no-person frames:

```bash
SESSION_ID=$SESSION_ID \
TRIAL_ID=det_no_person \
GROUND_TRUTH_PERSON_PRESENT=false \
MAX_FRAMES=30 \
./scripts/laptop_control.sh
```

### 15-45 min: 5 approach trials

For each trial:

1. Place Pupper `2.0 m` from a stationary person.
2. Start the laptop detector.
3. Let Pupper approach until it stops.
4. Record the final distance value at that stop.
5. Reset Pupper to the start position and repeat.

Laptop command for each trial:

```bash
SESSION_ID=$SESSION_ID \
TRIAL_ID=approach_01 \
CONDITION_DISTANCE_M=2.0 \
GROUND_TRUTH_PERSON_PRESENT=true \
./scripts/laptop_control.sh
```

Record final distance at the stop:

```bash
./scripts/record_trial_result.py \
  --session-id $SESSION_ID \
  --trial-id approach_01 \
  --final-distance-m 0.24
```

The recorder automatically marks success when:

```text
final_distance_m <= 0.30
```

Run `approach_01` through `approach_05`.

### 45-50 min: summarize data

```bash
./scripts/summarize_experiment.py --session-id $SESSION_ID
```

This writes:

```text
data/experiments/hw_eval_001_summary.md
```

### 50-60 min: update slides

Use the summary values in the final results slide:

- Detection rate
- False-positive rate
- Approach success rate
- Final distance mean/median
- Stop centering error mean

Do not report tail latency.

## What To Put In Results

Minimum acceptable results table:

| Metric | Result |
| --- | --- |
| Detection rate at 1 m | from summary |
| False-positive rate | from summary |
| Approach success | from summary |
| Mean final distance | from summary |
| Mean stop centering error | from summary |

This is enough for a homework-scale final presentation.
