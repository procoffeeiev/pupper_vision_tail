#!/usr/bin/env python3
"""Summarize Vision-Reactive Pupper experiment CSV logs."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def read_rows(log_dir: Path, session_id: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(log_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("session_id") == session_id:
                    row["_source_file"] = path.name
                    rows.append(row)
    rows.sort(key=lambda r: float_or_none(r.get("timestamp_unix_s")) or 0.0)
    return rows


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def float_or_none(value: str | None) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except ValueError:
        return None


def pct(numer: int, denom: int) -> str:
    if denom <= 0:
        return "n/a"
    return f"{100.0 * numer / denom:.1f}%"


def mean(values: list[float]) -> str:
    if not values:
        return "n/a"
    return f"{statistics.mean(values):.3f}"


def median(values: list[float]) -> str:
    if not values:
        return "n/a"
    return f"{statistics.median(values):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--log-dir", type=Path, default=Path("data/experiments"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = read_rows(args.log_dir, args.session_id)
    detection_rows = [r for r in rows if r.get("event") == "detection_frame"]
    person_rows = [r for r in detection_rows if r.get("ground_truth_person_present") == "true"]
    no_person_rows = [r for r in detection_rows if r.get("ground_truth_person_present") == "false"]

    detected_person = [r for r in person_rows if truthy(r.get("target_detected", ""))]
    false_positive = [r for r in no_person_rows if truthy(r.get("target_detected", ""))]

    inference_times = [v for r in detection_rows if (v := float_or_none(r.get("inference_time_s"))) is not None]

    manual_results = [r for r in rows if r.get("event") == "manual_trial_result"]
    final_distances = [v for r in manual_results if (v := float_or_none(r.get("final_distance_m"))) is not None]
    successes = [r for r in manual_results if truthy(r.get("approach_success", ""))]

    approach_stops = [r for r in rows if r.get("event") == "approach_stop"]
    yaw_errors = [
        abs(v)
        for r in approach_stops
        if (v := float_or_none(r.get("yaw_error_rad"))) is not None
    ]
    durations = [
        v
        for r in approach_stops
        if (v := float_or_none(r.get("approach_duration_s"))) is not None
    ]

    lines = [
        f"# Experiment Summary: `{args.session_id}`",
        "",
        "## Slide-Ready Metrics",
        "",
        f"- Detection rate: {pct(len(detected_person), len(person_rows))} ({len(detected_person)}/{len(person_rows)} frames)",
        f"- False-positive rate: {pct(len(false_positive), len(no_person_rows))} ({len(false_positive)}/{len(no_person_rows)} frames)",
        f"- Mean inference time: {mean(inference_times)} s",
        f"- Approach success: {pct(len(successes), len(manual_results))} ({len(successes)}/{len(manual_results)} trials)",
        f"- Final distance mean: {mean(final_distances)} m",
        f"- Final distance median: {median(final_distances)} m",
        f"- Stop centering error mean: {mean(yaw_errors)} rad",
        f"- Approach duration median: {median(durations)} s",
        "",
        "## Trial Distances",
        "",
    ]

    if manual_results:
        lines.append("| Trial | Final Distance (m) | Success |")
        lines.append("| --- | ---: | --- |")
        for i, row in enumerate(manual_results, 1):
            trial = row.get("trial_id") or f"trial_{i:02d}"
            lines.append(f"| {trial} | {row.get('final_distance_m', '')} | {row.get('approach_success', '')} |")
    else:
        lines.append("No `manual_trial_result` rows found.")

    lines.extend(
        [
            "",
            "## Data Counts",
            "",
            f"- CSV rows: {len(rows)}",
            f"- Detection frames with person ground truth: {len(person_rows)}",
            f"- Detection frames with no-person ground truth: {len(no_person_rows)}",
            f"- Approach stop events: {len(approach_stops)}",
            f"- Manual trial results: {len(manual_results)}",
        ]
    )

    output = "\n".join(lines) + "\n"
    if args.output is None:
        args.output = args.log_dir / f"{args.session_id}_summary.md"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(output)
    print(f"Wrote summary to {args.output}")


if __name__ == "__main__":
    main()
