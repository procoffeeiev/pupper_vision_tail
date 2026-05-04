#!/usr/bin/env python3
"""Append a manual trial-result row to the experiment CSV logs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_logging import build_logger  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--final-distance-m", required=True)
    parser.add_argument("--approach-success", choices=["true", "false"])
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(os.environ.get("EXPERIMENT_LOG_DIR", PROJECT_ROOT / "data" / "experiments")),
    )
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    final_distance = float(args.final_distance_m)
    approach_success = args.approach_success or ("true" if final_distance <= 0.30 else "false")

    logger = build_logger(
        side="manual",
        component="trial_result",
        session_id=args.session_id,
        trial_id=args.trial_id,
        log_dir=args.log_dir,
    )
    try:
        notes = f"approach_success={approach_success}"
        if args.notes:
            notes = f"{notes}; {args.notes}"
        logger.log(
            "manual_trial_result",
            final_distance_m=args.final_distance_m,
            approach_success=approach_success,
            notes=notes,
        )
    finally:
        logger.close()
    print(f"Wrote manual trial result to {logger.path}")


if __name__ == "__main__":
    main()
