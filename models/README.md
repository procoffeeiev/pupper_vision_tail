Place the locomotion policy JSON here as:

`models/policy_latest.json`

`robot.launch.py` overrides `neural_controller.model_path` to this file, so the
robot stack will load the policy from this repository instead of expecting it
inside the external `neural_controller` package.
