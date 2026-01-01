"""
Simple grid sweep over config parameters.

This script modifies YAML values programmatically and reruns experiments.
"""

import copy
import yaml
from pathlib import Path

from multi_agent_package.scripts.run_from_config import run


def sweep(param_name, values):
    base_cfg = Path("configs/observations.yaml")
    original = yaml.safe_load(base_cfg.read_text())

    for v in values:
        cfg = copy.deepcopy(original)
        cfg["observations"]["params"][param_name] = v

        base_cfg.write_text(yaml.dump(cfg))
        print(f"Running sweep with {param_name} = {v}")
        run("configs")

    base_cfg.write_text(yaml.dump(original))


if __name__ == "__main__":
    sweep("radius", [1, 2, 3, 4])
