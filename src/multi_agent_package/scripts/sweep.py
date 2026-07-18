"""
Grid sweep over a single observation-config parameter.

For each value it rewrites configs/observations.yaml and reruns the experiment,
then ALWAYS restores the file's exact original contents (comments included) in a
finally block -- so a crash mid-sweep can no longer leave the tracked config
overwritten. The swept parameter and its values are CLI-driven.

Run from the repository root:
    python -m multi_agent_package.scripts.sweep --param radius --values 1 2 3 4
"""

import argparse
import yaml
from pathlib import Path

from multi_agent_package.scripts.run_from_config import main as run


def sweep(param_name, values, config_dir: str = "configs"):
    obs_path = Path(config_dir) / "observations.yaml"
    # Capture the exact original bytes (comments, ordering) so we can restore
    # them verbatim regardless of what happens during the sweep.
    original_text = obs_path.read_text()
    try:
        for v in values:
            cfg = yaml.safe_load(original_text)
            cfg["observations"]["params"][param_name] = v
            obs_path.write_text(yaml.dump(cfg))
            print(f"Running sweep with {param_name} = {v}")
            run(config_dir)
    finally:
        obs_path.write_text(original_text)


if __name__ == "__main__":
    p = argparse.ArgumentParser("Sweep one observation param over a list of values")
    p.add_argument("--param", default="radius")
    p.add_argument("--values", nargs="+", default=["1", "2", "3", "4"])
    p.add_argument("--config-dir", default="configs")
    args = p.parse_args()

    # Coerce "3" -> 3 and "0.5" -> 0.5 so YAML gets native scalar types.
    def _coerce(s):
        f = float(s)
        return int(f) if f.is_integer() else f

    sweep(args.param, [_coerce(v) for v in args.values], args.config_dir)
