"""
sync_tf_events_to_wandb.py

Usage:
    python sync_tf_events_to_wandb.py \
        --base-dir src/baselines/IQL/logs \
        --project "MARL-Predator-Prey-Project" \
        --entity "hu-marl-research-group" \
        --dry-run  # optional, shows what would happen without uploading
"""

import os
import json
import glob
import argparse
import re
from collections import defaultdict
from datetime import datetime

import wandb
from tensorboard.backend.event_processing import event_accumulator

# Optional: YAML config loading if present
try:
    import yaml
except Exception:
    yaml = None


def find_latest_tfevent(run_dir):
    """Return path to the latest tfevents file in run_dir, or None."""
    files = sorted(glob.glob(os.path.join(run_dir, "*tfevents*")))
    if not files:
        return None
    # choose the newest by mtime (safer on Windows with many runs)
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def nice_name_from_folder(folder_name):
    """Make a prettier experiment name from a folder name.

    Examples:
      '07-10-2025_21-16-45_lr=0.01_gamma=0.99' -> '07-10-2025_lr0.01_gamma0.99'
      fallback: 'run-20251007-2116'
    """
    base = os.path.basename(folder_name).strip()
    # remove seconds and replace spaces/backslashes
    base = re.sub(r"[:\s]+", "_", base)
    # simplify param patterns like lr=0.001 -> lr0.001
    base = re.sub(r"([a-zA-Z]+)=([0-9eE\.\-]+)", r"\1\2", base)
    # limit length
    if len(base) > 60:
        base = base[:60]
    # fallback to timestamp if weird
    if not base:
        base = "run-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    return base


def load_run_config(run_dir):
    """Try to load config.json or config.yaml from run_dir if present."""
    for filename in ("config.json", "params.json", "config.yaml", "params.yaml"):
        path = os.path.join(run_dir, filename)
        if os.path.isfile(path):
            try:
                if path.lower().endswith(".json"):
                    with open(path, "r") as f:
                        return json.load(f)
                elif yaml is not None:
                    with open(path, "r") as f:
                        return yaml.safe_load(f)
                else:
                    print(
                        "YAML config found but PyYAML not installed — skipping YAML load."
                    )
            except Exception as e:
                print(f"Failed to parse config file {path}: {e}")
    return None


def sync_one_run(event_file, run_dir, project, entity, dry_run=False, exp_name=None):
    """Read scalars from event_file and upload them to a new wandb run."""
    if exp_name is None:
        exp_name = nice_name_from_folder(run_dir)

    print(
        f"\n--> Preparing run for '{run_dir}' (event file: {os.path.basename(event_file)})"
    )
    print(f"    will create W&B run name: '{exp_name}'")

    if dry_run:
        print("    [dry-run] skipping actual upload.")
        return

    # create wandb run
    wandb.init(
        project=project,
        entity=entity,
        name=exp_name,
        reinit=True,
        sync_tensorboard=True,
    )

    # try to set config if exists
    config = load_run_config(run_dir)
    if config:
        try:
            wandb.config.update(config, allow_val_change=True)
            print("    set wandb.config from file")
        except Exception as e:
            print("    failed to set config:", e)

    # read TF event file
    ea = event_accumulator.EventAccumulator(
        event_file, size_guidance={"scalars": 0}
    )  # load all scalars
    ea.Reload()

    scalar_tags = ea.Tags().get("scalars", [])
    if not scalar_tags:
        print("    no scalar tags found in this event file — skipping.")
        wandb.finish()
        return

    # Build step -> {tag: value} map (use wall_time to order)
    step_map = defaultdict(lambda: {"_wall_time": None})
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        for ev in events:
            step = int(ev.step) if ev.step is not None else None
            # use tuple (step, wall_time) as key to avoid collisions if step is None
            key = (step, float(ev.wall_time) if ev.wall_time is not None else 0.0)
            # we'll store by key but later sort by wall_time
            if step_map.get(key) is None:
                step_map[key] = {"_wall_time": ev.wall_time}
            # put the scalar value
            step_map[key][tag] = ev.value

    # convert to list and sort by wall_time then step
    items = []
    for (step, wall_time), d in step_map.items():
        # normalize step to int (if None, use -1)
        s = -1 if step is None else step
        items.append((wall_time if wall_time is not None else 0.0, s, d))
    items.sort(key=lambda x: (x[0], x[1]))

    # Upload in order: group all metrics per step and log once.
    print(f"    logging {len(items)} distinct timepoints to W&B...")
    for wall_time, step, metrics in items:
        # remove internal key
        metrics_to_log = {k: v for k, v in metrics.items() if not k.startswith("_")}
        if not metrics_to_log:
            continue
        try:
            # log with explicit step -- wandb will align curves by step
            wandb.log(metrics_to_log, step=step)
        except Exception as e:
            print(f"    warning: failed to wandb.log at step {step}: {e}")

    print("    finishing run.")
    wandb.finish()


def main(base_dir, project, entity=None, dry_run=False, pattern="*"):
    base_dir = os.path.abspath(base_dir)
    if not os.path.isdir(base_dir):
        raise RuntimeError(f"Base dir not found: {base_dir}")

    # find immediate subdirs (each expected to be an experiment folder). If there are tfevents also directly under base_dir,
    # optionally include them.
    candidate_dirs = [
        d for d in glob.glob(os.path.join(base_dir, pattern)) if os.path.isdir(d)
    ]
    if not candidate_dirs:
        print("No experiment subdirectories found under:", base_dir)
        return

    print(f"Found {len(candidate_dirs)} candidate run directories.")
    for run_dir in sorted(candidate_dirs):
        event_file = find_latest_tfevent(run_dir)
        if event_file is None:
            print(f"Skipping {run_dir}: no tfevents files found.")
            continue

        # Try to make a nice experiment name from folder (optionally further parse params)
        exp_name = nice_name_from_folder(os.path.basename(run_dir))
        sync_one_run(
            event_file, run_dir, project, entity, dry_run=dry_run, exp_name=exp_name
        )

    print(
        "\nDone. If you ran without --dry-run, check your W&B project for the new runs."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Parent folder containing experiment subfolders",
    )
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--entity", default=None, help="W&B entity (team/user)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't upload, just print actions"
    )
    parser.add_argument(
        "--pattern", default="*", help="Glob pattern for subfolders (default '*')"
    )
    args = parser.parse_args()
    main(
        args.base_dir,
        args.project,
        entity=args.entity,
        dry_run=args.dry_run,
        pattern=args.pattern,
    )
