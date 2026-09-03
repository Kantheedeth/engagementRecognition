#!/usr/bin/env python3
"""CLI for the additive V2 legacy golden-pair baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments_v2.pipeline.experiment_runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "experiments_v2" / "config" / "baseline_legacy.json",
    )
    args = parser.parse_args()
    runner = ExperimentRunner(project_root=PROJECT_ROOT, config_path=args.config)
    if args.command == "plan":
        print(json.dumps(runner.plan(), indent=2))
        return
    try:
        results = runner.run()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
