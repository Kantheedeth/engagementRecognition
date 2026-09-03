"""Command line interface for V2 readiness and certified baseline execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments_v2.certification.command import certify_baseline, preflight


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "experiments_v2/config/baseline_legacy.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "certify-baseline"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    if args.command == "preflight":
        report = preflight(project_root=PROJECT_ROOT, config_path=args.config)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["ready"] else 2)

    try:
        report, results = certify_baseline(
            project_root=PROJECT_ROOT, config_path=args.config
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"CERTIFICATION BLOCKED: {exc}")
        raise SystemExit(2) from exc
    print(json.dumps({"preflight": report, "results": results}, indent=2))


if __name__ == "__main__":
    main()
