#!/usr/bin/env python3
"""Verify that videos, sessions, and golden pairs do not cross dataset splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split_integrity import audit_split_integrity, format_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv_dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--group_manifest",
        type=Path,
        required=True,
        help=(
            "CSV with video_path,session_id,golden_pair_id. Session IDs must come "
            "from dataset metadata; do not infer them from class labels."
        ),
    )
    args = parser.parse_args()
    print(format_report(audit_split_integrity(args.csv_dir, args.group_manifest)))


if __name__ == "__main__":
    main()
