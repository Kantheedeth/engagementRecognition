"""Resumable Route C wrapper around the frozen legacy frame preprocessor."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from experiments_v2.core.artifacts import (
    create_exclusive_dir,
    new_id,
    utc_now,
    write_json_exclusive,
)
from experiments_v2.core.config import load_config
from src.data.preprocess_frames import process_single_video


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "experiments_v2/config/baseline_legacy.json"
EXPECTED_LABELS = {"low": 0, "mid": 1, "high": 2}


@dataclass(frozen=True)
class PreprocessingTask:
    csv_path: str
    label: int
    split: str
    category: str
    video_name: str


def _load_tasks(split_files: dict[str, str]) -> list[PreprocessingTask]:
    tasks: list[PreprocessingTask] = []
    output_keys: set[tuple[str, str, str]] = set()
    for split in ("train", "val", "test"):
        csv_path = Path(split_files[split])
        for line_number, raw_line in enumerate(
            csv_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            fields = raw_line.rsplit(maxsplit=1)
            if len(fields) != 2:
                raise ValueError(f"Malformed record at {csv_path}:{line_number}")
            relative_path, label_text = fields
            try:
                label = int(label_text)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid label at {csv_path}:{line_number}: {label_text!r}"
                ) from exc

            parts = PurePosixPath(relative_path).parts
            if len(parts) < 3 or parts[0] != "videos":
                raise ValueError(
                    f"Unexpected video path at {csv_path}:{line_number}: "
                    f"{relative_path!r}"
                )
            category = parts[1].lower()
            if EXPECTED_LABELS.get(category) != label:
                raise ValueError(
                    f"Label/folder mismatch at {csv_path}:{line_number}: "
                    f"{relative_path!r} label={label}"
                )
            video_name = PurePosixPath(relative_path).stem
            output_key = (split, category, video_name)
            if output_key in output_keys:
                raise ValueError(f"Duplicate preprocessing output key: {output_key}")
            output_keys.add(output_key)
            tasks.append(
                PreprocessingTask(
                    csv_path=relative_path,
                    label=label,
                    split=split,
                    category=category,
                    video_name=video_name,
                )
            )
    return tasks


def _output_paths(
    output_root: Path, task: PreprocessingTask
) -> tuple[Path, Path]:
    npz_path = (
        output_root
        / "yolov5_640x640"
        / task.split
        / task.category
        / f"{task.video_name}.npz"
    )
    pt_path = (
        output_root
        / "mobilenetv3_160x160"
        / task.split
        / task.category
        / f"{task.video_name}.pt"
    )
    return npz_path, pt_path


def _validate_npz(path: Path, task: PreprocessingTask, num_frames: int) -> str | None:
    if not path.is_file():
        return "missing"
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"frames", "label", "video_name", "category"}:
                return f"unexpected keys: {sorted(archive.files)}"
            frames = archive["frames"]
            label = int(archive["label"])
            video_name = str(archive["video_name"])
            category = str(archive["category"])
    except Exception as exc:
        return f"unreadable: {type(exc).__name__}: {exc}"
    if frames.shape != (num_frames, 640, 640, 3):
        return f"shape {frames.shape}"
    if frames.dtype != np.uint8:
        return f"dtype {frames.dtype}"
    if not np.isfinite(frames).all():
        return "non-finite frames"
    if (label, video_name, category) != (
        task.label,
        task.video_name,
        task.category,
    ):
        return "metadata mismatch"
    return None


def _load_pt(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _validate_pt(path: Path, task: PreprocessingTask, num_frames: int) -> str | None:
    if not path.is_file():
        return "missing"
    try:
        archive = _load_pt(path)
        frames = archive["frames"]
        label = int(archive["label"])
        video_name = str(archive["video_name"])
        category = str(archive["category"])
    except Exception as exc:
        return f"unreadable: {type(exc).__name__}: {exc}"
    if tuple(frames.shape) != (num_frames, 3, 160, 160):
        return f"shape {tuple(frames.shape)}"
    if frames.dtype != torch.float32:
        return f"dtype {frames.dtype}"
    if not torch.isfinite(frames).all():
        return "non-finite frames"
    if (label, video_name, category) != (
        task.label,
        task.video_name,
        task.category,
    ):
        return "metadata mismatch"
    return None


def _install_staged(staged: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged, destination)


def _process_task(
    task: PreprocessingTask,
    *,
    video_root: str,
    output_root: str,
    num_frames: int,
    force: bool,
    retries: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_root_path = Path(output_root)
    npz_path, pt_path = _output_paths(output_root_path, task)
    old_npz_error = _validate_npz(npz_path, task, num_frames)
    old_pt_error = _validate_pt(pt_path, task, num_frames)
    if not force and old_npz_error is None and old_pt_error is None:
        return {
            **asdict(task),
            "status": "reused",
            "seconds": time.perf_counter() - started,
            "attempts": 0,
            "npz_action": "reused",
            "pt_action": "reused",
        }

    staging_parent = output_root_path / ".preprocess_staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    last_error = "unknown preprocessing error"
    attempts = 0
    for attempts in range(1, retries + 2):
        with tempfile.TemporaryDirectory(
            prefix=f"{task.split}_{task.category}_{task.video_name}.",
            dir=staging_parent,
        ) as temporary:
            temporary_root = Path(temporary)
            success, message = process_single_video(
                (task.csv_path, str(task.label), task.split),
                str(temporary_root),
                num_frames,
                video_root,
            )
            if not success:
                last_error = message
                continue
            staged_npz, staged_pt = _output_paths(temporary_root, task)
            staged_npz_error = _validate_npz(staged_npz, task, num_frames)
            staged_pt_error = _validate_pt(staged_pt, task, num_frames)
            if staged_npz_error is not None or staged_pt_error is not None:
                last_error = (
                    f"staged validation failed: npz={staged_npz_error}, "
                    f"pt={staged_pt_error}"
                )
                continue

            replace_npz = force or old_npz_error is not None
            replace_pt = force or old_pt_error is not None
            if replace_npz:
                _install_staged(staged_npz, npz_path)
            if replace_pt:
                _install_staged(staged_pt, pt_path)
            return {
                **asdict(task),
                "status": "processed",
                "seconds": time.perf_counter() - started,
                "attempts": attempts,
                "npz_action": (
                    "forced" if force else "created" if old_npz_error == "missing" else "replaced_invalid"
                ) if replace_npz else "reused",
                "pt_action": (
                    "forced" if force else "created" if old_pt_error == "missing" else "replaced_invalid"
                ) if replace_pt else "reused",
            }

    return {
        **asdict(task),
        "status": "failed",
        "seconds": time.perf_counter() - started,
        "attempts": attempts,
        "error": last_error,
        "npz_previous_validation": old_npz_error,
        "pt_previous_validation": old_pt_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive")
    if args.retries < 0:
        raise ValueError("--retries must be non-negative")

    config = load_config(args.config, PROJECT_ROOT)
    dataset = config["dataset"]
    certification = config["certification"]
    num_frames = int(dataset["preprocessing"]["num_frames"])
    preprocessed_input = Path(dataset["preprocessed_input_dir"])
    if preprocessed_input.name != "yolov5_640x640":
        raise ValueError(
            "The legacy preprocessor requires preprocessed_input_dir to end in "
            "yolov5_640x640"
        )
    output_root = preprocessed_input.parent
    video_root = Path(certification["paths"]["dataset_root"])
    tasks = _load_tasks(dataset["split_files"])

    run_id = new_id("RUN")
    report_dir = create_exclusive_dir(
        Path(config["experiment"]["artifacts_root"])
        / "preprocessing"
        / run_id
    )
    started_at = utc_now()
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(
                _process_task,
                task,
                video_root=str(video_root),
                output_root=str(output_root),
                num_frames=num_frames,
                force=args.force,
                retries=args.retries,
            )
            for task in tasks
        ]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Route C preprocessing"
        ):
            results.append(future.result())

    total_seconds = time.perf_counter() - started
    results.sort(key=lambda item: (item["split"], item["category"], item["video_name"]))
    processed_times = [
        float(item["seconds"])
        for item in results
        if item["status"] == "processed"
    ]
    failures = [item for item in results if item["status"] == "failed"]
    report = {
        "run_id": run_id,
        "status": "failed" if failures else "complete",
        "started_at": started_at,
        "finished_at": utc_now(),
        "config_path": config["_config_path"],
        "video_root": str(video_root),
        "output_root": str(output_root),
        "num_frames": num_frames,
        "num_workers": args.num_workers,
        "retries_allowed": args.retries,
        "force": args.force,
        "summary": {
            "scheduled": len(tasks),
            "processed": sum(item["status"] == "processed" for item in results),
            "reused": sum(item["status"] == "reused" for item in results),
            "failed": len(failures),
            "retry_attempts": sum(max(0, int(item["attempts"]) - 1) for item in results),
            "wall_seconds": total_seconds,
            "average_processed_seconds": (
                statistics.fmean(processed_times) if processed_times else 0.0
            ),
            "median_processed_seconds": (
                statistics.median(processed_times) if processed_times else 0.0
            ),
        },
        "results": results,
    }
    report_path = report_dir / "report.json"
    write_json_exclusive(report_path, report)
    print(json.dumps({"report": str(report_path), **report["summary"]}, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
