"""Subprocess and validation helpers shared by legacy adapters."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, TextIO

from experiments_v2.core.artifacts import utc_now


_DOWNLOAD_MARKERS = ("download_path", "downloading", "download model")
_INITIALIZATION_MARKERS = (
    "fer device:",
    "yolo model:",
    "applied providers:",
    "find model:",
    "model ignore:",
    "set det-size:",
    "initializing",
    "loading model",
)
_EXTRACTION_MARKERS = (
    "extracting affect",
    "extracting interaction",
)


def _unbuffered_python_command(command: list[str]) -> list[str]:
    """Add ``-u`` to Python children without changing their program arguments."""
    if not command:
        raise ValueError("A child-process command is required")
    executable = Path(command[0]).name.lower()
    if "python" not in executable or "-u" in command[1:]:
        return list(command)
    return [command[0], "-u", *command[1:]]


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Publish the latest live-process state without exposing partial JSON."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _record_state(
    *,
    state_path: Path,
    events: TextIO,
    process_record: dict[str, Any],
    state: str,
    evidence: str | None = None,
) -> None:
    now = utc_now()
    event: dict[str, Any] = {"state": state, "at": now}
    if evidence:
        event["evidence"] = evidence[:500]
    process_record["state"] = state
    process_record["state_updated_at"] = now
    process_record["states"].append(event)
    events.write(json.dumps(event, sort_keys=True) + "\n")
    events.flush()
    _write_json_atomic(state_path, process_record)


def _state_for_output(line: str) -> str | None:
    normalized = line.strip().lower()
    if not normalized:
        return None
    if any(marker in normalized for marker in _DOWNLOAD_MARKERS):
        return "downloading_model"
    if any(marker in normalized for marker in _EXTRACTION_MARKERS):
        return "extracting"
    if any(marker in normalized for marker in _INITIALIZATION_MARKERS):
        return "initializing_model"
    return None


def run_logged(command: list[str], *, cwd: Path, log_path: Path) -> Mapping[str, Any]:
    """Run a legacy Python child while streaming and recording its true lifecycle.

    There is deliberately no inactivity timeout: a quiet model download or model
    initialization remains active until the operating system reports process exit.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = log_path.with_name("process.json")
    events_path = log_path.with_name("process_events.jsonl")
    if state_path.exists():
        raise FileExistsError(f"Process state already exists: {state_path}")

    executed_command = _unbuffered_python_command(command)
    started_at = utc_now()
    process_record: dict[str, Any] = {
        "state": "starting",
        "pid": None,
        "command": executed_command,
        "cwd": str(cwd.resolve()),
        "started_at": started_at,
        "state_updated_at": started_at,
        "ended_at": None,
        "return_code": None,
        "signal": None,
        "combined_output_log": str(log_path.resolve()),
        "states": [],
    }

    with (
        log_path.open("x", encoding="utf-8") as log,
        events_path.open("x", encoding="utf-8") as events,
    ):
        log.write("COMMAND\n")
        log.write(json.dumps(executed_command))
        log.write("\n\nOUTPUT\n")
        log.flush()

        try:
            child = subprocess.Popen(
                executed_command,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            process_record.update(
                {
                    "ended_at": utc_now(),
                    "launch_error": f"{type(exc).__name__}: {exc}",
                }
            )
            _record_state(
                state_path=state_path,
                events=events,
                process_record=process_record,
                state="failed",
                evidence=process_record["launch_error"],
            )
            raise

        process_record["pid"] = child.pid
        _record_state(
            state_path=state_path,
            events=events,
            process_record=process_record,
            state="starting",
            evidence=f"spawned pid {child.pid}",
        )

        assert child.stdout is not None
        current_state = "starting"
        with child.stdout:
            for line in child.stdout:
                log.write(line)
                log.flush()
                try:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                except (BrokenPipeError, OSError):
                    # The child must continue if an outer UI stops consuming output.
                    pass
                detected_state = _state_for_output(line)
                if detected_state is not None and detected_state != current_state:
                    current_state = detected_state
                    _record_state(
                        state_path=state_path,
                        events=events,
                        process_record=process_record,
                        state=detected_state,
                        evidence=line.strip(),
                    )

        return_code = child.wait()
        _record_state(
            state_path=state_path,
            events=events,
            process_record=process_record,
            state="finalizing",
            evidence="child output closed; return code collected",
        )
        termination_signal = None
        if return_code < 0:
            signal_number = -return_code
            try:
                termination_signal = signal.Signals(signal_number).name
            except ValueError:
                termination_signal = f"SIGNAL_{signal_number}"
        process_record.update(
            {
                "ended_at": utc_now(),
                "return_code": return_code,
                "signal": termination_signal,
            }
        )
        terminal_state = "complete" if return_code == 0 else "failed"
        _record_state(
            state_path=state_path,
            events=events,
            process_record=process_record,
            state=terminal_state,
            evidence=(
                f"return code {return_code}"
                if termination_signal is None
                else f"terminated by {termination_signal}"
            ),
        )

    if return_code != 0:
        termination = (
            f"signal {termination_signal}"
            if termination_signal is not None
            else f"exit code {return_code}"
        )
        raise RuntimeError(
            f"Legacy extractor failed with {termination}; see {log_path}"
        )
    return {
        "command": executed_command,
        "returncode": return_code,
        "signal": termination_signal,
        "log": str(log_path),
        "process": process_record,
    }


def validate_numpy_tree(output_dir: Path, expected_shape: tuple[int, int]) -> int:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required to validate extracted legacy features") from exc

    paths = sorted(output_dir.glob("*/*/*.npy"))
    if not paths:
        raise FileNotFoundError(f"No extracted .npy feature files found under {output_dir}")
    for path in paths:
        array = np.load(path, allow_pickle=False)
        if array.shape != expected_shape:
            raise ValueError(f"{path} has shape {array.shape}; expected {expected_shape}")
        if array.dtype != np.float32:
            raise ValueError(f"{path} has dtype {array.dtype}; expected float32")
        if not np.isfinite(array).all():
            raise ValueError(f"{path} contains non-finite values")
    return len(paths)


def load_legacy_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Legacy extraction manifest was not produced: {path}")
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid legacy manifest: {path}")
    return manifest
