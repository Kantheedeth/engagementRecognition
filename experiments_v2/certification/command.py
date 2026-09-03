"""Safe preflight and execution gate for the official baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments_v2.certification.preflight import build_preflight_report
from experiments_v2.core.config import load_config
from experiments_v2.pipeline.experiment_runner import ExperimentRunner


def preflight(*, project_root: Path, config_path: Path) -> dict[str, Any]:
    config = load_config(config_path, project_root)
    return build_preflight_report(config)


def certify_baseline(
    *, project_root: Path, config_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = preflight(project_root=project_root, config_path=config_path)
    if not report["ready"]:
        raise RuntimeError(
            "Certification preflight is NOT READY; no run or baseline was created"
        )
    runner = ExperimentRunner(project_root=project_root, config_path=config_path)
    environment_info = {
        **report["environment"],
        "certification_ready": True,
        "selected_certification_path": report["selected_path"],
    }
    results = runner.run(
        environment_info=environment_info, certify_official=True
    )
    return report, results
