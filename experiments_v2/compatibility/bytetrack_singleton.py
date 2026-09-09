"""V2-only normalization for Ultralytics 8.3.0 singleton indexing.

Ultralytics ByteTrack 8.3.0 converts bounding boxes to a NumPy array but can
retain confidence masks as PyTorch tensors.  A one-element ``torch.bool``
tensor is interpreted by NumPy as scalar integer index 0/1 rather than a row
mask.  Passing an equivalent NumPy-backed ``Boxes`` object avoids that
cross-library scalar coercion while preserving every value and all tracker
configuration/state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch


COMPATIBILITY_FIX_ID = "bytetrack_singleton_numpy_mask_v1"


@dataclass
class ByteTrackSingletonNumpyShim:
    """Handle and provenance for one installed instance-level shim."""

    tracker: Any
    original_update: Callable[..., Any]
    trigger_count: int = 0

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "id": COMPATIBILITY_FIX_ID,
            "scope": "V2 only",
            "trigger": "exactly one ByteTrack detection",
            "semantics": "representation normalization only",
            "normalization": "torch-backed Boxes to value-identical NumPy-backed Boxes",
            "trigger_count": self.trigger_count,
        }

    def restore(self) -> None:
        """Restore the original instance method without touching site-packages."""
        self.tracker.update = self.original_update


def _numpy_values(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def apply_bytetrack_singleton_numpy_mask_v1(
    affect_module: Any,
) -> ByteTrackSingletonNumpyShim:
    """Normalize only singleton ByteTrack inputs on one A1 module instance."""
    face_tracker = getattr(affect_module, "tracker", None)
    if face_tracker is None or not hasattr(face_tracker, "_tracker"):
        raise TypeError("METHOD_A1 does not expose the expected ByteTrack instance")
    byte_tracker = face_tracker._tracker
    original_update = byte_tracker.update
    registration = ByteTrackSingletonNumpyShim(
        tracker=byte_tracker,
        original_update=original_update,
    )

    def compatible_update(results: Any, img: Any = None) -> Any:
        if len(results) != 1:
            return original_update(results, img)

        source_data = _numpy_values(results.data)
        normalized = results.numpy()
        normalized_data = _numpy_values(normalized.data)
        if source_data.dtype != normalized_data.dtype or not np.array_equal(
            source_data, normalized_data
        ):
            raise RuntimeError(
                f"{COMPATIBILITY_FIX_ID} refused a value-changing normalization"
            )
        registration.trigger_count += 1
        return original_update(normalized, img)

    byte_tracker.update = compatible_update
    return registration
