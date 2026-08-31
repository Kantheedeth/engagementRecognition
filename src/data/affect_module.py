"""Track-aware group affect extraction.

The public ``AffectModule`` interface accepts one RGB uint8 frame at a time and
returns seven canonical emotion probabilities plus a scalar reliability score.
Detection, tracking, and FER backends are injectable so each component can be
ablated independently.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Protocol, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EMOTION_NAMES = (
    "anger",
    "disgust",
    "fear",
    "happiness",
    "sadness",
    "surprise",
    "neutral",
)

_LABEL_ALIASES = {
    "angry": "anger",
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "happy": "happiness",
    "happiness": "happiness",
    "sad": "sadness",
    "sadness": "sadness",
    "surprise": "surprise",
    "surprised": "surprise",
    "neutral": "neutral",
}


@dataclass(frozen=True)
class FaceDetection:
    """A RetinaFace observation in pixel coordinates."""

    bbox: np.ndarray
    landmarks: np.ndarray | None
    score: float


class FaceDetector(Protocol):
    def detect(self, frame_rgb_np: np.ndarray) -> list[FaceDetection]: ...


class FaceTracker(Protocol):
    def update(
        self,
        detections: Sequence[FaceDetection],
        frame_shape: tuple[int, int],
    ) -> dict[int, int]: ...

    def reset(self) -> None: ...


class EmotionClassifier(nn.Module):
    """Base class for FER backends returning canonical-order probabilities."""

    def forward(self, aligned_faces_rgb: Sequence[np.ndarray]) -> torch.Tensor:
        raise NotImplementedError


class RetinaFaceDetector:
    """InsightFace RetinaFace wrapper.

    ``buffalo_s`` uses the lightweight RetinaFace-500MF detector and provides
    five facial landmarks. InsightFace's detector consumes BGR arrays, so the
    public RGB input is converted internally.
    """

    def __init__(
        self,
        model_name: str = "buffalo_s",
        model_root: str = "~/.insightface",
        det_size: tuple[int, int] = (640, 640),
        det_threshold: float = 0.45,
        max_faces: int = 64,
        providers: Sequence[str] | None = None,
    ) -> None:
        os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
        try:
            import onnxruntime as ort
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError(
                "RetinaFace requires insightface and onnxruntime. Install "
                "requirements-affect.txt before running affect extraction."
            ) from exc

        if providers is None:
            available = set(ort.get_available_providers())
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in available
                else ["CPUExecutionProvider"]
            )

        self.max_faces = max_faces
        self.app = FaceAnalysis(
            name=model_name,
            root=model_root,
            allowed_modules=["detection"],
            providers=list(providers),
        )
        ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
        self.app.prepare(
            ctx_id=ctx_id,
            det_size=det_size,
            det_thresh=det_threshold,
        )

    def detect(self, frame_rgb_np: np.ndarray) -> list[FaceDetection]:
        _validate_rgb_frame(frame_rgb_np)
        frame_bgr = cv2.cvtColor(frame_rgb_np, cv2.COLOR_RGB2BGR)
        faces = self.app.get(frame_bgr, max_num=self.max_faces)
        return [
            FaceDetection(
                bbox=np.asarray(face.bbox, dtype=np.float32),
                landmarks=(
                    np.asarray(face.kps, dtype=np.float32)
                    if face.kps is not None
                    else None
                ),
                score=float(face.det_score),
            )
            for face in faces
        ]


class ByteTrackFaceTracker:
    """Adapter from RetinaFace detections to Ultralytics ByteTrack."""

    def __init__(
        self,
        track_high_threshold: float = 0.45,
        track_low_threshold: float = 0.10,
        new_track_threshold: float = 0.45,
        track_buffer: int = 8,
        match_threshold: float = 0.80,
    ) -> None:
        try:
            from ultralytics.engine.results import Boxes
            from ultralytics.trackers.byte_tracker import BYTETracker
        except ImportError as exc:
            raise ImportError(
                "ByteTrack requires ultralytics and lap>=0.5.12. Install "
                "requirements-affect.txt before enabling tracking."
            ) from exc

        self._boxes_type = Boxes
        args = SimpleNamespace(
            track_high_thresh=track_high_threshold,
            track_low_thresh=track_low_threshold,
            new_track_thresh=new_track_threshold,
            track_buffer=track_buffer,
            match_thresh=match_threshold,
            fuse_score=True,
        )
        self._tracker = BYTETracker(args)

    def update(
        self,
        detections: Sequence[FaceDetection],
        frame_shape: tuple[int, int],
    ) -> dict[int, int]:
        if detections:
            rows = np.asarray(
                [
                    [*det.bbox.tolist(), float(det.score), 0.0]
                    for det in detections
                ],
                dtype=np.float32,
            )
            data = torch.from_numpy(rows)
        else:
            data = torch.zeros((0, 6), dtype=torch.float32)

        boxes = self._boxes_type(data, orig_shape=frame_shape)
        tracked = self._tracker.update(boxes)
        if tracked is None or len(tracked) == 0:
            return {}

        # ByteTrack rows are x1,y1,x2,y2,track_id,score,class,detection_idx.
        return {int(row[7]): int(row[4]) for row in np.asarray(tracked)}

    def reset(self) -> None:
        self._tracker.reset()


class HuggingFaceFERClassifier(EmotionClassifier):
    """PyTorch FER backend using a Hugging Face image classifier.

    The default model is a FER2013-fine-tuned ViT. It is the fully public,
    reproducible reference backend; a smaller fine-tuned TorchScript model can
    be selected with ``TorchScriptFERClassifier`` without changing the affect
    pipeline.
    """

    def __init__(
        self,
        model_id: str = "abhilash88/face-emotion-detection",
        device: str | torch.device | None = None,
        local_files_only: bool = False,
        source_emotions: Sequence[str] = (
            "anger",
            "disgust",
            "fear",
            "happiness",
            "sadness",
            "surprise",
            "neutral",
        ),
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
        except ImportError as exc:
            raise ImportError(
                "The Hugging Face FER backend requires transformers. Install "
                "requirements-affect.txt first."
            ) from exc

        self.device = _select_torch_device(device)
        self.processor = AutoImageProcessor.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        )
        self.model = AutoModelForImageClassification.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        )
        self.model.to(self.device).eval()
        self.register_buffer(
            "canonical_indices",
            _canonical_indices(source_emotions).to(self.device),
            persistent=False,
        )

    @torch.inference_mode()
    def forward(self, aligned_faces_rgb: Sequence[np.ndarray]) -> torch.Tensor:
        if not aligned_faces_rgb:
            return torch.empty((0, len(EMOTION_NAMES)), device=self.device)
        inputs = self.processor(images=list(aligned_faces_rgb), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        logits = self.model(pixel_values=pixel_values).logits
        probs = torch.softmax(logits.float(), dim=-1)
        return probs.index_select(1, self.canonical_indices)


class TorchScriptFERClassifier(EmotionClassifier):
    """Backend for an optimized MobileNet/EfficientNet FER TorchScript model."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str | torch.device | None = None,
        input_size: int = 224,
        source_emotions: Sequence[str] = EMOTION_NAMES,
    ) -> None:
        super().__init__()
        self.device = _select_torch_device(device)
        self.input_size = input_size
        self.model = torch.jit.load(checkpoint_path, map_location=self.device)
        self.model.eval()
        self.register_buffer(
            "canonical_indices",
            _canonical_indices(source_emotions).to(self.device),
            persistent=False,
        )
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1),
            persistent=False,
        )

    @torch.inference_mode()
    def forward(self, aligned_faces_rgb: Sequence[np.ndarray]) -> torch.Tensor:
        if not aligned_faces_rgb:
            return torch.empty((0, len(EMOTION_NAMES)), device=self.device)
        batch = torch.stack(
            [torch.from_numpy(np.ascontiguousarray(face)).permute(2, 0, 1) for face in aligned_faces_rgb]
        ).float()
        batch = batch.to(self.device) / 255.0
        batch = F.interpolate(
            batch,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        logits = self.model((batch - self.mean) / self.std)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        probs = torch.softmax(logits.float(), dim=-1)
        return probs.index_select(1, self.canonical_indices)


class AffectModule(nn.Module):
    """RetinaFace + optional ByteTrack + FER group affect module.

    Args:
        detector: Face detector implementing ``detect(frame_rgb_np)``.
        fer_model: FER module returning ``N x 7`` canonical probabilities.
        tracker: Optional tracker. Required when ``use_tracking=True``.
        expected_faces: Face count at which count reliability saturates.
        emotion_momentum: EMA weight assigned to the previous track emotion.
        missed_detection_decay: Reliability decay during a cached one-frame gap.
        max_feature_age: Number of missed frames allowed for cached group affect.
    """

    def __init__(
        self,
        detector: FaceDetector,
        fer_model: EmotionClassifier,
        tracker: FaceTracker | None = None,
        use_tracking: bool = True,
        expected_faces: int = 8,
        emotion_momentum: float = 0.60,
        missed_detection_decay: float = 0.35,
        max_feature_age: int = 1,
        aligned_face_size: int = 224,
    ) -> None:
        super().__init__()
        if use_tracking and tracker is None:
            raise ValueError("tracker is required when use_tracking=True")
        if expected_faces <= 0:
            raise ValueError("expected_faces must be positive")
        if not 0.0 <= emotion_momentum < 1.0:
            raise ValueError("emotion_momentum must be in [0, 1)")
        if not 0.0 <= missed_detection_decay <= 1.0:
            raise ValueError("missed_detection_decay must be in [0, 1]")

        self.detector = detector
        self.fer_model = fer_model
        self.tracker = tracker
        self.use_tracking = use_tracking
        self.expected_faces = expected_faces
        self.emotion_momentum = emotion_momentum
        self.missed_detection_decay = missed_detection_decay
        self.max_feature_age = max_feature_age
        self.aligned_face_size = aligned_face_size

        self._track_emotions: dict[int, torch.Tensor] = {}
        self._last_group_affect: torch.Tensor | None = None
        self._last_reliability = 0.0
        self._missed_frames = 0
        self.last_observations: list[dict] = []

    def reset(self) -> None:
        """Reset all temporal state before processing a new video."""
        if self.tracker is not None:
            self.tracker.reset()
        self._track_emotions.clear()
        self._last_group_affect = None
        self._last_reliability = 0.0
        self._missed_frames = 0
        self.last_observations = []

    @torch.inference_mode()
    def forward(self, frame_rgb_np: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract group affect and reliability from one HxWx3 uint8 RGB frame."""
        _validate_rgb_frame(frame_rgb_np)
        detections = self.detector.detect(frame_rgb_np)
        frame_shape = frame_rgb_np.shape[:2]

        track_ids: dict[int, int] = {}
        if self.use_tracking and self.tracker is not None:
            track_ids = self.tracker.update(detections, frame_shape)

        if not detections:
            return self._missing_detection_result()

        aligned_faces = [
            align_face(frame_rgb_np, det, output_size=self.aligned_face_size)
            for det in detections
        ]
        probabilities = self.fer_model(aligned_faces)
        if probabilities.shape != (len(detections), len(EMOTION_NAMES)):
            raise ValueError(
                "FER backend returned shape "
                f"{tuple(probabilities.shape)}; expected {(len(detections), len(EMOTION_NAMES))}"
            )
        if not torch.isfinite(probabilities).all():
            raise ValueError("FER backend returned non-finite probabilities")

        probabilities = probabilities.float()
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)
        smoothed = []
        for detection_idx, current in enumerate(probabilities):
            track_id = track_ids.get(detection_idx)
            if track_id is not None and track_id in self._track_emotions:
                previous = self._track_emotions[track_id].to(current.device)
                current = self.emotion_momentum * previous + (1.0 - self.emotion_momentum) * current
                current = current / current.sum().clamp_min(1e-8)
            if track_id is not None:
                self._track_emotions[track_id] = current.detach().cpu()
            smoothed.append(current)

        self.last_observations = []
        for detection_idx, (detection, probability) in enumerate(zip(detections, smoothed)):
            probability_cpu = probability.detach().float().cpu()
            top_index = int(torch.argmax(probability_cpu).item())
            self.last_observations.append(
                {
                    "track_id": track_ids.get(detection_idx),
                    "tracking_status": (
                        "tracked" if detection_idx in track_ids else "unconfirmed"
                    ),
                    "bbox_xyxy": [float(value) for value in detection.bbox],
                    "detection_confidence": float(detection.score),
                    "emotion_probabilities": {
                        name: float(probability_cpu[index].item())
                        for index, name in enumerate(EMOTION_NAMES)
                    },
                    "top_emotion": EMOTION_NAMES[top_index],
                    "emotion_confidence": float(probability_cpu[top_index].item()),
                }
            )

        face_probs = torch.stack(smoothed)
        weights = torch.tensor(
            [max(0.0, min(1.0, det.score)) for det in detections],
            dtype=face_probs.dtype,
            device=face_probs.device,
        )
        group_affect = (face_probs * weights[:, None]).sum(dim=0) / weights.sum().clamp_min(1e-8)

        count_score = min(len(detections) / float(self.expected_faces), 1.0)
        mean_detection_confidence = float(weights.mean().item())
        reliability_value = count_score * mean_detection_confidence
        reliability = torch.tensor(
            reliability_value,
            dtype=group_affect.dtype,
            device=group_affect.device,
        )

        self._last_group_affect = group_affect.detach().cpu()
        self._last_reliability = reliability_value
        self._missed_frames = 0
        return group_affect, reliability

    def _missing_detection_result(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._missed_frames += 1
        self.last_observations = []
        device = _module_device(self.fer_model)
        if (
            self.use_tracking
            and self._last_group_affect is not None
            and self._missed_frames <= self.max_feature_age
        ):
            reliability = self._last_reliability * (
                self.missed_detection_decay ** self._missed_frames
            )
            return self._last_group_affect.clone().to(device), torch.tensor(reliability, device=device)

        return torch.zeros(len(EMOTION_NAMES), device=device), torch.tensor(0.0, device=device)


def align_face(
    frame_rgb_np: np.ndarray,
    detection: FaceDetection,
    output_size: int = 224,
) -> np.ndarray:
    """Align a face using RetinaFace five-point landmarks, with bbox fallback."""
    if detection.landmarks is not None and detection.landmarks.shape == (5, 2):
        reference = np.asarray(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        reference *= output_size / 112.0
        transform, _ = cv2.estimateAffinePartial2D(
            detection.landmarks.astype(np.float32),
            reference,
            method=cv2.LMEDS,
        )
        if transform is not None:
            return cv2.warpAffine(
                frame_rgb_np,
                transform,
                (output_size, output_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

    height, width = frame_rgb_np.shape[:2]
    x1, y1, x2, y2 = detection.bbox.astype(float)
    pad_x = 0.10 * max(0.0, x2 - x1)
    pad_y = 0.10 * max(0.0, y2 - y1)
    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(width, int(x2 + pad_x))
    y2 = min(height, int(y2 + pad_y))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid face bounding box: {detection.bbox.tolist()}")
    return cv2.resize(
        frame_rgb_np[y1:y2, x1:x2],
        (output_size, output_size),
        interpolation=cv2.INTER_LINEAR,
    )


def _canonical_indices(source_emotions: Sequence[str]) -> torch.Tensor:
    normalized = []
    for label in source_emotions:
        key = str(label).strip().lower().replace("_", " ")
        key = _LABEL_ALIASES.get(key, key)
        normalized.append(key)
    if set(normalized) != set(EMOTION_NAMES) or len(normalized) != len(EMOTION_NAMES):
        raise ValueError(
            "source_emotions must contain each canonical emotion exactly once; "
            f"received {tuple(source_emotions)}"
        )
    return torch.tensor([normalized.index(name) for name in EMOTION_NAMES], dtype=torch.long)


def _select_torch_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _module_device(module: nn.Module) -> torch.device:
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(), None)
    return buffer.device if buffer is not None else torch.device("cpu")


def _validate_rgb_frame(frame_rgb_np: np.ndarray) -> None:
    if not isinstance(frame_rgb_np, np.ndarray):
        raise TypeError("frame_rgb_np must be a numpy array")
    if frame_rgb_np.dtype != np.uint8:
        raise TypeError(f"frame_rgb_np must be uint8, got {frame_rgb_np.dtype}")
    if frame_rgb_np.ndim != 3 or frame_rgb_np.shape[2] != 3:
        raise ValueError(f"frame_rgb_np must have shape HxWx3, got {frame_rgb_np.shape}")
