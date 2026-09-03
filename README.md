# Track-Aware Classroom Group Engagement Recognition

This branch implements a zero-scene behavioral pipeline for classifying classroom
clips as **Low**, **Mid**, or **High** engagement. It combines:

- 40-D role-aware interaction features from YOLO pose + clip-local ByteTrack.
- 8-D group-affect features from RetinaFace + ByteTrack + FER.
- A temporal attention classifier over eight sampled frames.

The interaction redesign is versioned separately from the legacy 32-D descriptor.
It does not overwrite or relabel the old features.

`run_multimodal.py` remains a legacy scene-pipeline runner. It reuses only a
verified legacy 32-D interaction set and will not silently mix the new 40-D
schema into its 616-D matrices.

## Interaction data contract

The new schema is `yolov8_pose_bytetrack_role_pool_40_v1` with shape `(8, 40)`
per video. The default output is:

```text
preprocessed_features/interaction_track_features/
```

The old area-ranked `(8, 32)` files remain in:

```text
preprocessed_features/interaction_features/
```

The new representation has no fixed person slots and no five-person limit.
Every stable student track contributes through permutation-invariant pooling.
Its columns are grouped as follows:

- Teacher evidence: presence, position, coverage, detection confidence, motion,
  role confidence, and frame reliability.
- Student evidence: visible/valid counts, pooled position and spread, box size,
  coverage, detection/pose confidence, motion, and interaction reliability.
- Orientation evidence: `instruction_alignment_proxy`, aligned fraction, and
  explicit reliability values.
- Relations: student pairwise distance statistics, close-pair fraction,
  compactness, student-to-teacher distance, and peer-alignment proxy.
- Audit evidence: unknown/untracked counts, total clip tracks, visibility, and
  frame detection reliability.

A numeric zero caused by missing evidence is always paired with zero reliability.
It is not interpreted as neutral attention, disengagement, or any class label.

## Tracking and roles

Frames are passed chronologically to `YOLO.track(..., persist=True,
tracker="bytetrack.yaml", classes=[0])`. ByteTrack is reset before every clip;
track IDs are anonymous and local to one video.

Roles are assigned once after the complete clip history is collected:

- `teacher`: the strongest instruction-zone candidate only when all configured
  score, coverage, and detector-confidence safeguards pass.
- `student`: a sufficiently covered track outside the ambiguous instructor zone.
- `unknown`: low-coverage, untracked, or role-ambiguous evidence.

No teacher is forced. The selected teacher and all unknown tracks are excluded
from student pooling.

The reused 8-D affect stream is still a **group-affect** aggregate. Its face
tracker is independent of the pose tracker, so this implementation does not
claim that the teacher has been removed from affect. Reliably excluding teacher
affect would require synchronized face-to-body association and a newly versioned
affect extraction; subtracting it from already pooled 8-D arrays is impossible.

The orientation signal is deliberately named an alignment **proxy**. It compares
a weak 2-D pose axis with an instruction target; it is not gaze estimation and
does not establish whether a student is attentive.

## Install

Python 3.10+ is recommended.

```bash
pip install torch torchvision opencv-python numpy tqdm scikit-learn matplotlib
pip install ultralytics insightface transformers lap
```

The implementation was exercised with Ultralytics `8.4.127`. Other versions
must provide `YOLO.track()`, `result.boxes.id`, and tracker `reset()` support.

## Run a diagnostic first

This command processes only two videos, saves `(8,40)` diagnostic arrays in a
separate folder, and writes per-detection audit JSON:

```bash
python src/data/extract_interaction_features.py \
  --max_videos 2 \
  --save_track_details
```

Diagnostic outputs go to:

```text
debug_validation/interaction_track_features/
debug_validation/interaction_tracks/
```

Render a saved audit without rerunning YOLO:

```bash
python src/tools/visualize_interaction_orientation.py \
  --track_json debug_validation/interaction_tracks/train/low/view1000.json
```

The overlay shows local ID, role, detector/pose confidence, alignment proxy, and
alignment reliability. It never labels a person “Attentive” or “Looking Away.”

## Full behavioral pipeline

### 1. Extract new interaction features

```bash
python src/data/extract_interaction_features.py \
  --device auto \
  --save_track_details
```

Existing files are reused only when their shape and complete provenance match.
Use `--overwrite` only when intentionally replacing the full new feature set.
The old 32-D interaction directory is never reused by the new builder.

### 2. Reuse or extract affect

If `preprocessed_features/affect_track_features/extraction_manifest.json` is
valid and all 1,195 arrays exist, do not recompute affect. Otherwise run:

```bash
python src/data/extract_affect_features.py --save_track_details
```

### 3. Build `(8,48)` matrices

```bash
python src/data/build_behavioral_matrices.py
```

This creates a new directory:

```text
feature_matrices_behavioral_track/
```

Each row contains 40 interaction columns followed by 8 affect columns. The
model applies branch-local `LayerNorm`; no test-set statistics are fitted into
the feature builder.

### 4. Provide authoritative split groups

The repository CSVs identify clips and labels but do not identify their source
session or golden-pair relationship. Consequently, session-level isolation
cannot be inferred honestly from filenames.

Create a CSV from the dataset metadata with exactly these columns:

```text
video_path,session_id,golden_pair_id
```

It must contain every path from `train.csv`, `val.csv`, and `test.csv`.
`session_id` is required. `golden_pair_id` may be blank only when the dataset
metadata says the clip has no such group.

Audit it before training:

```bash
python src/tools/audit_split_integrity.py --group_manifest split_groups.csv
```

The audit fails if a video path, session, or non-empty golden-pair group crosses
train/validation/test. It stores SHA-256 hashes of the split CSVs and group map.

### 5. Train and evaluate

```bash
python src/training/train_behavioral.py \
  --group_manifest split_groups.csv

python src/training/evaluate_behavioral.py \
  --group_manifest split_groups.csv
```

The new checkpoint is saved as:

```text
checkpoints/best_model_behavioral_track.pth
```

The legacy `best_model_behavioral.pth` is not accepted. Evaluation verifies the
feature manifest and split-integrity evidence before inference, then reports:

- Accuracy
- Macro-F1
- Balanced accuracy
- Per-class precision, recall, and F1
- Low/Mid/High confusion matrix
- Ordinal MAE with Low=0, Mid=1, High=2

The master runner is also available:

```bash
# One command: audit first, then reuse/extract, build, train, and evaluate.
python run_behavioral_track.py \
  --stage all \
  --group_manifest split_groups.csv

# Or run individual stages.
python run_behavioral_track.py --stage extract
python run_behavioral_track.py --stage build
python run_behavioral_track.py --stage audit --group_manifest split_groups.csv
python run_behavioral_track.py --stage train --group_manifest split_groups.csv
python run_behavioral_track.py --stage eval --group_manifest split_groups.csv
```

`--overwrite_interaction` and `--overwrite_affect` are separate so regenerating
interaction does not accidentally trigger the much slower affect extraction.
With neither overwrite flag, `--stage all` verifies and reuses compatible
existing arrays. It does not rerun the 1,195-video affect inference merely
because the runner was invoked.

Do not create fake session or pair identifiers just to make the audit pass.
`golden_pair_id` may be blank when no authoritative pair is known, but every
`session_id` must be supplied from the real source-session metadata. Therefore,
the complete command intentionally stops before training when that metadata is
absent or inconsistent.

## Old-versus-new ablation

After building the new matrices, run:

```bash
python src/tools/ablation_study.py --group_manifest split_groups.csv
```

The tool computes, rather than hard-codes, majority, affect-only, old
interaction-only, old fusion, new interaction-only, new fusion, and
coordinate-removed baselines. It reports accuracy, Macro-F1, balanced accuracy,
per-class precision/recall/F1, confusion matrices, and ordinal MAE. Add
`--output_json ablation_results.json` to save the computed report. Random-Forest
importance is described as association, not a causal explanation.

## Current evaluation status

The new track-aware role-aware interaction model has **not been trained or
evaluated yet**. No accuracy is claimed for it.

Earlier numbers from the legacy 32-D interaction pipeline do not transfer to
this schema. New metrics may be reported only after full extraction, strict
group-split verification, retraining, and evaluation.

## Scientific limitations

- ByteTrack sees only eight sparsely sampled frames. IDs can fragment or switch;
  they are track associations, not verified identities.
- The configured left-side teacher zone is a transparent camera-specific
  heuristic and must be audited for each camera view.
- Pose alignment is not gaze, attention, or engagement ground truth.
- Facial emotion output is a model estimate, not a verified internal emotion.
- High classification performance does not by itself rule out dataset leakage.
- Session/golden-pair isolation is unverified until an authoritative group map
  passes `audit_split_integrity.py`.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover zone membership, coverage, teacher score and optional teacher
selection, role exclusion, missing-pose reliability, permutation-invariant
pooling, more than five students, and session/golden-pair split leakage.
