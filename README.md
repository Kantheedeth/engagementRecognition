# Classroom Group Engagement Recognition

This project studies lightweight group-engagement recognition for the **lecture-only** portion of OUC-CGE. As of 2026-09-26, the activity-reviewed subset contains **308 clips**. The current local experiment deliberately excludes scene embeddings and uses two behavioral branches:

- 32 interaction features from YOLOv8 pose detections
- 8 group-affect features from RetinaFace, ByteTrack smoothing, and FER

The output classes are Low, Mid, and High group engagement.

The lecture-only run achieved **95.48% macro-F1 and 96.77% accuracy (30/31 test clips)**. This is a preliminary result on filtered original splits, not established generalization to new recordings or classrooms.

### Documentation and implementation status

This branch (`docs/lecture-only-results-20260926`) includes the **32-interaction + 8-affect sampled-tracking implementation**, its lecture-only runner, core tests and exact 308-clip selection list. The remote `feat/track-aware-affect-fusion` branch contains a distinct 40-interaction + 8-affect role-aware ByteTrack implementation. The results below do not evaluate that implementation; do not mix their schemas, matrices or checkpoints.

Videos, preprocessed arrays, feature matrices, model weights and run outputs are not included in Git. Collaborators need compatible V3 matrices or must regenerate them from their own authorized dataset copy. See [collaborator setup](experiments/lecture_only/README.md#collaborator-setup) for the portable selection workflow and dependencies.

## Current system: V3 sampled behavioral model

Each clip is represented by eight uniformly sampled 640×640 frames. Clip durations can vary; do not assume every retained clip is exactly ten seconds.

```text
8 sampled frames
      │
      ├── YOLOv8 pose ── Hungarian association across 8 frames ── 32 interaction features
      │
      └── RetinaFace + ByteTrack + FER ─────────────────────────── 8 affect features
                                    │
                         (8, 40) behavioral sequence
                                    │
             Linear(32→64) + Linear(8→64) → 128-dimensional fusion
                                    │
                 temporal multi-head attention + mean pooling
                                    │
                              Low / Mid / High
```

The interaction output does not contain absolute bounding-box coordinates such as `cx`, `cy`, width, or height. Coordinates are still used internally for speaker-zone selection, identity association, and displacement features. Removing direct coordinate inputs does not by itself establish freedom from shortcuts or camera invariance.

### Speaker and listener definition

- Instruction zone: normalized `x ∈ [0.0, 0.27]`.
- A detected person in the zone is treated as the speaker.
- When no speaker is detected, the center of the instruction zone, `[0.135, 0.50]`, is used as the interaction target.
- People outside the zone are treated as listeners, except a previously associated speaker who temporarily moves outside it.
- IDs are anonymous, reset for every clip, and are not persistent student identities across videos.

### V3 behavioral corrections

The first manual audit showed that the former nose-to-eye vertical-distance rule was not a reliable head-pitch estimator for this camera view. V3 therefore uses observed normalized nose-to-shoulder elevation as a coarse **raised/lowered-head proxy**:

- Lowered head: elevation `< 0.130`
- Raised head: elevation `≥ 0.155`
- Upright posture: elevation `≥ 0.120`
- Very-low-head proxy: elevation `≤ 0.060`
- Orientation-consensus threshold: score `≥ 0.66`

Missing face or shoulder keypoints are treated as unknown for head/posture ratios. The model does **not** claim to detect sleeping: `very_low_head_proxy_ratio` describes an observed pose only.

The submitted development reviews were assigned to Kan (audit frames 01–15), Pan (19–20), and Bright (22–36). Frames 16–18 were excluded because they show the out-of-scope round-table clip `view1230`; frame 21 was left unannotated. Reviewers covered mostly disjoint frames, and reviews informed threshold development. This is calibration evidence, not independent validation or inter-rater reliability. The historical calibration aggregation needs its frame-index mapping rechecked before quoting reviewer-wide proxy-validation scores.

## Dataset status and evaluation limits

Earlier filtering used camera angle, but the same view can contain lecture and group-work activities. The new manual filtering uses the one-speaker/multiple-listeners activity rule. Inattentive listeners remain eligible; low engagement is not a reason to exclude a lecture clip.

The isolated snapshot reuses existing matrices without preprocessing or feature re-extraction:

| Split | Low | Mid | High | Total |
|---|---:|---:|---:|---:|
| Train | 133 | 55 | 64 | 252 |
| Validation | 10 | 3 | 12 | 25 |
| Test | 16 | 6 | 9 | 31 |
| Total | 159 | 64 | 85 | 308 |

The older angle-filtered matrix set is preserved separately: 938 train, 124 validation and 132 test clips (1,194 total). Its extraction manifest records 1,195 preprocessed clips; `view1230` was excluded from its training CSV. A later filesystem inventory found 1,152 raw clips in `videos/`; raw-folder counts and historical matrix counts describe different snapshots and should not be conflated.

### Corrections to earlier split claims

- Clip identity is **class folder + filename**, not the numeric `view` ID alone.
- The older 1,194 CSV rows have 1,194 distinct class/filename keys but 1,144 distinct numeric IDs. Twenty-three numeric IDs occur across splits in different class folders. This does **not** establish exact-video duplication or contradictory labels.
- No identical class/filename key was shared between those splits. The 308-clip snapshot also had no cross-split identical matrix hashes; neither check rules out similar or related source footage.
- The same camera, people and clothing indicate a shared recording setup, not necessarily one uninterrupted recording. Filename adjacency and gaps alone do not establish temporal continuity or parent-session boundaries.
- Parent sessions and clip independence remain unverified. Visually confirmed continuity blocks can support a more conservative evaluation, but must not be relabelled as confirmed lecture sessions without evidence.

The new snapshot preserves the original split assignments after filtering. It is an **original clip-split benchmark within the selected setup**, not a recording-held-out, cross-camera or unseen-student evaluation. Historical manual calibration also needs checking for overlap with any proposed independent test set.

## Current lecture-only result

Run: `experiments/lecture_only/runs/smoothed_seed42/`. Source: `reports/metrics.json`, `reports/train.log` and `run_config.json` in that run directory. The recorded seed is 42; the selected checkpoint is epoch 40, with validation accuracy 88% (22/25). Early stopping occurred at epoch 55.

| Actual class | Predicted Low | Predicted Mid | Predicted High |
|---|---:|---:|---:|
| Low | 16 | 0 | 0 |
| Mid | 0 | 6 | 0 |
| High | 0 | 1 | 8 |

- Macro-F1: **95.48%**
- Accuracy: **96.77% (30/31)**
- Always-Low baseline macro-F1: **22.70%**
- One error: `high/view2480.mp4` predicted Mid.

The task and test population changed, so this score is not directly comparable to the older 78.46% result. Activity filtering may better match the speaker–listener representation, but the test subset may also be easier. Neither explanation has been isolated experimentally.

Only six Mid clips are tested, and only three appear in validation. One additional test error would reduce macro-F1 to approximately 90.74–92.66%, depending on its class pair. This illustrates score sensitivity; it is not a confidence interval. The result is one run, not a multi-seed mean or proof that the features measure genuine attention.

### Loss and training configuration

The run uses weighted cross-entropy with square-root inverse-frequency weights computed from **training labels only** and normalized to mean one: Low 0.7506, Mid 1.1673, High 1.0821. `smoothed` names the class-weight rule; label smoothing is **0.0**.

Branch dimension 64, four heads, dropout 0.20, batch size 32, AdamW learning rate 0.001, weight decay 0.0001, maximum 60 epochs and patience 15 were retained as starting settings. Weighted epoch loss now uses the sum of target-class weights as its denominator. New checkpoints persist training settings and class weights. No loss tuning based on the test result has been performed.

## Previous angle-filtered V3 baseline

The original `checkpoints/best_model_behavioral.pth` was evaluated on 132 test matrices. It does not persist a seed, unlike new lecture-only checkpoints.

```text
              precision    recall  f1-score   support

         Low       0.93      0.75      0.83        72
         Mid       0.70      0.84      0.76        25
        High       0.68      0.86      0.76        35

    accuracy                           0.80       132
   macro avg       0.77      0.82      0.78       132
weighted avg       0.82      0.80      0.80       132

Macro-F1: 78.46%
Exact accuracy: 79.55% (105/132)
Always-Low baseline macro-F1: 23.53%
```

Confusion matrix (rows are true labels; columns are predictions):

```text
             Pred Low  Pred Mid  Pred High
True Low        54         7         11
True Mid         1        21          3
True High        3         2         30
```

That older result overpredicted High: recall 0.86 versus precision 0.68. It is a single-run baseline on the earlier mixed-activity selection, not the current lecture-only score.

## Historical experiments

These results document earlier feature versions and are not directly comparable to V3 because feature definitions, tracking, training data, or experimental procedures differed.

| Version | Main idea | Reported macro-F1 | Interpretation |
|---|---|---:|---|
| I1 | Bounding-box proximity and coordinates | 84.44% ± 1.92% | High risk of spatial/session shortcuts |
| I2 | Fixed-axis pose orientation | 79.52% ± 1.32% | Pose-based but retained coordinates |
| I3 | Dynamic target orientation | 82.62% ± 1.60% | Retained coordinates and only five people |
| V2 | All-person aggregate without exported coordinates | 82.69% verified single checkpoint | Used unvalidated nose-to-eye and missing-face heuristics |
| V3, earlier selection | Eight-frame association and reviewed proxy rules | 78.46% single run | 132 test clips; angle-filtered selection |
| V3, activity-filtered lecture subset | Same stored behavioral matrices; fresh classifier | 95.48% single run | 31 test clips; preliminary fixed-setup result |

Historical affect-only and reduced-feature ablations were produced with earlier schemas. They remain useful development records but must be rerun on V3 before being presented as current ablation evidence. The lower V3 score cannot yet be attributed specifically to “shortcut removal,” because several variables changed together.

## Efficiency measurement

A 10-clip pilot of the V3 interaction extractor on Apple MPS took approximately 15.8 seconds after startup, or 1.58 seconds per 10-second clip. This was about 6.7× faster than the attempted dense all-frame tracker (~10.57 seconds per clip). This timing covers the interaction extractor only; it is not an end-to-end real-time or FPS benchmark.

## Reproduction

The 308-clip snapshot and `smoothed_seed42` run already exist locally. Do not rerun extraction or preparation for this snapshot. Evaluate the saved run with:

```bash
python run_lecture_behavioral.py --stage eval --run_name smoothed_seed42
```

For a fresh training run using the same snapshot:

```bash
python run_lecture_behavioral.py --stage train --seed 43 --run_name smoothed_seed43
python run_lecture_behavioral.py --stage eval --run_name smoothed_seed43
```

Choose the seed list and settings in advance and report all runs, not the best test score. The recorded seed-42 run used the local `LEM` environment; earlier extraction used `slowfast`. Use a compatible environment with the project dependencies installed.

For a new snapshot only, select a new experiment directory:

```bash
python run_lecture_behavioral.py --stage prepare --experiment_dir experiments/lecture_only_v2
```

Pass the same `--experiment_dir` when training/evaluating that snapshot. Preparation copies matrices using class + filename and retains original split membership. It refuses existing destinations, validates shape/finite values, and records exact membership and hashes. Training runs also refuse overwrite. Data live under `dataset/`; checkpoints, history, logs, confusion matrix and per-clip predictions live under `runs/<run_name>/`. The original matrices and results are untouched.

The original 27-test local suite passed, including a synthetic end-to-end lecture-only training/evaluation test. The published core suite additionally covers portable selection lists; local audit-UI tooling is not required for training and is not included in this release. Run `python -m unittest discover -s tests -v` to check the published suite. Software tests do not establish scientific generalization.

## Next experiments

1. Freeze the current feature/loss settings and repeat a predetermined set of training seeds; report mean, standard deviation and all individual results. This measures training variability, not independence of the split.
2. Verify continuity relationships and class coverage before designing grouped/blocked evaluation. Do not invent sessions from filename ranges or assume a fixed five-fold design is feasible.
3. Check development-audit overlap and independently validate head/posture, orientation and speaker/listener proxies without tuning on evaluation labels.
4. After stabilizing the evaluation protocol, compare interaction-only, affect-only and feature-group ablations on the same splits.
5. Additional lecture recordings remain optional future evidence. No compatible external lecture test set has been identified; expanding to group discussions solely to increase counts is not the current plan.

## Important terminology

- Engagement labels are dataset-level group labels, not direct measurements of learning.
- Orientation scores are pose-derived proxies, not eye tracking or an attention probability.
- Facial-affect outputs are model estimates, not ground-truth internal emotions.
- Raised/lowered head and very-low-head values are physical pose proxies, not proof of attention, phone use, fatigue, or sleep.
