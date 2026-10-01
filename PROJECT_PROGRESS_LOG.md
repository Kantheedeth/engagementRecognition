# Project Progress and Experiment Log

Updated 2026-10-01. This log describes the 32-interaction + 8-affect V3 experiment, not the separate 40-interaction role-aware ByteTrack implementation on the remote feature branch. The `docs/lecture-only-results-20260926` branch now includes the matching core implementation, tests and portable 308-clip selection list. Matrices, videos and checkpoints remain outside Git.

## 1. Current research objective

Develop a lightweight and explainable model for **speaker-led lecture activities** that predicts Low, Mid, or High group engagement from how listeners behave relative to one speaker.

The current scope excludes group discussion/work even when filmed from the lecture camera angle. It focuses on observable behavioral proxies—head elevation, coarse body posture, orientation toward the instruction area, motion, and group facial affect—rather than scene appearance. Inattentive listeners remain valid lecture examples.

The model output is group engagement. It does not directly measure individual learning, comprehension, or internal attention.

## 2. Dataset curation and limitations

### Current modeling data

The team reviewed activity rather than camera angle and retained **308 videos** in `lecture_only_videos/`:

| Split | Low | Mid | High | Total |
|---|---:|---:|---:|---:|
| Train | 133 | 55 | 64 | 252 |
| Validation | 10 | 3 | 12 | 25 |
| Test | 16 | 6 | 9 | 31 |
| Total | 159 | 64 | 85 | 308 |

Their existing `(8, 40)` matrices were copied into `experiments/lecture_only/dataset/`, keeping original split assignments. No preprocessing or extraction was rerun. Original CSVs, matrices and results were preserved.

Historical counts are different snapshots: the earlier matrix set contains 1,194 clips (938/124/132), while its extractor processed 1,195 preprocessed clips including the subsequently excluded round-table `view1230`. The later raw-folder inventory found 1,152 clips (Low 589, Mid 251, High 312). It must not be substituted for historical matrix counts.

### Corrected split and recording interpretation

- Earlier claims of "23 duplicate clips across splits" confused numeric basenames with full clip identities. The 1,194 rows have 1,194 unique class/filename keys, despite only 1,144 numeric IDs. The 23 cross-split numeric-ID collisions occur in different class folders and are not proof of identical videos.
- No full class/filename key crossed those splits. The lecture-only selection also had no cross-split identical matrix hashes. Similar footage and shared-source dependence are still possible.
- Shared camera angle, clothing and participants identify a shared setup, not necessarily a continuous master recording. The user can assess continuity between particular clips; parent-session identities remain unknown.
- Do not infer a source timeline or lecture sessions solely from consecutive filenames or gaps. Earlier numerical "session" taxonomies and their class-purity claims are not established recording metadata.
- The current 31-clip test measures the activity-filtered original clip split. Independence from training footage and historical manual calibration is not established.
- A grouped/blocked evaluation requires verified continuity relationships and enough class coverage. It cannot yet be called leakage-free or lecture-held-out.

## 3. Evolution of the interaction module

| Version | Design | Main limitation | Historical result |
|---|---|---|---:|
| I1 | Top-five bounding boxes and distance to podium | Direct coordinate and room-layout shortcuts | 84.44% ± 1.92% macro-F1 |
| I2 | Pose orientation against a fixed classroom axis | Fixed target and retained coordinates | 79.52% ± 1.32% |
| I3 | Pose orientation toward a detected speaker | Retained coordinates; only five people | 82.62% ± 1.60% |
| V2 | Aggregate all listeners; no absolute coordinates exported | Invalid nose-to-eye head rule and aggressive missing-face heuristic | 82.69% verified single checkpoint |
| V3, earlier selection | Eight-frame identity association and reviewed pose proxies | Angle filtering still admitted group activities | 78.46% single run, 132 test clips |
| V3, lecture-only subset | Same stored features; classifier retrained on 252 clips | Small test set and unverified source independence | 95.48% single run, 31 test clips |

Historical values are not controlled ablations of V3. Feature definitions, tracking, data composition, and sometimes training procedures changed, so score differences cannot be assigned to one cause.

## 4. Current V3 interaction extraction

### Input and tracking

- Eight uniformly sampled 640×640 RGB frames per clip; actual clip durations can vary.
- YOLOv8n-pose detects people and 17 pose keypoints.
- Hungarian matching associates detections between sampled frames using normalized box-center distance, box IoU, and upper-body pose distance.
- IDs reset for every clip and do not identify students across videos.
- Motion features use only identities matched across consecutive sampled observations.
- Unmatched/new detections are excluded from motion estimates rather than counted as stationary.

The tracker uses coordinates internally. Absolute `cx`, `cy`, width, and height are not exported as classifier features, but speaker-zone selection and displacement still depend on geometry. Direct coordinate inputs are removed; freedom from shortcuts and camera invariance remain experimental questions.

### Speaker/listener rule

- Instruction zone: normalized `x ∈ [0.0, 0.27]`.
- A person detected in that zone is the speaker.
- If no speaker is detected, `[0.135, 0.50]`, the zone center, is used as the target.
- Other people are listeners, while an associated speaker remains excluded if temporarily outside the zone.

### Reviewed head and posture proxies

Three reviewers annotated assigned parts of the first audit: Kan 01–15, Pan 19–20, and Bright 22–36. Frames 16–18 were excluded because they came from round-table `view1230`; frame 21 was skipped as an annotation only.

The old nose-to-eye vertical-distance rule tended to predict head-up excessively. Reviews informed the switch to observed nose-to-shoulder elevation. Previously quoted 96-record head-state scores are withheld pending verification of the audit-frame index mapping and which reviewers' rows were included. Even correctly aggregated development scores would not be independent validation, because the reviews informed threshold selection.

Current settings:

- Lowered-head proxy: normalized nose-to-shoulder elevation `< 0.130`
- Raised-head proxy: elevation `≥ 0.155`
- Upright-posture proxy: elevation `≥ 0.120`
- Very-low-head proxy: elevation `≤ 0.060`
- Orientation-consensus threshold: score `≥ 0.66`

Missing pose evidence is recorded as unknown and excluded from head/posture denominators. A very-low-head observation is not labelled as sleeping.

The reviewers annotated mostly different frames; this work should be called a multi-reviewer calibration exercise, not an inter-rater agreement study.

## 5. Current model

### Feature sequence

- Interaction branch: 32 dimensions per frame
- Affect branch: 8 dimensions per frame
- Combined input: `(8, 40)`
- Interaction projection: `32 → 64`
- Affect projection: `8 → 64`
- Fused embedding: 128 dimensions
- Temporal self-attention: four heads
- Dropout: 0.20
- Lecture-only run: seed 42 recorded in run configuration; new checkpoints persist training settings and class weights. The older full-selection checkpoint did not persist its seed.

The active schemas are:

- Interaction: `yolov8_pose_sampledtrack_collective32_v3`
- Behavioral matrix: `interaction32_sampledtrack_affect8_v3`

### Current lecture-only result, verified 2026-09-26

Run directory: `experiments/lecture_only/runs/smoothed_seed42/`. Evidence: `reports/metrics.json`, `reports/train.log` and `run_config.json`.

| Actual class | Predicted Low | Predicted Mid | Predicted High |
|---|---:|---:|---:|
| Low | 16 | 0 | 0 |
| Mid | 0 | 6 | 0 |
| High | 0 | 1 | 8 |

- Macro-F1: **95.48%**
- Accuracy: **96.77% (30/31)**
- Always-Low baseline macro-F1: **22.70%**
- Only error: `high/view2480.mp4` predicted Mid.
- Best validation-loss checkpoint: epoch 40, validation accuracy 88% (22/25); training stopped at epoch 55.

The score is promising within this selected setup, but not yet a reliable estimate for unseen recordings. One additional mistake would reduce macro-F1 to approximately 90.74–92.66%, depending on the class pair. This is a sensitivity illustration, not a confidence interval. There are only three Mid validation examples and six Mid test examples.

Filtering changes the task population and the test set. The rise from 78.46% cannot be presented as a controlled model improvement or proof of shortcut removal. Better task alignment, easier retained examples and related source footage remain possible explanations.

### Conservative temporal evaluation, verified 2026-09-27

The 308 clips were also evaluated using 18 conservative temporal groups and a
size-matched randomized control. Both protocols use identical per-fold class
counts, matrices, hyperparameters, and model seeds.

| Protocol | Mean Macro-F1 (seeds 42--46) | Sample SD |
|---|---:|---:|
| Conservative temporal groups | **56.34%** | 1.77 |
| Size-matched randomized clips | **81.43%** | 1.66 |
| Paired difference | **25.10 points** | 2.51 |

The positive gap appears for all five seeds. It shows that random clip mixing
benefits from temporal correlation; it does not estimate generalization to a
new classroom or recording.

On the temporal folds, interaction-only achieved **54.02% ± 8.33%**,
affect-only **57.01% ± 4.37%**, and interaction+affect **56.34% ± 1.77%**.
Fold 1 remains weak for all branch modes. Feature diagnostics over all 308
matrices and sampled tracking diagnostics over 36 clips show no broad numerical
collapse in the large High group `H04`. H04 is instead markedly stiller and
lower-motion than the other High groups, with moderately lower face/affect
reliability. This initially supported behavioral distribution shift, possible
label ambiguity, and scarce independent High groups rather than a group-specific
extraction collapse. V3 remains frozen as the evaluated baseline rather than
being tuned directly against these test-fold observations.

A subsequent preliminary manual audit completed three clips (H04=2, H02=1).
Pose placement was marked good in all three, but orientation was mixed or
inconsistent in all three and every clip reported an ID switch. Two clips had
major tracking-continuity issues. This sample is too small and unbalanced for a
group comparison, but it demonstrates that automatic feature availability does
not establish semantic correctness. The current eye-to-nose image vector is not
a calibrated head-pose or gaze estimate. V3 orientation explanations are
therefore unvalidated.

The pre-specified orientation-family ablation was then run on both frozen
protocols for seeds 42--46 without rebuilding matrices. Removing interaction
columns 2--7 and 30 reduced temporal Macro-F1 from **56.34% ± 1.77%** to
**51.85% ± 1.13%** and matched-random Macro-F1 from **81.43% ± 1.66%** to
**80.15% ± 1.37%**. The model therefore relies on these fields, especially for
the difficult temporal High class, but the score contribution does not validate
their behavioral meaning.

Complete evidence and limitations are in
[`docs/BEHAVIORAL_INVESTIGATION_RESULTS.md`](docs/BEHAVIORAL_INVESTIGATION_RESULTS.md).

### Loss and configuration

- Cross-entropy: square-root inverse-frequency weights calculated from training counts 133/55/64, normalized to mean one.
- Weights: Low 0.7506, Mid 1.1673, High 1.0821.
- Label smoothing: 0.0; the `smoothed` run name refers to class weighting, not target smoothing.
- Batch size 32; AdamW learning rate 0.001; weight decay 0.0001; maximum 60 epochs; patience 15.
- Architecture retained: branch dimension 64, four heads, dropout 0.20.
- Weighted epoch-loss aggregation corrected to use target-class weight totals; per-batch CE optimization unchanged.
- No hyperparameter or loss tuning based on the new test score. Seed-42 training was recorded in the local `LEM` environment.

### Previous angle-filtered V3 result

```text
              precision    recall  f1-score   support

         Low       0.93      0.75      0.83        72
         Mid       0.70      0.84      0.76        25
        High       0.68      0.86      0.76        35

Macro-F1: 78.46%
Accuracy: 79.55% (105/132)
Always-Low baseline macro-F1: 23.53%
```

```text
             Pred Low  Pred Mid  Pred High
True Low        54         7         11
True Mid         1        21          3
True High        3         2         30
```

Interpretation:

- The model performs substantially above the always-Low baseline.
- Recall is reasonably balanced across the three classes.
- High is overpredicted: recall is 0.86 but precision is 0.68.
- Eleven Low clips are predicted as High, so severe Low/High errors remain.
- This is a single-run result under an original clip split with unverified source independence; it is no longer the active lecture-only benchmark.

## 6. Efficiency result

Dense all-frame ByteTrack was stopped after an observed rate of approximately 10.57 seconds per clip. A 10-clip V3 pilot using only the eight saved frames required approximately 15.8 seconds after startup, or 1.58 seconds per clip on Apple MPS—about 6.7× faster for interaction extraction.

This is an interaction-extraction pilot, not a full end-to-end latency or FPS benchmark. Affect extraction, preprocessing, training, and deployment latency were not included.

## 7. Work completed

- Defined the speaker–listener scope, initially removed `view1230`, then reviewed activity and retained 308 lecture clips.
- Corrected false exact-duplicate/session conclusions based on numeric filenames; shared-source dependence remains an evaluation risk to investigate.
- Removed scene features from the current behavioral classifier.
- Replaced top-five listener selection with all-listener aggregate statistics.
- Added eight-frame clip-local identity association.
- Changed motion computation from nearest boxes to matched identities.
- Created a local listener audit interface.
- Collected reviews from Kan, Pan, and Bright.
- Replaced the invalid head rule and removed missing-face-as-sleep semantics.
- Added schema, extraction-manifest, and checkpoint compatibility checks.
- Re-extracted all interaction arrays, rebuilt matrices, retrained, and evaluated V3.
- Created an isolated lecture-only snapshot by reusing existing matrices, with per-file hashes, strict membership checks and overwrite protection.
- Added separate run outputs, training configuration/history, machine-readable metrics and per-clip predictions.
- Verified all 308 copied matrices and passed the full 27-test local suite, including a synthetic end-to-end run. Software tests are not evidence of scientific validity.
- Completed the first lecture-only training/evaluation: 95.48% macro-F1 on 31 clips.
- Prepared the core implementation for collaborators: portable selection-list input, dependency instructions, sampled tracker/calibration/schema, subset safeguards and isolated training/evaluation. Legacy visualization changes asserting sleep from hidden faces are excluded from the release.
- Completed seeds 42--46 on the frozen temporal and size-matched randomized protocols.
- Audited temporal-group feature quality over all 308 matrices and sampled raw tracking records from 36 clips.
- Added and ran interaction-only and affect-only ablations over the same folds and seeds.
- Retained V3 as the frozen evaluated baseline instead of tuning directly against the held-out H04 group.
- Generated a blinded H04-versus-other-High visual audit from the exact saved frames and interaction tracking records. Thirty-five clips are locally renderable; 50 cloud-placeholder clips are explicitly recorded as unavailable pending download.
- Recorded the first three visual reviews: good pose placement but repeated orientation-arrow mismatch and ID-switch reports. This is preliminary, not an error-rate estimate.
- Implemented matrix-preserving interaction-column ablation with checkpointed feature indices and backward-compatible evaluation.
- Ran the orientation-family ablation for seeds 42--46 on both frozen protocols; all 40 local tests pass.

## 8. Immediate next experiments

1. **Freeze V3 and the evaluation:** keep the current 308 matrices, temporal
   groups, matched-random membership, hyperparameters, and seeds unchanged.
2. **Complete feature-family contribution tests:** use the existing
   matrix-preserving ablation support for head/posture, motion/tracking, and
   speaker/reliability families. Orientation columns 2--7 and 30 are already
   complete. Define every column set before running it.
3. **Perform independent semantic validation:** create a new manual-audit sample
   across Low, Mid, High, and multiple temporal groups. Do not reuse the earlier
   development annotations as independent validation. Freeze the sample,
   rubric, V3 thresholds, and reviewer overlap before annotation.
4. **Keep importance and correctness separate:** a Macro-F1 decline establishes
   model reliance only. Manual agreement with visible behavior is required
   before a feature is presented as an explanation.
5. **Create V4 only from diagnosed evidence:** replace useful but invalid
   features, beginning with a separately validated head-pose method. Do not add
   action recognition without individual action labels and a new evaluation
   plan.
6. **Report both frozen protocols:** present 56.34% ± 1.77% as the conservative
   within-recording temporal baseline and 81.43% ± 1.66% as the
   leakage-sensitive randomized control. The 51.85% and 80.15% results are only
   the orientation-removal diagnostic.
7. **Finish the efficiency claim later:** after V4 is fixed, measure interaction,
   affect, complete inference time, model parameters, and peak memory separately.

## 9. Questions for professor discussion

1. Is the five-seed conservative temporal protocol a defensible main result for this single-recording dataset?
2. Should H04's distinct but apparently valid High behavior be presented as label ambiguity, intra-class variation, or both?
3. Is a narrowly scoped speaker–listener study sufficient if no compatible external lecture dataset is available?
4. What independent manual validation is expected for head/posture, orientation and speaker/listener assignment?
5. If stronger generalization evidence is required, is collecting a small additional lecture recording feasible?

## 10. Reproduction commands

```bash
# Evaluate the existing lecture-only baseline.
python run_lecture_behavioral.py --stage eval --run_name smoothed_seed42

# A fresh, separately named seed run on the prepared snapshot.
python run_lecture_behavioral.py --stage train --seed 43 --run_name smoothed_seed43
python run_lecture_behavioral.py --stage eval --run_name smoothed_seed43
```

Preparation and extraction do not need rerunning for the existing snapshot. Use a new `--experiment_dir` if intentionally creating another snapshot; existing data and training run directories are protected from overwrite. New evaluation verifies checkpoint/subset provenance. Source matrices and raw videos are not uploaded with this documentation.
