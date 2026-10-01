# Project Handoff

Last updated: 2026-10-01

This is the shortest path for a collaborator to resume the project without
reconstructing decisions from chat history. Read this file first, then use
`PROJECT_PROGRESS_LOG.md` for the development chronology and
`docs/BEHAVIORAL_INVESTIGATION_RESULTS.md` for the complete controlled results.

## Current checkpoint

- Scope: Low/Mid/High **group engagement** during speaker-led classroom
  activity; group work, round-table discussion, and peer presentations are out
  of scope.
- Data: 308 manually selected OUC-CGE lecture clips (159 Low, 64 Mid, 85 High).
- Input: eight uniformly sampled 640x640 frames per clip.
- V3 representation: 32 interaction features plus 8 group-affect features per
  frame. Scene embeddings are excluded.
- Interaction identity association: Hungarian box/pose matching across the
  eight sampled frames. ByteTrack belongs to the separate affect branch.
- Main current result: 56.34% ± 1.77% Macro-F1 over seeds 42--46 under the
  conservative temporal-group protocol.
- Control: 81.43% ± 1.66% under size-matched randomized clip splits. The paired
  difference is 25.10 ± 2.51 points.
- Completed diagnostic: removing interaction columns 2--7 and 30 reduces the
  temporal result to 51.85% ± 1.13%. This proves model reliance on the
  orientation family, not semantic correctness.
- Manual evidence: only three High clips are completed in the latest visual
  audit. They show usable pose placement but repeated orientation mismatch and
  track-ID changes. This is not an error-rate estimate.

## What is frozen

Do not overwrite or silently tune:

- the 308 V3 matrices;
- the 18-group temporal protocol;
- the size-matched randomized protocol and split seed `20260926`;
- model seeds 42--46 and the recorded V3 training settings; or
- the completed baseline, branch-ablation, and orientation-ablation results.

V3 is the historical evaluated baseline. Any changed extraction definition is
V4 and must be evaluated separately on a pre-declared protocol.

## Data that Git does not contain

Raw videos, preprocessed frames, feature matrices, checkpoints, generated fold
datasets, and run directories are intentionally ignored. To rerun the frozen
experiments, a collaborator needs an authorized OUC-CGE copy plus either:

1. the existing V3 `(8, 40)` matrices and lecture-only dataset snapshot; or
2. the dependencies and source data required to regenerate V3.

The portable 308-clip selection is tracked under
`experiments/lecture_only/lecture_selection.csv`. Protocol JSON files and
human-readable assignments are tracked under the three lecture CV experiment
directories. `python run_lecture_episode_cv.py --stage prepare ...` reconstructs
generated fold directories from the local lecture-only dataset snapshot.

## Verify the code

Use an environment containing the packages in `requirements-behavioral.txt`
and `requirements-affect.txt`. The local verified command is:

```bash
python -m unittest discover -s tests -v
```

As of this handoff, all 40 tests pass. They cover the lecture subset,
person association, grouped protocols, randomized control, branch modes,
matrix-preserving feature omission, and audit tools. Tests establish software
behavior, not scientific validity.

## Resume the current experiments

Prepare the conservative temporal folds from a local lecture-only snapshot:

```bash
python run_lecture_episode_cv.py \
  --stage prepare \
  --experiment_dir experiments/lecture_temporal_group_cv \
  --episode_protocol experiments/lecture_temporal_group_cv/temporal_group_protocol.json
```

Prepare the fixed size-matched randomized control:

```bash
python run_lecture_episode_cv.py \
  --stage prepare \
  --experiment_dir experiments/lecture_matched_random_cv \
  --episode_protocol experiments/lecture_matched_random_cv/matched_random_protocol.json
```

Training creates a new run directory and refuses to overwrite an existing one.
Use `--branch_mode interaction`, `--branch_mode affect`, or
`--drop_interaction_indices` for controlled ablations. Never select the best
seed after observing test results; report every pre-specified seed.

## Next scientific work

The next checkpoint is systematic validation of the frozen behavioral feature
families, not immediate training or extractor redesign.

1. Define the remaining feature-family column sets before running them:
   head/posture, motion/tracking, and speaker/reliability.
2. Run each ablation under both frozen protocols and seeds 42--46. A score
   change measures model contribution only.
3. Freeze a new independent manual-audit sample across Low, Mid, High, and
   multiple temporal groups. Keep it separate from the annotations used to
   choose V3 thresholds.
4. Annotate speaker correctness, listener coverage, head state, posture,
   orientation consistency, tracking continuity, and face availability. Use an
   overlapping subset for reviewer agreement.
5. Compare model contribution with semantic correctness. Keep valid useful
   features, remove irrelevant ones, and replace useful but invalid ones.
6. Create V4 only after that decision. The first expected replacement is a
   separately validated head-pose method; the current eye-to-nose vector must
   not be described as gaze or attention.
7. After V4 is fixed, report end-to-end timing, parameters, and peak memory.

## Reporting language

- `95.48%` is the preliminary filtered original-split result on 31 test clips.
- `81.43% ± 1.66%` is the leakage-sensitive randomized control.
- `56.34% ± 1.77%` is the main current estimate for unseen temporal periods
  within the same recording environment, not a new classroom or new recording.
- Ablation identifies what the classifier uses. Manual comparison identifies
  whether the extracted value means what its name claims.
- Do not describe orientation as measured attention, group affect as internal
  emotional ground truth, or head posture as proof of sleep or disengagement.
