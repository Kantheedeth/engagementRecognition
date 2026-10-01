# Behavioral Model Investigation Results

Last updated: 2026-10-01

This document records the predetermined investigation performed after the
controlled split study. The frozen inputs are the 308 lecture-only V3 matrices,
the 18-group conservative temporal protocol, and the size-matched randomized
protocol. No feature thresholds or split assignments are changed during this
investigation.

## Step 1 — Model-seed stability

Model seeds 42--46 were run on both frozen protocols with identical model and
training settings. The randomized split seed remains fixed at `20260926`.

| Model seed | Temporal-group Macro-F1 | Matched-random Macro-F1 | Paired gap |
| ---: | ---: | ---: | ---: |
| 42 | 55.31% | 83.12% | 27.81 points |
| 43 | 58.58% | 81.59% | 23.00 points |
| 44 | 56.78% | 78.84% | 22.07 points |
| 45 | 53.94% | 81.04% | 27.10 points |
| 46 | 57.06% | 82.56% | 25.50 points |
| **Mean ± sample SD** | **56.34% ± 1.77** | **81.43% ± 1.66** | **25.10 ± 2.51 points** |

The separation is stable across all five model seeds. Training randomness does
not explain the controlled performance gap.

Temporal per-fold Macro-F1:

| Seed | Fold 1 | Fold 2 | Fold 3 |
| ---: | ---: | ---: | ---: |
| 42 | 33.04% | 77.18% | 69.71% |
| 43 | 45.03% | 58.43% | 68.51% |
| 44 | 44.89% | 63.10% | 70.19% |
| 45 | 19.05% | 71.72% | 70.39% |
| 46 | 34.00% | 61.15% | 68.00% |

Fold 1 remains the principal unstable and low-performing fold. The pooled High
class F1 across temporal runs ranges from 26.23% to 43.84%, compared with
73.20% to 78.95% in the matched-random runs.

**Step-1 conclusion:** the temporal difficulty and the random-versus-temporal
gap are reproducible over the predetermined model seeds.

## Step 2 — Extraction-quality audit

The audit was performed without changing extraction thresholds or regenerating
features. It has two complementary parts:

1. all 308 stored `(8, 40)` matrices were summarized by temporal group; and
2. raw per-frame tracking records were summarized for a deterministic sample of
   the first and last retained clip in every group (36/36 files completed).

The large held-out High group in Fold 1 is `H04` (62 clips). It was compared
with the other three High groups combined (23 clips).

### H04 extraction-quality indicators

| Indicator | H04 | Other High | Standardized difference (d) |
| --- | ---: | ---: | ---: |
| Speaker in instruction zone | 0.972 | 0.951 | +0.145 |
| Normalized listener count | 0.531 | 0.534 | -0.055 |
| Mean face visibility | 0.744 | 0.771 | -0.433 |
| Hidden-face ratio | 0.024 | 0.022 | +0.068 |
| Posture observed | 0.979 | 0.982 | -0.094 |
| Orientation alignment/reliability | 0.956 | 0.959 | -0.075 |
| Mean orientation to target | 0.717 | 0.724 | -0.115 |
| Orientation consensus | 0.730 | 0.765 | -0.188 |
| Affect reliability | 0.461 | 0.520 | -0.378 |

Speaker presence, listener count, posture availability, and orientation
reliability do not show a systematic collapse in H04. Face visibility and
affect reliability are moderately lower, but not absent.

The 36-clip raw-record sample also does not show failed H04 tracking:

| High group | Speaker target ratio | Speaker track coverage | Mean listeners/frame | Mean listener-track coverage | Long-track ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| H01 | 1.000 | 1.000 | 8.125 | 0.856 | 0.894 |
| H02 | 0.625 | 0.813 | 7.500 | 0.833 | 0.778 |
| H03 | 1.000 | 1.000 | 8.875 | 0.848 | 0.768 |
| **H04** | **1.000** | **1.000** | **8.250** | **0.870** | **0.844** |

### H04 behavioral distribution

H04 differs most strongly in motion-related features:

| Feature | H04 | Other High | d |
| --- | ---: | ---: | ---: |
| Stillness ratio | 0.616 | 0.472 | +1.098 |
| Mean center displacement (scaled) | 0.117 | 0.181 | -0.857 |
| Displacement variability (scaled) | 0.131 | 0.201 | -0.792 |
| High-motion ratio | 0.055 | 0.100 | -0.755 |
| Speaker displacement (scaled) | 0.064 | 0.091 | -0.480 |
| Raised-head ratio | 0.476 | 0.562 | -0.379 |

The affect distribution also differs, especially estimated surprise (0.023
versus 0.036, `d=-0.864`). The large standardized effect for estimated disgust
(`d=-1.005`) concerns very small absolute probabilities (0.0037 versus 0.0060)
and should not be given a substantive emotional interpretation.

**Step-2 conclusion:** the stored features and sampled tracking records contain
no evidence of a broad *numerical* H04 extraction collapse. H04 is instead a
much stiller, lower-motion High period with somewhat weaker face/affect
reliability. These automatic statistics only measure availability and internal
consistency. They cannot establish that an orientation vector agrees with
visible head direction or that a track ID follows the correct person. The
preliminary visual audit in Step 5 therefore supersedes any stronger semantic
interpretation of this step.

Audit artifacts:

- `experiments/lecture_temporal_group_cv/diagnostics/temporal_group_feature_audit.json`
- `experiments/lecture_temporal_group_cv/diagnostics/clip_feature_audit.csv`
- `experiments/lecture_temporal_group_cv/diagnostics/h04_vs_other_high.csv`
- `experiments/lecture_temporal_group_cv/diagnostics/tracking_sample_audit.json`

## Step 3 — Branch ablations

Interaction-only, affect-only, and interaction+affect models were trained on
the same frozen temporal folds for seeds 42--46. All other training settings
were held constant.

| Seed | Interaction only | Affect only | Interaction + affect |
| ---: | ---: | ---: | ---: |
| 42 | 57.41% | 51.51% | 55.31% |
| 43 | 65.95% | 55.96% | 58.58% |
| 44 | 53.77% | 56.91% | 56.78% |
| 45 | 48.66% | 63.73% | 53.94% |
| 46 | 44.27% | 56.93% | 57.06% |
| **Mean ± sample SD** | **54.02% ± 8.33** | **57.01% ± 4.37** | **56.34% ± 1.77** |
| **Range** | 44.27--65.95% | 51.51--63.73% | 53.94--58.58% |

The interaction+affect mean is 0.67 points below affect-only, which is too small
and inconsistent across five seeds to justify removing interaction. Its much
lower seed variability is useful: neither single branch produces a stable
improvement.

Mean per-fold Macro-F1 across the five seeds:

| Branch mode | Fold 1 | Fold 2 | Fold 3 | Pooled High F1 |
| --- | ---: | ---: | ---: | ---: |
| Interaction only | 34.68% | 65.64% | 56.88% | 42.31% |
| Affect only | 37.46% | 79.39% | 65.02% | 31.93% |
| Interaction + affect | 35.20% | 66.32% | 69.36% | 34.25% |

Fold 1 remains weak for every branch mode. Therefore, its failure cannot be
assigned solely to interaction extraction, affect extraction, or the current
fusion. Interaction contributes more to pooled High-class F1 on average, while
affect performs especially strongly in Fold 2. Their value is group-dependent.

For the single-branch runs the temporal embedding is 64-D; the combined model
concatenates both 64-D projections into 128-D. These are true branch-removal
ablations, but model capacity consequently differs. The results identify branch
dependence; they are not a perfectly parameter-matched architecture comparison.

## Step 4 — Decision after automatic diagnostics and branch ablations

At this stage, the evidence did **not** justify changing extraction thresholds
or declaring a V4 extractor merely to improve H04:

- the randomized-versus-temporal gap persists across every seed;
- H04 speaker selection, listener count, pose availability, orientation
  reliability, and sampled tracking continuity are comparable with other High
  groups;
- H04 has a distinct behavioral distribution, especially much greater
  stillness and lower motion;
- Fold 1 remains difficult for interaction-only, affect-only, and combined
  models; and
- neither single branch gives a stable improvement over the combined model.

The evidence supported **behavioral distribution shift across temporal groups,
possible label ambiguity, and only four independent High groups**. It did not,
however, visually validate every proxy. H04 is labelled High even though several
observable proxies differ from the other High periods. The remaining three
small High groups do not provide enough evidence for a stable conclusion about
the full range of behavior that the High label can contain.

Accordingly:

1. Keep V3 feature extraction frozen while proxy validity is audited. Do not
   create V4 merely to improve H04 after observing its test labels.
2. Retain interaction + affect as the main model because it has the smallest
   seed variance and neither branch is consistently sufficient alone.
3. Report **56.34% ± 1.77** as the five-seed conservative temporal result and
   **81.43% ± 1.66** as the matched randomized control. Report the paired gap
   as **25.10 ± 2.51 points**.
4. Present the branch results as evidence that the Fold 1 problem is not isolated
   to one modality.
5. If additional work is available, perform an independent visual review of a
   predetermined H04 sample against other High groups. Use it to document label
   ambiguity and proxy validity, not to tune thresholds on the held-out fold.

A blinded visual-audit package was then generated at
`audit_outputs/high_group_visual_audit_20260927/index.html`. It uses the exact
eight saved frames and YOLO-pose/Hungarian records, hides temporal-group identity
during normal review, and exports per-clip annotations. Thirty-five clips are
currently included (H01=4, H02=4, H03=3, H04=24). The other 50 selected High
clips have at least one macOS cloud-placeholder source file; their exact keys and
reasons are recorded in the package's `audit.json`. After the placeholders are
downloaded, it can be regenerated into a new directory with `--require-all` to
enforce the full 85-clip audit.

## Step 5 — Preliminary manual visual audit

Submitted file: `high_group_annotations.csv`, SHA-256
`ee4b1a721556eca216356a258accf119eb6413e3f16bcb7b991146ef1463d291`.

The file contains 35 audit rows but only three completed reviews: two H04 clips
and one H02 clip. It is therefore not a balanced H04-versus-other-High
comparison and must not be reported as an accuracy estimate.

| Reviewed field | Three-clip observation |
| --- | --- |
| Lecture scope | 3/3 in scope |
| Pose placement | 3/3 marked good |
| Speaker role | 2 correct, 1 incorrect |
| Listener coverage | 1 correct, 2 mostly correct |
| Track continuity | 2 major issues, 1 minor issue |
| Head/posture validity | 2 mixed, 1 inconsistent |
| Orientation validity | 2 inconsistent, 1 mixed |
| High-label fit | 3/3 ambiguous |

Every reviewed clip marked `bad_orientation_arrow`, `missed_person`, and
`id_switch`. The written notes consistently describe students visibly turning
toward the speaker/front while the purple arrow continues to point downward.

This exposes a limitation that automatic availability metrics could not detect.
The current face-based direction is constructed from the image-plane vector
from the eye midpoint to the nose. For a normally visible frontal face this
vector naturally points mostly downward; it is not a calibrated 3-D head-pose
or gaze estimate. Good YOLO keypoint placement therefore does not validate the
derived orientation score.

### What an orientation ablation can and cannot do

An ablation will **not repair** the direction calculation. It will answer
whether the classifier depends on the questionable orientation family. The
pre-specified interaction columns for removal are:

- `head_turned_ratio` (2);
- orientation mean, consensus, standard deviation, minimum and maximum (3--7);
- `orientation_alignment` (30).

The existing matrices can be sliced during training, so feature extraction does
not need to be rerun. The ablation must use the same frozen temporal and
matched-random protocols and report all seeds 42--46:

- similar or improved temporal performance means these fields can be omitted
  safely from the next model;
- a substantial decline means the classifier relied on an unreliable signal,
  which is itself an important finding rather than a reason to retain it; and
- tracking/motion features require a separate sensitivity test because all
  three reviewed clips reported ID switches.

The defensible position before the ablation was: V3 is the completed evaluated
baseline and its orientation interpretation is not visually validated. Step 6
below now measures the model's dependence on this feature family. Full action
recognition is not justified because the dataset contains no individual action
labels and it would introduce a new, more computationally expensive task.

## Step 6 — Orientation-family ablation

The pre-specified ablation was completed on both frozen protocols for model
seeds 42--46. Interaction columns 2--7 and 30 were omitted inside the model:
`head_turned_ratio`, orientation mean/consensus/standard deviation/minimum/
maximum, and `orientation_alignment`. The original
`(8, 40)` matrices, extraction settings, split membership, loss, and all other
training settings were unchanged.

| Seed | Temporal baseline | Temporal without orientation | Matched-random baseline | Matched-random without orientation |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 55.31% | 49.92% | 83.12% | 78.37% |
| 43 | 58.58% | 52.57% | 81.59% | 81.08% |
| 44 | 56.78% | 52.42% | 78.84% | 79.08% |
| 45 | 53.94% | 52.60% | 81.04% | 80.62% |
| 46 | 57.06% | 51.71% | 82.56% | 81.62% |
| **Mean ± sample SD** | **56.34% ± 1.77** | **51.85% ± 1.13** | **81.43% ± 1.66** | **80.15% ± 1.37** |

Relative to the full V3 model, orientation removal changed pooled Macro-F1 by
**-4.49 ± 1.86 points** on the temporal protocol and **-1.28 ± 1.99 points**
on the matched-random protocol. The temporal decline occurred for all five
seeds; the random-control change was small and mixed around zero for one seed.
High-class F1 declined from a five-seed mean of **34.25% to 25.05%** under
temporal grouping and from **76.08% to 73.86%** in the random control.

The random-minus-temporal gap therefore increased from **25.10 ± 2.51** to
**28.31 ± 1.16 points**. Removing these fields does not solve temporal
generalization. It shows that the classifier used them, particularly to
recognize the difficult held-out High behavior. Because the visual audit and
the feature geometry do not validate the arrow as head pose or gaze, this
performance contribution is **not evidence that the orientation measurement is
semantically correct**. It is evidence of reliance on a questionable proxy.

**Step-6 decision:** preserve V3 and its results as the historical full-feature
baseline. Do not describe its orientation fields as measured attention or gaze.
The orientation-removed runs are a diagnostic ablation, not a replacement
model selected for higher score. A later V4 should replace, rather than merely
restore, these columns with an independently validated head-pose method and
should be evaluated on the same frozen protocols. Tracking/motion sensitivity
remains a separate experiment.

Run artifacts are stored under:

- `experiments/lecture_temporal_group_cv/runs/no_orientation_seed42` through
  `no_orientation_seed46`; and
- `experiments/lecture_matched_random_cv/runs/no_orientation_seed42` through
  `no_orientation_seed46`.

The main completed scientific result remains the reproducible demonstration
that randomly mixed clips overestimate transfer to unseen temporal periods in
this single-recording dataset.
