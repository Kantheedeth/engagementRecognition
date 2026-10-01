# Conservative temporal-group-held-out evaluation

This experiment is the stricter successor to the provisional 23-candidate-
episode evaluation. It keeps every selected ten-second clip as an individual
sample but assigns temporally adjacent ranges to the same split group when they
are separated by only one excluded source clip.

The excluded clips (`view697`, `view881`, `view1198`, `view1200`, and
`view2521`) remain excluded. They contribute no matrices or predictions. Their
surrounding selected ranges share a group ID only to prevent temporal leakage.

The resulting 18 temporal groups are evaluated using three outer folds. Every
selected clip receives exactly one out-of-fold test prediction, and validation
also holds out complete temporal groups. All groups still originate from one
fixed recording; this is not cross-recording or cross-classroom evidence.

```bash
python run_lecture_episode_cv.py \
  --stage prepare \
  --experiment_dir experiments/lecture_temporal_group_cv \
  --episode_protocol experiments/lecture_temporal_group_cv/temporal_group_protocol.json

python run_lecture_episode_cv.py \
  --stage all \
  --experiment_dir experiments/lecture_temporal_group_cv \
  --episode_protocol experiments/lecture_temporal_group_cv/temporal_group_protocol.json \
  --run_name smoothed_seed42
```

## Seed-42 result

| Result | Macro-F1 | Accuracy | Test clips |
| --- | ---: | ---: | ---: |
| Fold 1 | 33.04% | 44.62% | 130 |
| Fold 2 | 77.18% | 79.57% | 93 |
| Fold 3 | 69.71% | 81.18% | 85 |
| **Pooled out-of-fold** | **55.31%** | **65.26%** | **308** |

Pooled class F1 was 80.91% Low, 56.95% Mid and 28.07% High. The large
62-clip High temporal block is held out in Fold 1; none of its clips were
recognized as High in this run.

## Seeds 42--46 and branch ablations

| Branch mode | Mean pooled Macro-F1 | Sample SD | Range |
| --- | ---: | ---: | ---: |
| Interaction only | 54.02% | 8.33 | 44.27--65.95% |
| Affect only | 57.01% | 4.37 | 51.51--63.73% |
| Interaction + affect | **56.34%** | **1.77** | 53.94--58.58% |
| Interaction + affect, orientation family removed | **51.85%** | **1.13** | 49.92--52.60% |

Fold 1 remains weak in every branch mode. Automatic feature/tracking diagnostics
found no broad numerical collapse in its large High group (`H04`); H04 is
instead substantially stiller and lower-motion than the other High groups.
A later three-clip visual review found good pose placement but repeated
orientation-arrow mismatch and ID-switch reports. This is too small for a group
error rate, but it shows that automatic availability/reliability does not prove
semantic correctness. The temporal result still does not show cross-recording
generalization.

The orientation-family row omits interaction columns 2--7 and 30 inside the
model while retaining the original matrices and all other settings. Its paired
change versus full V3 is **-4.49 ± 1.86 points** over seeds 42--46, with a
decline at every seed. Mean pooled High-class F1 falls from 34.25% to 25.05%.
This establishes model reliance on those fields, not semantic validity of the
orientation arrow.

See
[`docs/BEHAVIORAL_INVESTIGATION_RESULTS.md`](../../docs/BEHAVIORAL_INVESTIGATION_RESULTS.md)
for seed-level, per-fold, class-level, extraction-audit, and decision details.
