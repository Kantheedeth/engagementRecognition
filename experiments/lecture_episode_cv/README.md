# Candidate-episode-held-out lecture evaluation

This experiment reuses the frozen 308 lecture-only behavioral matrices. It does
not decode videos or rerun interaction/affect extraction. The original 95.48%
clip-level baseline remains under `experiments/lecture_only/`.

The 23 candidate episodes are maximal contiguous selected `view####` ranges.
Whole episodes—not individual clips—are assigned to train, validation, or test.
Across the three outer folds, every clip receives exactly one out-of-fold test
prediction. Validation also holds out complete episodes.

This is a stricter **within-recording temporal generalization** test. All clips
still come from one fixed classroom recording, so it does not demonstrate
generalization to new cameras, rooms, participants, or recording sessions.
Tiny gaps between candidate ranges still require manual boundary review.

```bash
# Creates three isolated fold datasets from the existing frozen snapshot.
python run_lecture_episode_cv.py --stage prepare

# Trains and evaluates every fold using seed 42.
python run_lecture_episode_cv.py --stage all --run_name smoothed_seed42

# Or run the expensive stages separately.
python run_lecture_episode_cv.py --stage train --run_name smoothed_seed42
python run_lecture_episode_cv.py --stage eval --run_name smoothed_seed42
```

Generated data and runs are ignored by Git. The tracked files are:

- `episode_protocol.json`: episode ranges, outer test folds and validation episodes.
- `episode_assignments.csv`: generated human-readable split membership.
- `folds/fold_N/`: generated matrices and manifests.
- `runs/<name>/reports/pooled_metrics.json`: pooled out-of-fold result.

## Seed-42 result

Using the frozen V3 behavioral features and the same classifier configuration as
the preliminary lecture-only baseline:

| Result | Macro-F1 | Accuracy | Test clips |
| --- | ---: | ---: | ---: |
| Fold 1 | 57.34% | 58.54% | 123 |
| Fold 2 | 70.56% | 73.68% | 95 |
| Fold 3 | 55.95% | 78.89% | 90 |
| **Pooled out-of-fold** | **64.87%** | **69.16%** | **308** |

Pooled class F1 was 80.63% Low, 58.90% Mid and 55.07% High. Fold 3 did
not correctly recognize any clip in its held-out High episode. This is a single
seed under provisional episode boundaries, but it is substantially more
informative than the 95.48% clip-level result because temporally related clips
within a candidate episode cannot cross splits.

The main score is pooled out-of-fold macro-F1 over all 308 clips. Also report
every fold's macro-F1 and confusion matrix because the 51-clip High episode makes
fold sizes necessarily unequal.
