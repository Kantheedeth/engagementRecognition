# Size-matched randomized-clip control

This is a diagnostic control for the conservative 18-temporal-group experiment.
It uses the same 308 frozen behavioral matrices and exactly matches every fold's
per-class train, validation, and test counts. The only intended split difference
is that individual clips are randomized, allowing temporally adjacent clips to
cross splits.

The split seed (`20260926`) is separate from model-training seeds 42--46 and is
frozen in `matched_random_protocol.json`. This control is deliberately
leakage-sensitive and must not replace the temporal-group result as the main
evaluation.

```bash
# Regenerate only if the frozen protocol file does not exist.
python \
  src/data/create_matched_random_protocol.py

python run_lecture_episode_cv.py \
  --stage prepare \
  --experiment_dir experiments/lecture_matched_random_cv \
  --episode_protocol experiments/lecture_matched_random_cv/matched_random_protocol.json

python run_lecture_episode_cv.py \
  --stage all \
  --experiment_dir experiments/lecture_matched_random_cv \
  --episode_protocol experiments/lecture_matched_random_cv/matched_random_protocol.json \
  --run_name smoothed_seed42
```

## Seed-42 controlled result

| Result | Macro-F1 | Accuracy | Test clips |
| --- | ---: | ---: | ---: |
| Fold 1 | 79.81% | 80.00% | 130 |
| Fold 2 | 84.88% | 88.17% | 93 |
| Fold 3 | 81.77% | 87.06% | 85 |
| **Pooled out-of-fold** | **83.12%** | **84.42%** | **308** |

Pooled class F1 was 87.73% Low, 83.21% Mid and 78.43% High. The matching
conservative temporal-group result was 55.31% pooled macro-F1. Because the
per-fold class counts, frozen matrices, model settings and training seed are
matched, the 27.81-point difference is strong evidence that random clip mixing
benefits from temporal correlation and does not measure temporal-block
generalization.

## Five-model-seed result

Across seeds 42--46, this fixed randomized protocol achieved **81.43% ±
1.66%** pooled Macro-F1 (sample SD; range 78.84--83.12%). The corresponding
conservative temporal protocol achieved **56.34% ± 1.77%**, giving a paired
gap of **25.10 ± 2.51 points**. The gap is positive for all five seeds. This
measures training-seed variation for one frozen random split; it is not
uncertainty over independent recordings.

The matching orientation-family ablation (interaction columns 2--7 and 30
omitted only at model input) achieved **80.15% ± 1.37%** over seeds 42--46,
compared with **81.43% ± 1.66%** for full V3. The paired change was **-1.28 ±
1.99 points**. Together with the larger temporal decline, this indicates that
the questionable fields contribute more to temporal High-class recognition;
it does not validate them as head pose, gaze, or attention.

See
[`docs/BEHAVIORAL_INVESTIGATION_RESULTS.md`](../../docs/BEHAVIORAL_INVESTIGATION_RESULTS.md)
for the complete seed table and interpretation.
