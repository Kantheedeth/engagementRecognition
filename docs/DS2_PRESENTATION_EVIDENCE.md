# DS2 and Midterm Presentation Evidence Record

Last updated: 2026-10-01

This document is the source of truth for the revised project concept, DS2,
midterm presentation, and related discussion with the professor. It records the
project decisions and experimental evidence without relying on conversation
history. Generated run files remain the authoritative source for exact metrics.

The post-control multi-seed, extraction-audit, and branch-ablation investigation
is recorded separately in
[Behavioral Model Investigation Results](BEHAVIORAL_INVESTIGATION_RESULTS.md).

## 1. Current project scope

The system recognizes group engagement in **speaker-led classroom lecture
activity**. One person acts as the speaker and the other visible people are
listeners. The intended evidence is behavioral:

- interaction evidence describing listeners relative to the speaker;
- group-affect evidence describing the overall estimated affective climate;
- lightweight temporal modeling over eight sampled frames;
- Low, Mid, or High group-engagement prediction.

Round-table discussion, peer presentation, and group-work clips are outside the
current scope. Low-engagement lecture clips remain eligible; low engagement is
not a reason for exclusion.

The current model deliberately excludes scene embeddings. Early scene-based
results reached approximately 98--100% macro-F1, but the fixed room, camera,
participants, clothing, and lighting made scene shortcuts a serious concern.
The current behavioral matrix contains 32 interaction features and 8 affect
features for each of eight sampled frames.

## 2. Dataset development story

The project initially selected 1,195 OUC-CGE clips using classroom camera angle.
Manual review later showed that the same camera angle contained multiple
activities, including lecture, group work, and presentations. Activity-level
filtering using the one-speaker/multiple-listeners rule retained 308 lecture
clips:

| Class | Clips |
| --- | ---: |
| Low | 159 |
| Mid | 64 |
| High | 85 |
| **Total** | **308** |

The selected clips come from the same fixed recording environment and are
temporally indexed by `view####`. Adjacent source IDs are consecutive ten-second
segments. The dataset therefore has many highly correlated neighboring clips.
This prevents the project from claiming cross-classroom or cross-recording
generalization.

## 3. Behavioral pipeline development

The interaction branch was progressively changed from spatial representations
that retained seating coordinates to behavior-focused features. The current V3
pipeline:

- identifies the speaker using the instruction zone and target fallback;
- treats all other detected people as listeners;
- uses eight sampled frames per clip;
- uses Hungarian box/pose association to create anonymous identities across the
  eight sampled interaction frames;
- measures physical pose/orientation proxies without exporting spatial
  coordinates as predictive features;
- applies manually reviewed proxy rules for raised/lowered head and posture;
- combines the 32-D interaction vector with an 8-D tracked affect vector.

Important terminology:

- orientation is a pose-derived alignment proxy, not attention probability;
- preliminary visual review found that the current eye-to-nose image-plane
  direction often disagrees with visible head direction, so orientation-based
  explanations must currently be described as unvalidated;
- head/posture fields are physical proxies, not proof of attention, fatigue,
  phone use, or sleep;
- facial affect is a model estimate, not internal emotional ground truth.

## 4. Why the original 95.48% is preliminary

Filtering the existing CSV split to the 308 lecture clips produced:

- 252 training clips;
- 25 validation clips;
- 31 test clips;
- 95.48% test macro-F1;
- 96.77% accuracy (30/31 clips).

This result uses the original clip membership. Temporally adjacent clips from
the same recording can occur in different splits. It is therefore a useful
fixed-setup or benchmark-like baseline, but it is not the main generalization
result.

## 5. Temporal grouping experiments

### 5.1 Provisional 23-range evaluation

The selected clips initially formed 23 maximal contiguous filename ranges. A
three-fold grouped evaluation kept each candidate range inside one split. Every
clip received exactly one out-of-fold test prediction.

| Result | Macro-F1 |
| --- | ---: |
| Fold 1 | 57.34% |
| Fold 2 | 70.56% |
| Fold 3 | 55.95% |
| **Pooled** | **64.87%** |

This was retained as a sensitivity analysis. It was not conservative enough
because ranges separated by only one excluded ten-second source clip could
still enter different folds.

### 5.2 Conservative 18-group temporal evaluation

Ranges separated by one excluded source clip were assigned the same temporal
group:

- `695--696` and `698--702`;
- `846--880` and `882--895`;
- `1196--1197`, `1199`, and `1201`;
- `2470--2520` and `2522--2532`.

The excluded clips remain excluded and contribute no features or predictions.
Merging only changes split membership. The resulting 18 temporal groups were
divided into three outer folds, with complete temporal groups also held out for
validation.

| Result | Macro-F1 | Accuracy | Test clips |
| --- | ---: | ---: | ---: |
| Fold 1 | 33.04% | 44.62% | 130 |
| Fold 2 | 77.18% | 79.57% | 93 |
| Fold 3 | 69.71% | 81.18% | 85 |
| **Pooled** | **55.31%** | **65.26%** | **308** |

Pooled class F1:

- Low: 80.91%;
- Mid: 56.95%;
- High: 28.07%.

The large 62-clip High temporal group (`H04`) was held out in Fold 1, and none of
those clips were predicted as High in the seed-42 run. This exposes substantial
variation between High lecture periods and a shortage of independent High
groups. The multi-seed result below confirms that this is not explained by one
unlucky model seed.

## 6. Size-matched randomized control

### 6.1 What was controlled

The control tests whether the temporal score fell merely because each fold had
less training data or different class counts.

For every fold and every class, the randomized control uses exactly the same
numbers of training, validation, and test clips as the conservative temporal
experiment. For example, Fold 1 uses:

| Split | Low | Mid | High | Total |
| --- | ---: | ---: | ---: | ---: |
| Train | 98 | 42 | 18 | 158 |
| Validation | 11 | 4 | 5 | 20 |
| Test | 50 | 18 | 62 | 130 |

The same equality holds for Folds 2 and 3. The following were unchanged:

- all 308 behavioral matrices;
- feature schema and extraction;
- model architecture;
- loss and class-weight calculation;
- optimizer and early stopping;
- training seed 42.

The only intended split difference is that individual clips were randomized
instead of being kept inside temporal groups. Test folds partition all 308 clips
exactly once. The randomized membership is frozen with split seed `20260926`.

### 6.2 Controlled result

| Result | Macro-F1 | Accuracy | Test clips |
| --- | ---: | ---: | ---: |
| Fold 1 | 79.81% | 80.00% | 130 |
| Fold 2 | 84.88% | 88.17% | 93 |
| Fold 3 | 81.77% | 87.06% | 85 |
| **Pooled** | **83.12%** | **84.42%** | **308** |

Pooled class F1:

- Low: 87.73%;
- Mid: 83.21%;
- High: 78.43%.

### 6.3 Controlled comparison

| Protocol | Pooled Macro-F1 | Pooled accuracy |
| --- | ---: | ---: |
| Size-matched randomized clips | **83.12%** | 84.42% |
| Conservative temporal groups | **55.31%** | 65.26% |
| **Difference** | **27.81 points** | **19.16 points** |

Because sample counts and model settings are matched, smaller training sets do
not explain the entire decline. The result is strong evidence that random clip
splitting benefits from temporal correlation and that the current behavioral
model has limited transfer to unseen lecture periods.

### 6.4 Five-seed stability result

Seeds 42--46 were run with frozen features, folds, hyperparameters, and the one
randomized membership generated with split seed `20260926`.

| Protocol | Mean Macro-F1 | Sample SD | Range |
| --- | ---: | ---: | ---: |
| Size-matched randomized clips | **81.43%** | 1.66 | 78.84--83.12% |
| Conservative temporal groups | **56.34%** | 1.77 | 53.94--58.58% |
| Paired random-minus-temporal gap | **25.10 points** | 2.51 | 22.07--27.81 points |

The separation persists for all five model seeds. These numbers measure
training-seed variability for the two fixed protocols; they are not confidence
intervals over independent classrooms or recordings.

Automatic diagnostics found no broad H04 feature-availability collapse. H04 is
instead much stiller and lower-motion than the other High groups, with somewhat
lower face/affect reliability. Branch ablations also leave Fold 1 weak regardless
of whether interaction, affect, or both are used. Full evidence is in
[Behavioral Model Investigation Results](BEHAVIORAL_INVESTIGATION_RESULTS.md).

### 6.5 Preliminary visual proxy audit

Three clips have been manually completed so far (H04=2, H02=1). All three had
good pose placement but mixed or inconsistent orientation arrows, and every
clip reported an ID switch. Two had major tracking-continuity issues. All three
label-fit entries were `ambiguous`.

This is too small and unbalanced to compare High groups quantitatively. It is
nevertheless enough to reject the claim that the current purple orientation
arrow has been visually validated. The automatic reliability fields measured
whether a direction could be calculated, not whether that direction was
semantically correct.

### 6.6 Orientation-family ablation

The pre-specified diagnostic was completed without re-extraction. Interaction
columns 2--7 and 30 were sliced from the model input while the stored matrices,
folds, loss, and training settings remained frozen.

| Protocol | Full V3 | Without orientation family | Change |
| --- | ---: | ---: | ---: |
| Conservative temporal groups | **56.34% ± 1.77** | **51.85% ± 1.13** | **-4.49 ± 1.86 points** |
| Size-matched randomized clips | **81.43% ± 1.66** | **80.15% ± 1.37** | **-1.28 ± 1.99 points** |

The temporal decline occurred for all five seeds, and mean temporal High-class
F1 fell from 34.25% to 25.05%. Thus the model uses the orientation family, but
the ablation does not establish that its eye-to-nose direction is valid head
pose or attention. Combined with the visual mismatch, the result is best
reported as dependence on an unvalidated proxy. The orientation-removed runs
are diagnostic results, not a new model chosen for superior accuracy.

## 7. Claims supported by the current evidence

The project can currently say:

1. Scene-based evaluation is vulnerable to fixed-environment shortcuts in this
   dataset.
2. Behavioral-only features achieve strong performance when clips are randomly
   mixed across time.
3. Under exactly matched class/sample counts, temporal-group-held-out
   performance is substantially lower than randomized-clip performance.
4. The model therefore has limited robustness to unseen lecture periods within
   the same recording.
5. Leakage-aware evaluation is essential for interpreting results on segmented
   classroom video.
6. The controlled gap is stable over five predetermined model seeds.
7. H04 has a distinct behavioral distribution and the dataset has scarce
   independent High groups, while preliminary visual review additionally
   identifies orientation-validity and tracking risks.
8. Removing the orientation family reduces temporal Macro-F1 by 4.49 points on
   average, showing model reliance on those fields without validating their
   behavioral meaning.

## 8. Claims that must not be made

Do not claim that:

- the original papers or official benchmark results are invalid;
- the current model generalizes to new classrooms, cameras, participant groups,
  or recording sessions;
- 55.31% is a cross-domain score;
- the five-seed variance represents uncertainty over new recordings or
  classrooms;
- pose orientation is direct attention measurement;
- the current orientation arrow is visually validated or equivalent to gaze;
- three reviewed clips establish error rates for all High groups;
- facial affect is emotional ground truth;
- the score decline proves one specific feature is defective or semantically
  correct.

The official/original CSV and the temporal protocol answer different research
questions. Benchmark-like scores should be reported as such, not compared as if
they were independent-session results.

## 9. What to include in the revised DS2 concept

The revised concept should retain the original motivation but openly describe
the project evolution.

Suggested structure:

1. **Introduction:** explain group-engagement recognition, teacher utility, and
   the danger of contextual shortcuts in fixed classroom video.
2. **Problem refinement:** explain why the scope was restricted to
   one-speaker/multiple-listener lecture activity.
3. **Dataset and data audit:** describe activity filtering, the 308-clip subset,
   temporal adjacency, and the lack of an external compatible dataset.
4. **Requirements specification:** require lecture-activity input, speaker and
   listener separation, lightweight behavioral extraction, class prediction,
   interpretable evidence output, and reproducible leakage-aware evaluation.
5. **Framework:** show interaction extraction, group affect, reliability-aware
   behavioral fusion, temporal modeling, and Low/Mid/High output. Scene features
   should be shown as investigated and excluded from the current predictive
   architecture, not silently removed from the project history.
6. **Evaluation design:** distinguish the original clip split, conservative
   temporal grouping, and size-matched randomized control.
7. **Current results:** present the five-seed means, 81.43% randomized versus
   56.34% temporal (25.10-point paired gap), including the one-recording
   limitation.
8. **Demonstration/mockup:** show detected speaker/listeners, behavioral fields,
   group-affect summary, engagement output, and explanation panel.
9. **Completed diagnostic and next work:** report the orientation-family
   ablation (51.85% temporal; 80.15% matched-random), complete a small balanced
   proxy audit if time permits, and investigate a proper head-pose replacement
   after midterm.

## 10. Seven-minute presentation story

Use the experiment as a concise research story rather than listing every model
version.

1. **Problem and intended user:** classroom group engagement and explainable
   behavioral evidence.
2. **Original plan:** lightweight scene, interaction, and affect decomposition.
3. **Finding 1:** scene features reached nearly perfect scores because the
   recording environment was fixed; scene was removed from prediction.
4. **Finding 2:** camera-angle filtering still included group work and
   presentations; activity review reduced the data to 308 lecture clips.
5. **Behavioral framework:** speaker zone/fallback, tracked listeners,
   interaction proxies, group affect, eight-frame temporal classifier.
6. **Evaluation problem:** neighboring ten-second clips were distributed across
   original splits.
7. **Controlled evidence:** show one table or bar chart comparing the five-seed
   means: 81.43% size-matched random versus 56.34% temporal grouping.
8. **Conclusion:** the lightweight model works under random fixed-recording
   evaluation but does not yet robustly generalize between lecture periods.
9. **Latest diagnostic and next work:** orientation removal lowers temporal
   performance, showing reliance on an unvalidated proxy; pursue a validated
   head-pose replacement rather than unlabelled action recognition.

For a seven-minute presentation, the controlled comparison is more important
than explaining all 40 feature dimensions. Explain feature families and show a
single example; keep the exact field table available for questions.

## 11. Likely committee questions

### Why did performance decrease?

The score decreased when temporally related clips were prevented from crossing
splits. The exact size-matched random control recovered 83.12%, while temporal
grouping achieved 55.31%, showing that training-set size alone is not the main
explanation.

### Is 55.31% a bad model?

It is evidence of limited temporal-block generalization, not a failed
implementation. It reveals that the dataset contains few independent behavioral
conditions, especially for High engagement.

### Why not use scene features?

The fixed camera, room, participants, clothing, and lighting allow scene
features to identify recording context instead of engagement behavior.

### Why merge ranges across excluded activities?

The excluded clips remain unused. Immediately surrounding selected clips share
a temporal group only because a ten-second interruption does not make their
visual and behavioral context independent.

### Does this generalize to other classrooms?

No. All current evidence comes from one fixed recording environment. The
temporal evaluation is stricter but remains within-recording.

## 12. Frozen artifacts and provenance

- Preliminary lecture-only baseline: `experiments/lecture_only/`
- Provisional 23-range sensitivity analysis: `experiments/lecture_episode_cv/`
- Main conservative temporal protocol: `experiments/lecture_temporal_group_cv/`
- Exact matched randomized control: `experiments/lecture_matched_random_cv/`
- Episode/CV runner: `run_lecture_episode_cv.py`
- Temporal fold builder: `src/data/prepare_episode_cv.py`
- Matched-random protocol generator:
  `src/data/create_matched_random_protocol.py`

The generated matrices and run outputs are ignored by Git, while the protocol
definitions, code, and summary documentation are intended to be versioned.

## 13. Completed frozen investigation

The predetermined investigation has been completed without changing split
membership, feature matrices, extraction thresholds, hyperparameters, or loss:

- seeds 42--46 on both temporal and size-matched randomized protocols;
- 308-matrix temporal-group feature diagnostics;
- deterministic raw tracking diagnostics on 36 clips;
- interaction-only, affect-only, and combined ablations on the same temporal
  folds and seeds;
- orientation-family removal on both frozen protocols for seeds 42--46; and
- an evidence-based decision to retain V3 as the evaluated baseline rather than
  tune a V4 directly against H04.

Subsequent preliminary visual review found a repeated semantic mismatch in the
orientation arrows and ID-switch issues in three clips. These findings do not
invalidate the recorded V3 evaluations, but they limit the explanations that
can be claimed. The subsequently completed orientation-removal ablation reduced
mean Macro-F1 from 56.34% to 51.85% temporally and from 81.43% to 80.15% in the
matched-random control. This establishes model reliance, not semantic validity.
See Steps 5--6 of the detailed investigation record.

The detailed seed tables, H04 comparisons, ablations, limitations, and decision
are recorded in
[Behavioral Model Investigation Results](BEHAVIORAL_INVESTIGATION_RESULTS.md).
