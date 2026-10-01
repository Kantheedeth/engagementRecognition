# High-group visual audit guide

## Purpose

This audit compares `H04`, the large High group held out in Fold 1, with
`H01`--`H03`. It checks whether H04 fails because the saved extraction is
visibly wrong or because valid observable behavior differs across High groups.

The generator targets all 85 selected High clips. Their order is
deterministically shuffled, and the normal review screen hides the filename and
temporal-group identity. If cloud-backed source files are not downloaded, the
generated package records and skips them rather than hanging. Complete the
annotations before using **Reveal groups and comparison** whenever possible.

## What the overlay means

- Orange boxes: a person identified as the current or previously known speaker.
- Colored listener boxes: anonymous track IDs local to this eight-frame clip.
- Skeleton lines: YOLOv8-Pose keypoints whose confidence exceeds 0.30.
- Purple arrow: the same coarse face/torso direction used by the interaction
  proxy. It is not eye gaze or an attention probability.
- Yellow cross: detected speaker center, or the instruction-zone fallback when
  the speaker is not detected.
- Orange rectangle: instruction zone.

The interaction tracker is Hungarian box/pose association across the eight
sampled frames. ByteTrack is used by the separate affect branch, not by this V3
interaction overlay.

## Annotation fields

- **Speaker-led lecture activity:** whether the clip still satisfies the project
  scope.
- **Speaker identification:** whether the orange speaker role is correct across
  frames.
- **Listener detection coverage:** whether visible listeners are detected and
  non-people are avoided.
- **Track-ID continuity:** whether the same visible person generally retains the
  same ID across sampled frames.
- **Pose keypoint placement:** whether keypoints lie on the corresponding body
  parts when visible.
- **Head/posture proxy:** whether the visible head elevation and posture implied
  by the pose are reasonable.
- **Orientation arrows:** whether the purple coarse direction agrees with the
  visible head/torso direction. Use `unclear` under occlusion.
- **Visible behavior:** descriptive context only. It is not used as an action
  label or automatically interpreted as engagement.
- **High-label fit:** whether the dataset High label looks plausible from the
  available visual evidence. This is an ambiguity audit, not relabeling.

## Recommended procedure

1. Enter the annotator name or initials.
2. Inspect all eight overlay frames for one audit code.
3. Toggle **Show raw frames** to check anything covered by an overlay.
4. Select `unclear` whenever the image does not support a judgment.
5. Record problem frame numbers and a short factual note for major issues.
6. Download progress JSON periodically and the CSV when finished.
7. Reveal the groups only after completing the audit.

Do not change thresholds or extraction based directly on these held-out clips.
If a systematic defect is found, define a new V4 hypothesis and evaluate it
with a newly frozen protocol rather than silently replacing V3.
