# Listener posture mismatch audit

Open `index.html` in a browser. Everything runs locally; no server or upload is
needed. Review without revealing predictions, then reveal the current rule
outputs to investigate disagreements. The original frame toggle hides all boxes.
Download annotations CSV and progress JSON before closing; browser storage on
local files may not persist. Restoring JSON replaces the browser's current audit
progress, so download a backup first. Give each annotator a separate copy or
browser profile and a distinct annotator name.

## What to annotate

Each P-number identifies a detection in this frame only, not a tracked student.
Use the surrounding frame to interpret the numbered person's visible geometry.

- **Role:** listener, speaker, unclear. Correct incorrect listener assignments.
- **Head position:** up, level, down, unknown. Judge visible head tilt; missing
  facial keypoints alone are not evidence that the head is down.
- **Posture:** upright, forward lean, head rest, other, unknown. Head rest means
  visible support/contact with a desk, arms or hand; it is not a sleep label.
- **Facing:** toward instruction area, away, unknown. This is observable coarse
  head/body orientation, not eye gaze, listening, or internal attention.
- **Activity (optional):** writing, reading, device use, other, unknown. Select
  only what the frame supports. A still frame may not establish reading/writing;
  use unknown if necessary. Device use alone does not establish off-task behavior.
- **Notes:** describe ambiguous cases and mismatches. Use frame notes for missed
  people, duplicate boxes, or target problems not represented by listener rows.

Unknown is an explicit annotation; unlabeled means not yet reviewed. There are
no generated human labels. Session identities and engagement classes are hidden
in the default review. Source paths appear only when predictions are revealed.

## What this audit measures

The images are the actual 640x640 stretched RGB frames used by the current
extractor, converted to BGR for YOLO. They are not full-resolution source images.
Each selected frame gets raw, numbered-box, and diagnostic images. Detection and
keypoint arrays, confidence, rule outputs, input hashes and extractor/checkpoint
hashes are saved in audit.json. Posture/orientation values are obtained by calling
the production extractor on each listener plus the detected speaker, then checked
against the full-frame aggregates. No replacement posture rules are installed.

Target selection is the current instruction-zone speaker candidate or the zone
center fallback. The zone is retained. If the out-of-zone speaker is treated as
a listener by the current extractor, mark their role as speaker and add a note.

The flags expose questionable CURRENT rules: low face confidence can trigger
slump, missing elevation defaults to 0.08 (classified as slouching), missing pitch
can be imputed, and slump forces orientation to zero. These are diagnostics,
not claims that the rules are valid. The label "slumped" is not a diagnosis of sleep.

No motion, persistent tracking, FER, engagement classifier, or thresholds are
changed by this tool. Eight sparse frames cannot establish continuous behavior
or duration between samples. Human review must precede accuracy claims.

## Sampling and subsequent evaluation

Default: two training clips from each of six user-supplied numeric ranges, three
sampled frames per clip (zero-based indices 1,4,6). The ranges are provisional
sampling strata, not verified independent sessions. Duplicate IDs are collapsed
within train; IDs present in validation/test CSVs and ambiguous IDs associated
with multiple training paths are excluded and recorded in audit.json. This is an
exploratory development set, not a held-out proxy validation benchmark. Freeze a
separate set of clips for evaluation before using these annotations to tune rules.
Repeated people/frames are correlated observations, not independent samples.

Later comparisons should report unknown/visibility coverage and task-specific
agreement on assessable, manually labeled listeners. "Forward lean" and "head
rest" do not map automatically to the current "slumped" flag: agree on the
annotation rubric first. No p-values, sleep sensitivity, or accuracy are produced
by this unannotated audit.

## Reproduce

```bash
python src/tools/create_listener_posture_audit.py --device cpu
```

Use `--clips-per-range 1 --frame-indices 4` for a smaller pilot. An explicit
`--output-dir` must not exist; this prevents overwriting annotations. Outputs go
under the ignored `audit_outputs/` directory by default. Original images, matrices,
checkpoints and existing audits remain untouched.
