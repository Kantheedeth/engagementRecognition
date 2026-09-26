# Lecture-only behavioral experiment

This experiment copies existing (8, 40) matrices for the manually selected
`lecture_only_videos/{class}/*.mp4` files. It never decodes videos or runs feature
extraction. Original CSVs, matrices, checkpoints and reports are left in place.
Matching uses **class + filename**, not the numeric video ID alone.

## Initial snapshot

| Split | Low | Mid | High | Total |
| --- | ---: | ---: | ---: | ---: |
| Train | 133 | 55 | 64 | 252 |
| Validation | 10 | 3 | 12 | 25 |
| Test | 16 | 6 | 9 | 31 |
| Total | 159 | 64 | 85 | 308 |

This preserves original split assignments after activity filtering. It is a
**preliminary fixed-setup baseline**, not a recording-held-out or cross-camera
evaluation. Related source footage can still span splits. The small validation
and test sets make scores sensitive to individual clips. Do not select loss
settings, hyperparameters or seeds based on test results. Historical proxy
calibration is inherited unchanged; this setup does not establish that the test
clips were excluded from earlier manual development audits.

## Commands

Run from the project root in a compatible Python environment (see collaborator setup below):

```bash
# Prepare once. Refuses to overwrite an existing snapshot.
python run_lecture_behavioral.py --stage prepare

# Train a fresh model on ONLY the selected training matrices.
python run_lecture_behavioral.py --stage train

# Evaluate the matching lecture-only checkpoint, after configuration is fixed.
python run_lecture_behavioral.py --stage eval
```

The default run name is `smoothed_seed42`. Existing training runs cannot be
overwritten. Use `--run_name NAME` for a new experiment and the same name when
evaluating it. `--stage all` prepares a missing snapshot, or verifies an existing
one, then trains and evaluates. It does not refresh an existing selection.

For another snapshot, pass `--experiment_dir experiments/lecture_only_v2` to
every command. If the selection count changes, set `--expected_count N` explicitly.
Changing the video folder later does not mutate the prepared snapshot.

## Collaborator setup

Use this branch, not the separate 40-interaction role-aware pipeline:

```bash
git fetch origin
git switch --track origin/docs/lecture-only-results-20260926
# If the branch already exists locally: git switch docs/lecture-only-results-20260926
python -m pip install -r requirements-behavioral.txt
python -m unittest discover -s tests -v
```

Use a clean checkout or preserve your local edits before switching branches.
Python 3.10+ is recommended. Install matching PyTorch/torchvision builds suitable
for your CPU/CUDA platform. The seed-42 training run used Python 3.10.20,
PyTorch 2.13.0, torchvision 0.28.0, NumPy 2.2.6 and scikit-learn 1.7.2 in the
local `LEM` environment. These are observed versions, not a portable environment
lock or a guarantee of identical results on another device. Extraction provenance
records its own settings and software versions.

### If compatible V3 matrices are available

Obtain `feature_matrices_behavioral/` (including `build_manifest.json` and its
train/val/test matrices) separately from the project owner. Do not use the
`feature_matrices_behavioral_before_tracking/` backup or a 48-column matrix set.
The small tracked `selection.txt` lists the exact activity-reviewed clips; no
local `lecture_only_videos/` copy is required in this mode:

```bash
python run_lecture_behavioral.py --stage prepare \
  --selection_file experiments/lecture_only/selection.txt
python run_lecture_behavioral.py --stage train
python run_lecture_behavioral.py --stage eval
```

Use `--source_dir /path/to/feature_matrices_behavioral` during preparation if the
matrices are elsewhere. Use `--csv_dir` only for the matching original split
lists; changing assignments changes the experiment. The repository's training
CSV excludes round-table `mid/view1230.mp4`.

Alternatively, obtain the complete prepared `experiments/lecture_only/dataset/`
folder, including both manifests and split lists. Then skip preparation and run
training/evaluation. The source paths stored in its manifest are provenance;
verification of the snapshot does not require those paths to exist on your machine.

### If matrices are unavailable

The code is shared, not the dataset or model weights. You need your own authorized
OUC-CGE files arranged at the exact `videos/{class}/viewN.mp4` paths in the root
split lists (including their capitalization on case-sensitive systems), plus
the YOLO pose and affect model assets used by the extractors. Install
`requirements-affect.txt` as well if regenerating affect features.

On a fresh data/output layout, the extraction workflow is:

```bash
python src/data/preprocess_frames.py
python src/data/extract_interaction_features.py --device auto
python src/data/extract_affect_features.py --save_track_details
python src/data/build_behavioral_matrices.py
```

Then prepare the lecture-only snapshot using `--selection_file` as above. This
regenerates the broader source matrices before selecting 308 clips and requires
the corresponding source data. It is unnecessary if compatible matrices are
already available. Use fresh output directories instead of mixing old schemas
or leaving stale matrix files. Regenerated feature versions/hashes can differ;
do not claim exact reproduction of 95.48% without checking the resulting provenance.

## Loss and initial configuration

Use weighted cross-entropy with square-root inverse-frequency weights, calculated
from the **252 training labels only**:

`raw_weight[c] = sqrt(N_train / (3 * count_train[c]))`

Weights are normalized to mean one. `smoothed` refers to these class weights,
not label smoothing. Class weights are re-estimated for the subset rather than
copied from the full dataset. The smaller total alone does not require a different
loss. No oversampling or augmentation is added.

Initial settings preserve the direct trainer's defaults: branch dimension 64,
4 attention heads, dropout 0.20, batch size 32, AdamW learning rate 0.001,
weight decay 0.0001, up to 60 epochs and patience 15. Checkpoint selection uses
validation loss. These are starting settings, not claimed optimal settings.

Optional controlled comparisons (choose using validation only):

```bash
python run_lecture_behavioral.py --stage train --class_weight_mode none --run_name unweighted_seed42
python run_lecture_behavioral.py --stage train --label_smoothing 0.05 --run_name smoothing005_seed42
```

`--class_weight_mode balanced` uses full inverse-frequency weighting. Default
label smoothing remains **0.0**. Change one factor at a time. With just three
Mid validation clips, extensive tuning would readily overfit validation.

Weighted epoch loss is now accumulated using the sum of target-class weights,
matching PyTorch's weighted-mean CE denominator, rather than weighting batch
means by batch size. This fixes reporting and checkpoint comparison across
unequal class-mixed batches; the per-batch optimization objective is unchanged.
See [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

## Files and safeguards

- `selection.txt`: tracked, portable class/filename list for the 308 selected clips.
- `dataset/splits/`: filtered copies of the original split lists.
- `dataset/matrices/`: independent matrix copies, with a subset-specific build manifest.
- `dataset/selection_manifest.json`: exact clip identities, class counts, source paths and hashes.
- `runs/<run_name>/checkpoints/`: model checkpoint and training history.
- `runs/<run_name>/reports/`: logs, confusion matrix, metrics and per-clip predictions.
- `runs/<run_name>/run_config.json`: exact training command and settings.

Preparation validates matrix shapes, finite values, label consistency, duplicate
clip keys and split coverage. Every runner invocation checks snapshot membership
and file hashes to reject stale extra matrices or changed files. Checkpoints carry
the subset provenance, preventing the full-dataset checkpoint from being accepted
as a lecture-only model. The source extraction metadata inside the build manifest
still describes the original extraction, while top-level counts describe this subset.
