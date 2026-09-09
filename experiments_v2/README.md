# Experiments V2: Legacy Golden-Pair Baseline

This package is an additive experiment layer. It does not replace or modify the
stable pipeline under `src/`.

Registered baseline methods:

- `A1` / `METHOD_A1`: `legacy_affect`
- `I1` / `METHOD_I1`: `legacy_interaction`

## Certified A1 + I1 baseline

The frozen certified baseline is:

- baseline: `BASELINE_20260909T081236880690Z_1C3F901E`
- pair: `PAIR_20260909T065613679907Z_35FC674B`
- matrix: `MATRIX_20260909T081205334857Z_227ED566`
- run: `RUN_20260909T081205329267Z_641DB332`
- checkpoint: `CHECKPOINT_20260909T081211467023Z_D17C3274`
- A1 feature: `FEATURE_20260909T032903202647Z_9B3C9779`
- I1 feature: `FEATURE_20260908T113809721154Z_474E2629`

Its fixed test metrics are 85.6061% accuracy, 83.6910% macro precision,
84.5132% macro recall, and 83.9936% macro F1. The confusion matrix is
`[[62, 5, 5], [4, 19, 2], [3, 0, 32]]`.

The certified matrix set contains 939 train, 124 validation, and 132 test
samples. Each matrix is float32 with shape `(8, 40)` and the metadata-derived
layout I1 `[0:32]` followed by A1 `[32:40]`. The certification records the
V2-only `bytetrack_singleton_numpy_mask_v1` compatibility provenance for A1.
Generated artifacts remain ignored; these identifiers document the immutable
local certification records rather than adding model/data binaries to Git.

Each method publishes `category`, `feature_dim`, and `feature_schema` metadata.
Pair manifests turn those declarations into a contiguous `feature_layout`; the
baseline configuration explicitly selects `interaction` then `affect`. Matrix
width and engagement branch widths are derived from that layout, so `40` is a
property of A1 (8) + I1 (32), not a framework-wide constant.

Inspect the resolved baseline plan without loading ML dependencies:

```bash
python experiments_v2/runner.py plan
```

Run extraction, pair building, engagement training, and evaluation:

```bash
python experiments_v2/runner.py run
```

The run requires the same preprocessed data and Python dependencies as the
legacy pipeline. Published artifacts under `experiments_v2/artifacts/` are
versioned and are never overwritten. Setting `force_extract` creates a new
feature version; it does not replace a previously published cache.

Run the V2 contract and artifact-safety checks with:

```bash
python -m unittest discover -s experiments_v2/tests -v
```

The numerical matrix-builder test needs NumPy, which is also a runtime
dependency of the legacy training pipeline.

## Certification gate

The official baseline requires the documented Python 3.10 environment. The
checker never installs packages:

```bash
python -m experiments_v2 preflight \
  --config experiments_v2/config/baseline_legacy.json
```

It reports separate readiness for reuse of legacy artifacts, building from
complete immutable V2 features, and regeneration from preprocessed frames. Raw
video alone is reported as requiring the existing preprocessing stage first.

Large data can remain outside the repository. Copy `baseline_legacy.json` to an
ignored `*.local.json` file and set `certification.paths` there. In particular,
`dataset_root`, `preprocessed_root`, `legacy_feature_root`,
`legacy_matrix_root`, and `legacy_checkpoint_root` may be absolute external
paths. No local absolute path is committed.

After preflight reports `READY`, run:

```bash
python -m experiments_v2 certify-baseline \
  --config experiments_v2/config/certification.local.json
```

Only this command passes the successful-preflight marker that permits official
baseline publication. A normal `experiments_v2/runner.py run` can create an
ordinary immutable run but cannot publish the official baseline.

Certification binds approved `MODEL_*` and `FEATURE_*` identities before cache
resolution, performs a three-sample exact segment-equivalence gate, validates
and checksums all matrices, verifies the engagement parameter contract, selects
the checkpoint using validation loss only, and evaluates the test split only
after model selection.

## Fixed Affect comparison protocol

The first future Affect comparison is A1 + I1 versus A2 + I1. I1 and the
engagement protocol remain fixed: seed 42, batch size 32, AdamW at `1e-3`,
weight decay `1e-4`, cosine annealing, 60 requested epochs, patience 15,
weighted cross-entropy, and minimum validation loss for checkpoint selection.
Future candidates compare against 85.6061% accuracy, 83.9936% macro F1, and a
335.993 MB combined A1 + I1 learned-weight footprint. The comparison metadata
provides accuracy, macro-F1, model-size, parameter, extraction-time,
inference-time, and FPS deltas. No aggregate Golden Pair score is defined.
