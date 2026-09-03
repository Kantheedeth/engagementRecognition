# Experiments V2: Legacy Golden-Pair Baseline

This package is an additive experiment layer. It does not replace or modify the
stable pipeline under `src/`.

Registered baseline methods:

- `A1` / `METHOD_A1`: `legacy_affect`
- `I1` / `METHOD_I1`: `legacy_interaction`

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

It reports separate readiness for reuse of legacy artifacts and regeneration
from preprocessed frames. Raw video alone is reported as requiring the existing
preprocessing stage first.

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
