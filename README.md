# Multi-Modal Student Engagement Recognition in Classroom Videos

A lightweight, real-time deep learning pipeline designed to recognize student engagement levels (**Low**, **Mid**, **High**) from 10-second classroom video clips using offline multi-modal feature extraction, multi-branch balanced attention, and standalone track-aware pure-behavioral modeling.

---

## 📌 Architectures & Pipelines

This repository supports two execution modes:
1. **Multi-Branch Balanced Fusion** (`16/32/32`): Fuses Scene (MobileNetV3), Interaction (YOLOv8), and Track-Aware Affect (RetinaFace + ByteTrack + ViT FER) while allocating **80% of embedding representation to behavioral signals**.
2. **Pure Behavioral Pipeline** (Zero Scene Shortcut): Strips away visual background features entirely, relying **100% on student body posture, spatial density, and facial expressions**.

> **Baseline stability:** The pipelines and source files documented below are the
> frozen, reproducible baseline. Future method-comparison work must be added as a
> separate `experiments_v2/` layer and must not refactor, rename, move, or delete
> the existing implementation.

```text
========================================================================================
                          MULTI-BRANCH BALANCED PIPELINE (80-dim)
========================================================================================
 10-Second Video Clip (1,195 Samples) ──► 8 Uniformly Sampled Frames
                                                   │
  ┌────────────────────────────────────────────────┼──────────────────────────────────┐
  │ (160x160)                                      │ (640x640)                        │ (640x640)
  ▼                                                ▼                                  ▼
┌──────────────────┐               ┌───────────────────────────────┐  ┌───────────────────────────────┐
│ MobileNetV3      │               │ YOLOv8 32-dim Interaction     │  │ RetinaFace + ByteTrack        │
│ Small (576-dim)  │               │ Geometry & Density (32-dim)   │  │ PyTorch FER (7 + reliability) │
└────────┬─────────┘               └───────────────┬───────────────┘  └───────────────┬───────────────┘
         │                                         │                                  │
         ▼                                         ▼                                  ▼
┌──────────────────┐               ┌───────────────────────────────┐  ┌───────────────────────────────┐
│ Scene Branch     │               │ Interaction Branch            │  │ Affect Branch                 │
│ Linear(576, 16)  │               │ Linear(32, 32)                │  │ Linear(8, 32)                 │
└────────┬─────────┘               └───────────────┬───────────────┘  └───────────────┬───────────────┘
         │ (16-dim, 20%)                           │ (32-dim, 40%)                    │ (32-dim, 40%)
         └─────────────────────────────────────────┼──────────────────────────────────┘
                                                   ▼
                                  [ Fused Embedding State: (8, 80) ]
                                      (Behavioral Signals = 80%)
                                                   │
                                                   ▼
                               ┌───────────────────────────────────────┐
                               │  Temporal Multi-Head Self-Attention   │
                               │  • Multi-Head Attention (4 heads)     │
                               │  • Residual + LayerNorm + Dropout     │
                               │  • Temporal Mean Pooling → (80,)      │
                               │  • Classifier Head (80 → 3 Classes)   │
                               └───────────────────┬───────────────────┘
                                                   ▼
                                 [ Engagement: Low / Mid / High ]

========================================================================================
                    STANDALONE PURE BEHAVIORAL PIPELINE (40-dim ➔ 96-dim)
========================================================================================
 YOLO Interaction (32-dim) ──► Linear(32, 48) ──┐
                                                ├──► [ 96-dim ] ──► Temporal Attention ──► Output
 Track-Aware Affect (8-dim) ─► Linear(8,  48) ──┘
 (Zero Scene Features / Zero Background Memorization)
```

---

## 🧪 Planned Additive V2: Affect × Interaction Golden-Pair Search

The next research layer will compare versioned Affect and Interaction methods
while leaving the existing project unchanged. Scene extraction remains available
in the baseline but is excluded from the default golden-pair search.

```text
                    FROZEN EXISTING PROJECT
                              │
                    Legacy adapters/wrappers
                              │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
           Affect methods               Interaction methods
       Legacy, A2, A3, ...           Legacy, I2, I3, ...
                 │                           │
                 ▼                           ▼
      Versioned saved features       Versioned saved features
                 └─────────────┬─────────────┘
                              ▼
                  Affect × Interaction pair
                              │
                              ▼
                 Train engagement classifier
                              │
                              ▼
               Immutable engagement checkpoint
                              │
                              ▼
                          Evaluate
                              │
                              ▼
       Accuracy / F1 / Size / Parameters / FPS / Time
                              │
                              ▼
                  Pareto-optimal golden pairs
```

The V2 system is planned around these rules:

- Existing extractors, models, builders, trainers, and evaluators remain usable
  without modification.
- Legacy Affect and Interaction enter V2 through adapters and form the first
  reproducibility baseline pair.
- A method may contain one model, several models, method-specific preprocessing,
  and internal fusion logic.
- Model checkpoints, feature sets, pair definitions, engagement checkpoints, and
  results receive immutable version IDs and are never automatically overwritten.
- Compatible saved features are reused. A forced training or extraction request
  creates a new version instead of replacing an old one.
- Registered Affect and Interaction versions are paired using their Cartesian
  product, so adding a method does not require editing hard-coded pair lists.
- Pair comparisons retain raw accuracy, precision, recall, F1, confusion matrix,
  model size, parameter count, extraction time, engagement inference time, and
  FPS where meaningful.
- Golden-pair selection reports performance, efficiency, and Pareto-optimal
  trade-offs rather than using accuracy alone or an implicit combined score.
- Dataset splits, engagement architecture, optimizer, learning rate, epochs,
  batch size, evaluation procedure, and random seed remain fixed unless an
  experiment explicitly changes them.

This section describes the approved architecture direction. The `experiments_v2/`
implementation will be introduced incrementally on a dedicated branch.

---

## 📊 Reproducible Evaluation Status

The checkpoint currently available locally for the pure-behavioral model was
evaluated against all 132 files in `feature_matrices_behavioral/test`. The
checkpoint is ignored by Git, so these numbers are not reproducible from a
fresh clone until that checkpoint is supplied or the model is retrained.

| Evaluation Setup | Test Accuracy | Macro-F1 | Status |
| :--- | :---: | :---: | :--- |
| **Pure Behavioral (Zero Scene)** | **84.09%** | **82.69%** | Reproduced on the current 132-sample test matrices |
| **Majority Class Baseline** | 54.55% | 23.53% | Computed from the same test labels |
| **Multi-Branch Balanced Fusion** | — | — | No matching 616-D checkpoint is currently available for verification |

The previously documented 100% multi-branch and 88% behavioral results were
not reproduced by the checkpoint and matrices currently present, so they are
not reported as verified results here.

### Pure Behavioral Model: Current Checkpoint

```text
=================================================================
Running Pure Behavioral Pipeline Evaluation Phase...
  • Features: 32 Interaction + 8 Affect (ZERO Scene Features)
=================================================================
              precision    recall  f1-score   support

         Low       0.91      0.83      0.87        72
         Mid       0.72      0.84      0.78        25
        High       0.81      0.86      0.83        35

    accuracy                           0.84       132
   macro avg       0.81      0.84      0.83       132
weighted avg       0.85      0.84      0.84       132

Model Macro-F1 Score:     82.69%
Baseline (Always Low) F1:  23.53%
=================================================================
```

Confusion matrix (rows are true classes; columns are predictions):

```text
[[60, 6, 6],
 [ 3,21, 1],
 [ 3, 2,30]]
```

---

## 🛠️ Requirements & Installation

Recommended Python version: `3.10` (or `3.8+` with compatibility shims).

```bash
# 1. Clone repository
git clone https://github.com/Kantheedeth/engagementRecognition.git
cd engagementRecognition

# 2. Create and activate conda environment
conda create -n engagement python=3.10 -y
conda activate engagement

# 3. Install dependencies
pip install torch torchvision opencv-python numpy tqdm scikit-learn matplotlib ultralytics insightface transformers lap
```

---

## 🚀 Execution Guide

### Option 1: Master Runner Scripts (Recommended)

You can run either pipeline end-to-end or stage-by-stage using the unified root entrypoints:

#### A. Pure Behavioral Pipeline (Zero Scene Shortcut: 32 Inter + 8 Affect)
```bash
# Run full pipeline end-to-end (extract -> build -> train -> eval)
python run_behavioral.py --stage all

# Or run individual stages:
python run_behavioral.py --stage extract    # Extract interaction & affect features
python run_behavioral.py --stage build      # Build 40-dim behavioral matrices
python run_behavioral.py --stage train      # Train pure behavioral attention model
python run_behavioral.py --stage eval       # Evaluate test set & plot confusion matrix
```

#### B. Multi-Branch Balanced Pipeline (Scene 16 + Inter 32 + Affect 32 = 80-dim)
```bash
# Run full multi-modal pipeline end-to-end
python run_multimodal.py --stage all

# Or run individual stages:
python run_multimodal.py --stage extract    # Extract scene, interaction & affect
python run_multimodal.py --stage build      # Build 616-dim multi-branch matrices
python run_multimodal.py --stage train      # Train multi-branch balanced model
python run_multimodal.py --stage eval       # Evaluate test set & plot confusion matrix
```

---

### Option 2: Step-by-Step Module Execution

#### Phase 1: Video Preprocessing
Extracts 8 uniformly spaced frames per 10-second video:
```bash
python src/data/preprocess_frames.py
# (Optional) Verify preprocessed frame colors & aspect ratios
python src/data/validate_preprocessing.py
```

#### Phase 2: Feature Extraction
Extract numerical vectors offline from the three modules:
```bash
# 1. Scene features (MobileNetV3-Small -> 576-dim)
python src/data/extract_scene_features.py

# 2. Rich interaction features (YOLOv8 layout + student coordinates -> 32-dim)
python src/data/extract_interaction_features.py

# 3. Track-aware affect (RetinaFace + ByteTrack + PyTorch FER -> 8-dim)
python src/data/extract_affect_features.py --save_track_details
```

#### Phase 3 & 4: Model Training & Evaluation
```bash
# Pure Behavioral Workflow:
python src/data/build_behavioral_matrices.py
python src/training/train_behavioral.py
python src/training/evaluate_behavioral.py

# Multi-Branch Workflow:
python src/data/build_feature_matrices.py
python src/training/train.py --scene_branch_dim 16 --inter_branch_dim 32 --affect_branch_dim 32
python src/training/evaluate.py --scene_branch_dim 16 --inter_branch_dim 32 --affect_branch_dim 32
```

---

### Audit: Camera Drift & Track Inspection
```bash
# Visual VFOA camera drift audit
python src/tools/audit_camera_drift.py

# Visualize saved anonymous face tracks and affect estimates
python src/tools/create_affect_audit_view.py \
  --track_json debug_validation/affect_tracks/train/low/view1000.json
```

---

## 📁 Repository Structure

```text
engagementRecognition/
├── src/                               # Core Source Code Package
│   ├── data/                          # Phase 1 & 2: Preprocessing & Feature Extraction
│   │   ├── preprocess_frames.py       # Dual-branch video frame preprocessor
│   │   ├── validate_preprocessing.py  # Visual validation gate
│   │   ├── extract_scene_features.py  # MobileNetV3 scene extraction (576-dim)
│   │   ├── extract_interaction_features.py # YOLOv8 interaction extraction (32-dim)
│   │   ├── extract_affect_features.py # RetinaFace + ByteTrack + FER extraction (8-dim)
│   │   ├── affect_module.py           # RetinaFace + ByteTrack + FER aggregation engine
│   │   ├── build_feature_matrices.py  # Builds 616-dim multi-branch matrices
│   │   ├── build_behavioral_matrices.py # Builds 40-dim pure behavioral matrices
│   │   └── feature_schema.py          # Cross-stage feature contract definitions
│   │
│   ├── models/                        # Neural Network Architectures & Datasets
│   │   ├── dataset.py                 # PyTorch Dataset loader with manifest validation
│   │   ├── model.py                   # Multi-Branch Balanced Attention Network (16/32/32)
│   │   └── model_behavioral.py        # Pure Behavioral Attention Network (48/48)
│   │
│   ├── training/                      # Training & Evaluation Loops
│   │   ├── train.py                   # Multi-Branch training loop
│   │   ├── evaluate.py                # Multi-Branch test evaluation
│   │   ├── train_behavioral.py        # Pure Behavioral training loop
│   │   └── evaluate_behavioral.py     # Pure Behavioral test evaluation
│   │
│   └── tools/                         # Auditing & Inspection Utilities
│       ├── audit_camera_drift.py      # Visual VFOA boundary inspection
│       └── create_affect_audit_view.py# Saved-track visual/CSV/provenance audit
│
├── run_behavioral.py                  # 🚀 Master CLI runner for Pure Behavioral Pipeline
├── run_multimodal.py                  # 🚀 Master CLI runner for Multi-Branch Pipeline
│
├── train.csv / val.csv / test.csv     # Dataset split annotations
├── requirements-affect.txt            # Affect-specific dependencies
├── confusion_matrix_behavioral.png    # Pure Behavioral test confusion matrix
├── .gitignore                         # Git exclusion rules for large datasets
└── README.md                          # Complete project documentation
```

---

## 🛡️ Affect Interpretation & Provenance

- **Anonymous Tracking**: Track IDs are anonymous and reset for each video; they do not identify students across videos.
- **Sparse Video Sampling**: Eight uniformly sampled frames are sparse for ByteTrack. Inspect audit views for ID fragmentation and compare against `--no-tracking` experimentally.
- **Uncertain Affect Modeling**: Facial expression probabilities are model estimates, not ground-truth internal emotions. They are treated as uncertain group-level behavioral evidence.
- **Strict Schema Enforcement**: Both matrix builders require valid extraction manifests (`extraction_manifest.json`) before building arrays.
- **Embedded Checkpoint Provenance**: Training checkpoints embed the complete feature schema and configuration. Evaluation verifies this schema before calculating metrics.
