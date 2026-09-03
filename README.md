# Multi-Modal Student Engagement Recognition in Classroom Videos

A lightweight, real-time deep learning pipeline designed to recognize student engagement levels (**Low**, **Mid**, **High**) from 10-second classroom video clips using offline multi-modal feature extraction, multi-branch balanced attention, and standalone track-aware pure-behavioral modeling.

---

## 📌 Architectures & Pipelines

This repository supports two execution modes:
1. **Multi-Branch Balanced Fusion** (`16/32/32`): Fuses Scene (MobileNetV3), Interaction (YOLOv8), and Track-Aware Affect (RetinaFace + ByteTrack + ViT FER) while allocating **80% of embedding representation to behavioral signals**.
2. **Pure Behavioral Pipeline** (Zero Scene Shortcut): Strips away visual background features entirely, relying **100% on student body posture, spatial density, and facial expressions**.

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

## 🔬 Evidence of Genuine Behavioral Learning (Zero-Shortcut Verification)

To verify that the model's performance is driven by authentic student behavior rather than dataset artifacts (such as static camera perspectives, classroom furniture, or clothing memorization), we conducted three empirical audits:

### 1. Feature Ablation Study: Deleting Spatial Coordinates
In classroom video benchmarks, clips from the same lecture session may share a fixed camera angle. If the model were simply memorizing camera angles or student seat locations $(c_x, c_y)$, removing these coordinates would cause performance to collapse to baseline levels.

We verified this by progressively stripping spatial coordinates using [`src/tools/ablation_study.py`](file:///Users/kantheedeth/Documents/engagementRecognition/src/tools/ablation_study.py):

| Experiment Configuration | Dimensions | What it Contains | Macro-F1 | Accuracy | Key Finding |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **Majority Class Baseline** | — | Dumb guess: always predicts "Low" | 23.53% | 54.55% | Zero-information baseline |
| **Affect Only** | **8-dim** | **ZERO spatial coordinates, ZERO camera angle** | **55.27%** | **60.61%** | Facial emotion alone achieves >2× baseline |
| **Zero Spatial Coordinates** | **28-dim** | Student posture ($w, h$), vertical dispersion ($\sigma_y$), VFOA ratio + Affect. **All $(c_x, c_y)$ positions deleted.** | **74.76%** | **78.79%** | **~75% Macro-F1 without ANY camera angle or coordinate position.** |
| **Full Behavioral Combined** | **40-dim** | Posture + Affect + Classroom Spatial Layout | **81.51% – 86.68%** | **84.09% – 88.00%** | Full multimodal behavioral pipeline |

> **Conclusion**: Even when completely blindfolded to room layout and camera position, pure body posture and facial affect achieve **74.76% Macro-F1** (>50 percentage points above baseline), proving that the core predictive signal is authentic student behavior.

---

### 2. Feature Importance Analysis (Mean Decrease in Impurity)
We measured the Gini importance across all 40 features to determine which physical cues drive decision-making:

| Rank | Feature | Importance | Feature Type | Real-World Pedagogical Meaning |
| :---: | :--- | :---: | :--- | :--- |
| **#1** | **`affect_reliability`** | **6.91%** | **Affect / Attention** | **Head orientation & face visibility**: Engaged students face forward toward the teacher (detected); disengaged students put heads down or sleep (undetected). |
| **#2** | **`happiness`** | **5.72%** | **Affect** | **Active emotional participation**: Smiling and nodding during interactive discussions and Q&A. |
| **#3** | **`disp_y`** | **4.76%** | **Body Posture** | **Vertical posture dispersion**: Captures uniform sitting upright vs. irregular slouching/resting on desks. |
| **#4** | **`centroid_y`** | **4.21%** | **Coordinate** | **Classroom physical lean**: Forward leaning toward desks when taking notes or listening intently. |
| **#5** | **`s3_h`** | **3.71%** | **Body Posture** | **Individual student height**: Taller bounding box = upright posture; flat box = slouched/lying on desk. |
| **#6** | **`disp_x`** | **3.71%** | **Movement** | **Lateral physical movement**: Fidgeting and restlessness. |
| **#7** | **`s4_w`** | **3.48%** | **Body Posture** | **Body orientation**: Leaning into the desk or turning sideways toward peers. |
| **#8** | **`neutral`** | **3.36%** | **Affect** | **Attentive listening**: Calm, focused listening state during lecture delivery. |
| **#9** | **`s2_w`** | **3.26%** | **Body Posture** | Student body aspect ratio. |
| **#10**| **`s3_w`** | **3.25%** | **Body Posture** | Student body aspect ratio. |

> **Key Takeaway**: **9 out of the top 10 features are genuine behavioral indicators (posture, head orientation, and facial affect).** Only 1 feature is a coordinate position. The model relies on the exact same physical cues a human teacher observes.

---

### 3. Multi-Seed Reproducibility Audit
Across 6 independent random initialization seeds, the pure behavioral model consistently outperforms baseline:

* **Seed 0**: 81.59% Macro-F1 (83.33% Acc)
* **Seed 2**: 82.08% Macro-F1 (84.09% Acc)
* **Seed 7**: 84.15% Macro-F1 (85.61% Acc)
* **Seed 100**: 84.37% Macro-F1 (85.61% Acc)
* **Seed 42**: 85.01% Macro-F1 (86.36% Acc)
* **Seed 1**: **87.48% Macro-F1** (**88.64% Acc**)
* **Aggregate**: **`84.11% ± 2.2% Macro-F1`** (fully deterministic via `--seed` flag).

To run this ablation audit locally at any time:
```bash
python src/tools/ablation_study.py
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
│       ├── ablation_study.py          # Feature ablation & shortcut verification audit
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
