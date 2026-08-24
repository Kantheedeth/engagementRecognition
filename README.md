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

## 📊 Benchmark & Evaluation Results

Evaluated on the unseen test split (132 video clips):

### 1. Classification Metrics Summary

| Evaluation Setup | Modalities Used | Fused Dimensions | Test Accuracy | Macro-F1 Score |
| :--- | :--- | :---: | :---: | :---: |
| **Multi-Branch Balanced Fusion** | Scene (16) + Interaction (32) + Affect (32) | **80-dim** | **100.00%** | **100.00%** |
| **Pure Behavioral (Zero Scene)** | Interaction (48) + Affect (48) | **96-dim** | **88.00%** | **86.68%** |
| **Independent Random Forest Baseline** | Static Temporal Average (Pure Behavioral) | 40-dim | 82.58% | **79.32%** |
| **Majority Class Baseline** | Always predicts Low | — | 54.54% | **23.53%** |

### 2. Detailed Breakdown: Pure Behavioral Model (No Scene Shortcut)

```text
=================================================================
Running Pure Behavioral Pipeline Evaluation Phase...
  • Features: 32 Interaction + 8 Affect (ZERO Scene Features)
=================================================================
              precision    recall  f1-score   support

         Low       0.90      0.89      0.90        72
         Mid       0.83      0.80      0.82        25
        High       0.86      0.91      0.89        35

    accuracy                           0.88       132
   macro avg       0.87      0.87      0.87       132
weighted avg       0.88      0.88      0.88       132

Model Macro-F1 Score:     86.68%
Baseline (Always Low) F1:  23.53%
=================================================================
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

### Phase 1: Video Preprocessing
Extracts 8 uniformly spaced frames per 10-second video:
```bash
python preprocess_frames.py
# (Optional) Verify preprocessed frame colors & aspect ratios
python validate_preprocessing.py
```

Paths are resolved relative to the script and detected dataset root. Missing or undecodable videos fail preprocessing; they are never replaced with fabricated black frames or cross-class data.

### Phase 2: Feature Extraction
Extract numerical vectors offline from the three modules:
```bash
# 1. Scene features (MobileNetV3-Small -> 576-dim)
python extract_scene_features.py

# 2. Rich interaction features (YOLOv8 layout + student coordinates -> 32-dim)
python extract_interaction_features.py

# 3. Track-aware affect (RetinaFace + ByteTrack + PyTorch FER -> 8-dim)
python extract_affect_features.py --save_track_details
```

The affect tensor is `(8, 8)` with columns: `anger, disgust, fear, happiness, sadness, surprise, neutral, affect_reliability`. Outputs are written to `preprocessed_features/affect_track_features`. Missing faces produce zero evidence and zero reliability; there is no hard-coded fallback.

---

### Phase 3 & 4: Multi-Branch Balanced Pipeline (Default 16/32/32)

```bash
# 1. Build 616-dim feature matrices (Scene 576 + Interaction 32 + Affect 8)
python build_feature_matrices.py

# 2. Train Multi-Branch Attention Model (16 Scene / 32 Interaction / 32 Affect)
python train.py --scene_branch_dim 16 --inter_branch_dim 32 --affect_branch_dim 32

# 3. Evaluate and generate confusion matrix
python evaluate.py --scene_branch_dim 16 --inter_branch_dim 32 --affect_branch_dim 32
```

---

### Alternative: Standalone Pure Behavioral Pipeline (Zero Scene Shortcut)

```bash
# 1. Build 40-dim behavioral matrices (Interaction 32 + Affect 8)
python build_behavioral_matrices.py

# 2. Train Pure Behavioral Model (Zero background features)
python train_behavioral.py

# 3. Evaluate and generate confusion matrix
python evaluate_behavioral.py
```

---

### Audit: Camera Drift & Instruction Zone Inspection
Visualizes the VFOA boundary overlaid on actual classroom frames:
```bash
# Visual VFOA camera drift audit
python audit_camera_drift.py

# Visualize saved anonymous face tracks and affect estimates
python create_affect_audit_view.py \
  --track_json debug_validation/affect_tracks/train/low/view1000.json
```

The affect audit produces a contact sheet, annotated MP4, CSV table, and SHA-256 provenance summary without rerunning inference.

---

## 📁 Repository Structure

```text
engagementRecognition/
├── audit_outputs/                 # Annotated frames from camera drift audit
├── checkpoints/                   # Provenance-bearing best-model checkpoints
│
├── dataset.py                     # PyTorch Dataset loader for feature matrices
├── model.py                       # Multi-Branch Balanced Attention architecture (16/32/32)
├── model_behavioral.py            # Standalone Pure Behavioral Attention architecture
│
├── preprocess_frames.py           # Phase 1: Dual-branch video frame preprocessor
├── validate_preprocessing.py      # Phase 1: Visual validation gate
│
├── extract_scene_features.py      # Phase 2A: MobileNetV3 scene feature extraction (576-dim)
├── extract_interaction_features.py# Phase 2B: YOLOv8 rich interaction extraction (32-dim)
├── affect_module.py               # RetinaFace + ByteTrack + FER aggregation
├── extract_affect_features.py     # Phase 2C: track-aware affect extraction (8-dim)
├── create_affect_audit_view.py    # Saved-track visual/CSV/provenance audit
├── feature_schema.py              # Cross-stage feature contracts
├── requirements-affect.txt        # Affect-specific dependencies
│
├── build_feature_matrices.py      # Combines Scene + Interaction + Affect (616-dim)
├── build_behavioral_matrices.py   # Combines Interaction + Affect (40-dim, Zero Scene)
│
├── train.py                       # Trains Multi-Branch Balanced model
├── evaluate.py                    # Evaluates Multi-Branch model
│
├── train_behavioral.py            # Trains Pure Behavioral model
├── evaluate_behavioral.py         # Evaluates Pure Behavioral model
│
├── audit_camera_drift.py          # Visual VFOA boundary inspection
├── train.csv / val.csv / test.csv # Dataset split annotations
├── confusion_matrix.png           # Multi-Branch test confusion matrix
├── confusion_matrix_behavioral.png# Pure Behavioral test confusion matrix
├── .gitignore                     # Git exclusion rules for large datasets
└── README.md                      # Complete project documentation
```

---

## 🛡️ Affect Interpretation & Provenance

- **Anonymous Tracking**: Track IDs are anonymous and reset for each video; they do not identify students across videos.
- **Sparse Video Sampling**: Eight uniformly sampled frames are sparse for ByteTrack. Inspect audit views for ID fragmentation and compare against `--no-tracking` experimentally.
- **Uncertain Affect Modeling**: Facial expression probabilities are model estimates, not ground-truth internal emotions. They are treated as uncertain group-level behavioral evidence.
- **Strict Schema Enforcement**: Both matrix builders require valid extraction manifests (`extraction_manifest.json`) before building arrays.
- **Embedded Checkpoint Provenance**: Training checkpoints embed the complete feature schema and configuration. Evaluation verifies this schema before calculating metrics.
