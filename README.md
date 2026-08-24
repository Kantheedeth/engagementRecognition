# Multi-Modal Student Engagement Recognition in Classroom Videos

A lightweight, real-time deep learning pipeline designed to recognize student engagement levels (**Low**, **Mid**, **High**) from 10-second classroom video clips using multi-modal feature extraction, multi-branch balanced attention, and standalone pure-behavioral modeling.

---

## 📌 Architectures & Pipelines

This repository supports two execution modes:
1. **Multi-Branch Balanced Fusion** (`16/32/32`): Fuses Scene (MobileNetV3), Interaction (YOLOv8), and Affect (HSEmotion/YuNet) while allocating **80% of embedding representation to behavioral signals**.
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
│ MobileNetV3      │               │ YOLOv8 32-dim Interaction     │  │ OpenCV YuNet + HSEmotion      │
│ Small (576-dim)  │               │ Geometry & Density (32-dim)   │  │ Emotion Probabilities (8-dim) │
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
 HSEmotion Affect (8-dim)  ──► Linear(8,  48) ──┘
 (Zero Scene Features / Zero Background Memorization)
```

---

## 📊 Benchmark & Evaluation Results

Evaluated on the unseen test split (132 video clips):

### 1. Classification Metrics Summary

| Evaluation Setup | Modalities Used | Fused Dimensions | Test Accuracy | Macro-F1 Score |
| :--- | :--- | :---: | :---: | :---: |
| **Multi-Branch Balanced Fusion** | Scene (16) + Interaction (32) + Affect (32) | **80-dim** | **100.00%** | **100.00%** |
| **Pure Behavioral (Zero Scene)** | Interaction (48) + Affect (48) | **96-dim** | **73.00%** | **71.65%** |
| **Majority Class Baseline** | Always predicts Low | — | 54.54% | **23.53%** |

### 2. Detailed Breakdown: Pure Behavioral Model (No Scene Shortcut)

```text
              precision    recall  f1-score   support

         Low       0.81      0.72      0.76        72
         Mid       0.66      0.84      0.74        25
        High       0.64      0.66      0.65        35

    accuracy                           0.73       132
   macro avg       0.70      0.74      0.72       132
weighted avg       0.74      0.73      0.73       132

Model Macro-F1 Score:     71.65%
Baseline (Always Low) F1:  23.53%
```

---

## 🛠️ Requirements & Installation

Recommended Python version: `3.8+` (Tested on Apple Silicon M-Series & Linux).

```bash
# 1. Clone repository
git clone https://github.com/Kantheedeth/engagementRecognition.git
cd engagementRecognition

# 2. Create and activate conda environment
conda create -n slowfast python=3.8 -y
conda activate slowfast

# 3. Install dependencies
pip install torch torchvision opencv-python numpy tqdm scikit-learn matplotlib ultralytics emotiefflib timm onnx onnxruntime
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

### Phase 2: Feature Extraction
Extract numerical vectors offline from the three modules:
```bash
# 1. Scene features (MobileNetV3-Small -> 576-dim)
python extract_scene_features.py

# 2. Rich interaction features (YOLOv8 layout + student coordinates -> 32-dim)
python extract_interaction_features.py

# 3. Affect emotion features (YuNet + HSEmotion ONNX -> 8-dim)
python extract_affect_features.py
```

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
python audit_camera_drift.py
```
*(Annotated audit images are saved to `audit_outputs/`)*.

---

## 📁 Repository Structure

```text
engagementRecognition/
├── audit_outputs/                 # Annotated frames from camera drift audit
├── checkpoints/                   # Trained model weights (model_weights.pth, best_model_behavioral.pth)
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
├── extract_affect_features.py     # Phase 2C: YuNet + HSEmotion affect extraction (8-dim)
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
