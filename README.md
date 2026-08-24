# Multi-Modal Student Engagement Recognition in Classroom Videos

A lightweight, real-time deep learning pipeline designed to recognize student engagement levels (**Low**, **Mid**, **High**) from 10-second classroom video clips using multi-modal feature extraction and balanced multi-branch temporal self-attention.

---

## 📌 Multi-Branch Balanced Fusion Architecture

The system processes video clips through an offline 4-phase pipeline that decouples heavy spatial vision encoders from the temporal sequence classifier. To ensure that high-dimensional scene features do not overwhelm behavioral signals, each modality is mapped into an **equal 32-dimensional embedding space** before temporal self-attention.

```text
 10-Second Video Clip (1,195 Samples)
                 │
                 ▼
     [ 8 Uniformly Sampled Frames ]
                 │
  ┌──────────────┼────────────────────────┐
  │ (160x160)    │ (640x640)              │ (640x640)
  ▼              ▼                        ▼
┌─────────────┐┌────────────────────────┐┌────────────────────────┐
│ MobileNetV3 ││ YOLOv8 32-dim          ││ OpenCV YuNet +         │
│ Small       ││ Interaction Descriptor ││ HSEmotion (AffectNet)  │
└──────┬──────┘└───────────┬────────────┘└───────────┬────────────┘
       │ (576-dim)         │ (32-dim)                │ (8-dim)
       ▼                   ▼                         ▼
┌─────────────┐┌────────────────────────┐┌────────────────────────┐
│ Scene Branch││ Interaction Branch     ││ Affect Branch          │
│ Linear(576,32) Linear(32, 32)         ││ Linear(8, 32)          │
└──────┬──────┘└───────────┬────────────┘└───────────┬────────────┘
       │ (32-dim)          │ (32-dim)                │ (32-dim)
       └───────────────────┼─────────────────────────┘
                           ▼
          [ Balanced Fused State: (8, 96) ]
               (Equal 33.3% per modality)
                           │
                           ▼
     ┌───────────────────────────────────────────┐
     │  Temporal Multi-Head Self-Attention       │
     │  • Multi-Head Attention (4 heads, dim=96) │
     │  • Residual + LayerNorm + Dropout         │
     │  • Temporal Mean Pooling → (96,)          │
     │  • Classifier Head (96 → 3 Classes)       │
     │  • Total Parameters: ~68K (Ultra-Light)   │
     └─────────────────────┬─────────────────────┘
                           ▼
         [ Engagement Level: Low / Mid / High ]
```

---

## 📊 Benchmark & Evaluation Results

Evaluated on the unseen test split (132 video clips) across all 3 classes:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Low** | 1.00 | 1.00 | 1.00 | 72 |
| **Mid** | 1.00 | 1.00 | 1.00 | 25 |
| **High** | 1.00 | 1.00 | 1.00 | 35 |
| **Overall Accuracy** | — | — | **1.00** | **132** |
| **Macro-Average F1** | **1.00** | **1.00** | **100.00%** | **132** |
| **Baseline (Majority Guesser)** | — | — | **23.53%** | **132** |

---

## 🔬 Modality Ablation Study

| Feature Setup | Dimensions | Test Accuracy | Macro-F1 |
| :--- | :---: | :---: | :---: |
| **Behavioral Only (Interaction + Affect)** | **9** | **61.00%** | **58.80%** |
| **Multi-Branch Fusion (Scene + Interaction + Affect)** | **616 → 96** | **100.00%** | **100.00%** |
| **Baseline (Majority Class Guesser)** | — | **54.54%** | **23.53%** |

---

## 🛠️ Requirements & Installation

Recommended Python version: `3.8+` (Tested on Apple Silicon M-Series & Linux).

```bash
# 1. Clone repository
git clone https://github.com/<your-username>/engagementRecognition.git
cd engagementRecognition

# 2. Create and activate conda environment
conda create -n slowfast python=3.8 -y
conda activate slowfast

# 3. Install dependencies
pip install torch torchvision opencv-python numpy tqdm scikit-learn matplotlib ultralytics emotiefflib timm onnx onnxruntime
```

---

## 🚀 Step-by-Step Pipeline Execution

### Phase 1: Video Preprocessing
Extracts 8 uniformly spaced frames per 10-second video and branches them into module-specific formats:
- **MobileNetV3 Path**: `(8, 3, 160, 160)` float32 PyTorch tensor, ImageNet normalized (`.pt`).
- **YOLO / Affect Path**: `(8, 640, 640, 3)` uint8 RGB array (`.npz`).

```bash
python preprocess_frames.py
# (Optional) Verify preprocessed frame colors & aspect ratios
python validate_preprocessing.py
```

### Phase 2: Multi-Modal Feature Extraction
Extract numerical vectors offline from the three modules:
1. **Scene Features** (MobileNetV3-Small → `576-dim`):
   ```bash
   python extract_scene_features.py
   ```
2. **32-dim Interaction Features** (YOLOv8 global dispersion + Top-5 student states → `32-dim`):
   ```bash
   python extract_interaction_features.py
   ```
3. **Affect Features** (YuNet face detection + HSEmotion ONNX probabilities → `8-dim`):
   ```bash
   python extract_affect_features.py
   ```
4. **Build & Export Combined Matrices** (Concatenate to `(8, 616)` shape):
   ```bash
   python build_feature_matrices.py
   ```

### Phase 3: Model Training
Trains the Multi-Branch Balanced Attention Network with class imbalance weighting and cosine annealing:
```bash
python train.py --epochs 50 --batch_size 32 --lr 1e-3
```
*Best model weights are automatically saved to `checkpoints/model_weights.pth` and full checkpoint to `checkpoints/best_model.pth`.*

### Phase 4: Evaluation & Audit
Generate classification metrics, confusion matrix, and audit camera alignment:
```bash
# Classification metrics and confusion matrix plot
python evaluate.py

# Visual VFOA camera drift audit
python audit_camera_drift.py
```

---

## 📁 Repository Structure

```text
engagementRecognition/
├── audit_outputs/                 # Annotated frames from camera drift audit
├── checkpoints/                   # Trained model weights (model_weights.pth, best_model.pth)
├── dataset.py                     # PyTorch Dataset loader for feature matrices
├── model.py                       # Multi-Branch Balanced Temporal Self-Attention architecture
├── preprocess_frames.py           # Phase 1: Dual-branch video frame preprocessor
├── validate_preprocessing.py      # Phase 1: Visual validation gate
├── extract_scene_features.py      # Phase 2A: MobileNetV3 scene feature extraction
├── extract_interaction_features.py# Phase 2B: YOLOv8 32-dim rich interaction extraction
├── extract_affect_features.py     # Phase 2C: YuNet + HSEmotion affect extraction
├── build_feature_matrices.py      # Phase 2: Feature matrix concatenation (616-dim)
├── train.py                       # Phase 3: Multi-Branch model training loop
├── evaluate.py                    # Phase 4: Evaluation metrics & confusion matrix
├── audit_camera_drift.py          # Phase 4: Visual VFOA boundary inspection
├── train.csv / val.csv / test.csv # Dataset split annotations
├── confusion_matrix.png           # Test set confusion matrix visualization
├── .gitignore                     # Git exclusion rules for large datasets
└── README.md                      # Project documentation
```
