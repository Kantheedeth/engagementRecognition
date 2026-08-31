import os
import random
import numpy as np
import torch
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# ImageNet normalization stats
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def denormalize(tensor):
    """Converts a normalized (3, H, W) float32 tensor back to a (H, W, 3) uint8 RGB image."""
    # CHW to HWC
    arr = tensor.cpu().numpy().transpose(1, 2, 0)
    # De-normalize
    arr = arr * IMAGENET_STD + IMAGENET_MEAN
    # Clip and scale
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    return arr

def create_grid(frames, cols=4):
    """Combines a list of 8 frames (H, W, 3) into a single grid image."""
    h, w, c = frames[0].shape
    rows = (len(frames) + cols - 1) // cols
    grid = np.zeros((rows * h, cols * w, c), dtype=np.uint8)
    
    for idx, frame in enumerate(frames):
        r = idx // cols
        c_idx = idx % cols
        grid[r*h:(r+1)*h, c_idx*w:(c_idx+1)*w] = frame
    return grid

def main():
    print("=" * 60)
    print("Running Preprocessing Validation Gate...")
    print("=" * 60)
    
    output_dir = os.path.join(PROJECT_ROOT, "debug_validation")
    os.makedirs(output_dir, exist_ok=True)
    
    splits = ["train", "val", "test"]
    categories = ["low", "mid", "high"]
    
    # Select random samples
    selected_samples = []
    
    # We search the directories to find one sample from each category
    for cat in categories:
        samples_found = []
        for split in splits:
            path = os.path.join(PROJECT_ROOT, "preprocessed_data", "mobilenetv3_160x160", split, cat)
            if os.path.exists(path):
                files = [f for f in os.listdir(path) if f.endswith(".pt")]
                for f in files:
                    samples_found.append((split, cat, f))
        
        if samples_found:
            selected_samples.append(random.choice(samples_found))
    
    if not selected_samples:
        print("Error: No preprocessed files found!")
        return
        
    print(f"Selected {len(selected_samples)} samples for visual verification:")
    for split, cat, fname in selected_samples:
        print(f"  • Category: {cat.upper()} | Split: {split} | File: {fname}")
        
        vname = os.path.splitext(fname)[0]
        
        # Load MobileNetV3 .pt file
        pt_path = os.path.join(PROJECT_ROOT, "preprocessed_data", "mobilenetv3_160x160", split, cat, fname)
        pt_data = torch.load(pt_path, weights_only=False)
        mb_tensor = pt_data['frames']  # shape: (8, 3, 160, 160)
        
        # Load YOLOv5/v8 .npz file
        npz_path = os.path.join(PROJECT_ROOT, "preprocessed_data", "yolov5_640x640", split, cat, f"{vname}.npz")
        npz_data = np.load(npz_path)
        yolo_arr = npz_data['frames']  # shape: (8, 640, 640, 3)
        
        # 1. Process MobileNetV3 frames
        mb_rgb_frames = []
        for t in mb_tensor:
            mb_rgb_frames.append(denormalize(t))
            
        mb_grid = create_grid(mb_rgb_frames, cols=4)
        mb_grid_bgr = cv2.cvtColor(mb_grid, cv2.COLOR_RGB2BGR)
        mb_out_path = os.path.join(output_dir, f"mobilenet_{vname}_{cat}.jpg")
        cv2.imwrite(mb_out_path, mb_grid_bgr)
        
        # 2. Process YOLO frames
        yolo_rgb_frames = [f for f in yolo_arr]
        yolo_grid = create_grid(yolo_rgb_frames, cols=4)
        yolo_grid_bgr = cv2.cvtColor(yolo_grid, cv2.COLOR_RGB2BGR)
        yolo_out_path = os.path.join(output_dir, f"yolo_{vname}_{cat}.jpg")
        cv2.imwrite(yolo_out_path, yolo_grid_bgr)
        
        print(f"    -> Saved MobileNetV3 visual grid to {mb_out_path}")
        print(f"    -> Saved YOLO visual grid to {yolo_out_path}")
        print(f"    -> MobileNetV3 Tensor range: min={mb_tensor.min():.4f}, max={mb_tensor.max():.4f}")
        print(f"    -> YOLO Array range: min={yolo_arr.min()}, max={yolo_arr.max()}")
        print("-" * 60)
        
    print("Visual verification grids created successfully inside debug_validation/ directory.")
    print("=" * 60)

if __name__ == "__main__":
    main()
