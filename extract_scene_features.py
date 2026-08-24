import os
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from tqdm import tqdm

def main():
    print("=" * 60)
    print("Extracting Scene Features using MobileNetV3-Small...")
    print("=" * 60)

    # Output directory
    output_base = os.path.join("preprocessed_features", "scene_features")
    os.makedirs(output_base, exist_ok=True)

    # Device selection (use MPS if available for M4 hardware)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load MobileNetV3-Small
    # Using modern weights API
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    
    # Remove classifier to get the 576-dim features
    model.classifier = nn.Identity()
    model = model.to(device)
    model.eval()

    splits = ["train", "val", "test"]
    categories = ["low", "mid", "high"]

    # Collect all tasks
    tasks = []
    for split in splits:
        for cat in categories:
            src_dir = os.path.join("preprocessed_data", "mobilenetv3_160x160", split, cat)
            if os.path.exists(src_dir):
                files = [f for f in os.listdir(src_dir) if f.endswith(".pt")]
                for f in files:
                    tasks.append((split, cat, f))

    print(f"Found {len(tasks)} videos to process.")

    # Process all files
    with torch.no_grad():
        for split, cat, fname in tqdm(tasks, desc="Extracting Scene Features"):
            vname = os.path.splitext(fname)[0]
            
            # Source path
            src_path = os.path.join("preprocessed_data", "mobilenetv3_160x160", split, cat, fname)
            
            # Destination path
            dest_dir = os.path.join(output_base, split, cat)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, f"{vname}.npy")
            
            # Skip if already exists
            if os.path.exists(dest_path):
                continue

            # Load the preprocessed normalized frames: (8, 3, 160, 160)
            data = torch.load(src_path, weights_only=False)
            frames = data['frames'].to(device)
            
            # Forward pass -> (8, 576)
            features = model(frames)
            
            # Convert to numpy and save
            features_np = features.cpu().numpy()
            np.save(dest_path, features_np)

    print("Scene feature extraction complete.")
    print("=" * 60)

if __name__ == "__main__":
    main()
