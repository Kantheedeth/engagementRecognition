import os
import numpy as np
from tqdm import tqdm

def main():
    print("=" * 65)
    print("Building Behavioral Feature Matrices (40-dim: Interaction 32 + Affect 8)")
    print("  • Scene (MobileNetV3) is EXCLUDED (Zero Background Shortcut)")
    print("  • Output Directory: feature_matrices_behavioral")
    print("=" * 65)

    output_base = os.path.join("feature_matrices_behavioral")
    os.makedirs(output_base, exist_ok=True)

    csv_files = ["train.csv", "val.csv", "test.csv"]
    
    total_processed = 0
    total_missing = 0

    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            continue
            
        split_name = os.path.splitext(os.path.basename(csv_file))[0]
        dest_split_dir = os.path.join(output_base, split_name)
        os.makedirs(dest_split_dir, exist_ok=True)

        with open(csv_file, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]

        print(f"\nProcessing split '{split_name}' ({len(lines)} videos)...")

        for line in tqdm(lines, desc=f"Split {split_name}"):
            parts = line.split()
            if len(parts) < 2:
                continue
            video_path, label = parts[0], parts[1]
            
            vname = os.path.splitext(os.path.basename(video_path))[0]
            category = video_path.split('/')[1] if '/' in video_path else 'default'

            # Load 32-dim interaction features (8, 32)
            inter_path = os.path.join("preprocessed_features", "interaction_features", split_name, category, f"{vname}.npy")
            # Load 8-dim affect features (8, 8)
            affect_path = os.path.join("preprocessed_features", "affect_features", split_name, category, f"{vname}.npy")

            if not (os.path.exists(inter_path) and os.path.exists(affect_path)):
                total_missing += 1
                continue

            inter_feat = np.load(inter_path)   # (8, 32)
            affect_feat = np.load(affect_path) # (8, 8)

            # Concatenate to (8, 40): 32 Interaction + 8 Affect (Zero Scene)
            behavioral_matrix = np.concatenate([inter_feat, affect_feat], axis=1).astype(np.float32)

            dest_path = os.path.join(dest_split_dir, f"{vname}_label{label}.npy")
            np.save(dest_path, behavioral_matrix)
            total_processed += 1

    print("\n" + "=" * 65)
    print("Behavioral Feature Matrix Exporter Complete!")
    print(f"  Successfully built: {total_processed} matrices (Shape: 8x40)")
    if total_missing > 0:
        print(f"  Missing raw features: {total_missing} videos")
    print("=" * 65)

if __name__ == "__main__":
    main()
