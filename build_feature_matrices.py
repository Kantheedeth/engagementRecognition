import os
import numpy as np
from tqdm import tqdm

def main():
    print("=" * 60)
    print("Building Combined Feature Matrices (616-dim: Scene 576 + Interaction 32 + Affect 8)...")
    print("=" * 60)

    # Output base directory
    output_base = os.path.join("feature_matrices")
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

        print(f"Processing split '{split_name}' ({len(lines)} videos)...")

        for line in tqdm(lines, desc=f"Split {split_name}"):
            parts = line.split()
            if len(parts) < 2:
                continue
            video_path, label = parts[0], parts[1]
            
            vname = os.path.splitext(os.path.basename(video_path))[0]
            category = video_path.split('/')[1] if '/' in video_path else 'default'

            # Load scene features (8, 576)
            scene_path = os.path.join("preprocessed_features", "scene_features", split_name, category, f"{vname}.npy")
            # Load 32-dim interaction features (8, 32)
            inter_path = os.path.join("preprocessed_features", "interaction_features", split_name, category, f"{vname}.npy")
            # Load affect features (8, 8)
            affect_path = os.path.join("preprocessed_features", "affect_features", split_name, category, f"{vname}.npy")

            if not (os.path.exists(scene_path) and os.path.exists(inter_path) and os.path.exists(affect_path)):
                total_missing += 1
                continue

            scene_feat = np.load(scene_path)
            inter_feat = np.load(inter_path)
            affect_feat = np.load(affect_path)

            # Concatenate to (8, 616): 576 Scene + 32 Interaction + 8 Affect
            combined_matrix = np.concatenate([scene_feat, inter_feat, affect_feat], axis=1).astype(np.float32)

            # Destination filename format: feature_matrices/<split>/<video_name>_label<label>.npy
            dest_path = os.path.join(dest_split_dir, f"{vname}_label{label}.npy")
            np.save(dest_path, combined_matrix)
            total_processed += 1

    print("\n" + "=" * 60)
    print("Feature Matrix Exporter Complete!")
    print(f"  Successfully built: {total_processed} matrices (Shape: 8x616)")
    if total_missing > 0:
        print(f"  Missing raw features: {total_missing} videos")
    print("=" * 60)

if __name__ == "__main__":
    main()
