import os
import random
import numpy as np
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# Instruction zone boundaries (left-most 27%)
ZONE_X_MIN = 0.0
ZONE_X_MAX = 0.27
ZONE_Y_MIN = 0.0
ZONE_Y_MAX = 1.0

def main():
    print("=" * 60)
    print("Running Camera Drift Audit...")
    print("=" * 60)

    output_dir = os.path.join(PROJECT_ROOT, "audit_outputs")
    os.makedirs(output_dir, exist_ok=True)

    splits = ["train", "val", "test"]
    categories = ["low", "mid", "high"]

    # Gather all .npz paths
    npz_paths = []
    for split in splits:
        for cat in categories:
            dir_path = os.path.join(PROJECT_ROOT, "preprocessed_data", "yolov5_640x640", split, cat)
            if os.path.exists(dir_path):
                files = [f for f in os.listdir(dir_path) if f.endswith(".npz")]
                for f in files:
                    npz_paths.append((split, cat, f))

    if not npz_paths:
        print("Error: No preprocessed .npz files found!")
        return

    # Sample 10 random videos to audit
    sample_size = min(10, len(npz_paths))
    samples = random.sample(npz_paths, sample_size)
    print(f"Auditing {sample_size} random videos...")

    for split, cat, fname in samples:
        vname = os.path.splitext(fname)[0]
        path = os.path.join(PROJECT_ROOT, "preprocessed_data", "yolov5_640x640", split, cat, fname)
        
        data = np.load(path)
        frames = data['frames'] # (8, 640, 640, 3)
        
        # Take the middle frame (frame index 4) for visualization
        frame_idx = 4
        frame_rgb = frames[frame_idx].copy()
        
        # Convert RGB to BGR for OpenCV saving
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Calculate visual pixel coordinates for the instruction zone
        px_min = int(ZONE_X_MIN * 640)
        px_max = int(ZONE_X_MAX * 640)
        py_min = int(ZONE_Y_MIN * 640)
        py_max = int(ZONE_Y_MAX * 640)
        
        # Draw the zone box (semi-transparent overlay or red outline)
        # We will draw a semi-transparent green box for the instruction zone
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (px_min, py_min), (px_max, py_max), (0, 255, 0), -1)
        # Apply the overlay (alpha = 0.3)
        cv2.addWeighted(overlay, 0.3, frame_bgr, 0.7, 0, frame_bgr)
        
        # Draw outline
        cv2.rectangle(frame_bgr, (px_min, py_min), (px_max, py_max), (0, 255, 0), 2)
        
        # Label the zone
        cv2.putText(
            frame_bgr, 
            "Instruction Zone (Left 27%)", 
            (px_min + 10, py_min + 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 255, 0), 
            2
        )
        
        # Save output image
        out_filename = f"audit_{vname}_{split}_{cat}.jpg"
        out_path = os.path.join(output_dir, out_filename)
        cv2.imwrite(out_path, frame_bgr)
        print(f"  • Saved annotated frame to {out_path}")

    print("\nCamera Drift Audit complete. Inspect audit_outputs/ to verify camera positioning.")
    print("=" * 60)

if __name__ == "__main__":
    main()
