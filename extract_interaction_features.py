import os
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# VFOA "Instructional Zone" geometric boundary configs
# Instruction zone is around 27% left side of the video
ZONE_X_MIN = 0.0
ZONE_X_MAX = 0.27
ZONE_Y_MIN = 0.0
ZONE_Y_MAX = 1.0

def compute_vfoa_ratio(boxes):
    """
    Given YOLO detection boxes, filters for class 0 (person),
    checks how many fall within the instructional zone, and returns the ratio.
    """
    person_count = 0
    in_zone_count = 0
    
    for box in boxes:
        cls_id = int(box.cls[0])
        if cls_id != 0:  # Skip classes other than person (class 0)
            continue
            
        person_count += 1
        
        # Get coordinates [x1, y1, x2, y2]
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = xyxy
        
        # Bounding box center normalized
        cx = ((x1 + x2) / 2.0) / 640.0
        cy = ((y1 + y2) / 2.0) / 640.0
        
        if ZONE_X_MIN <= cx <= ZONE_X_MAX and ZONE_Y_MIN <= cy <= ZONE_Y_MAX:
            in_zone_count += 1
            
    if person_count == 0:
        return 0.0  # Fallback baseline when no students/people are detected
        
    return in_zone_count / person_count

def main():
    print("=" * 60)
    print("Extracting Interaction Features (YOLO VFOA Ratio)...")
    print(f"Instruction Zone Bounds: X ∈ [{ZONE_X_MIN}, {ZONE_X_MAX}]")
    print("=" * 60)

    # Output directory
    output_base = os.path.join("preprocessed_features", "interaction_features")
    os.makedirs(output_base, exist_ok=True)

    # Load YOLOv8n model
    model = YOLO("yolov8n.pt")

    splits = ["train", "val", "test"]
    categories = ["low", "mid", "high"]

    # Collect all tasks
    tasks = []
    for split in splits:
        for cat in categories:
            src_dir = os.path.join("preprocessed_data", "yolov5_640x640", split, cat)
            if os.path.exists(src_dir):
                files = [f for f in os.listdir(src_dir) if f.endswith(".npz")]
                for f in files:
                    tasks.append((split, cat, f))

    print(f"Found {len(tasks)} videos to process.")

    # Process all files
    for split, cat, fname in tqdm(tasks, desc="Extracting Interaction Features"):
        vname = os.path.splitext(fname)[0]
        
        # Source path
        src_path = os.path.join("preprocessed_data", "yolov5_640x640", split, cat, fname)
        
        # Destination path
        dest_dir = os.path.join(output_base, split, cat)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, f"{vname}.npy")
        
        # Skip if already exists
        if os.path.exists(dest_path):
            continue

        # Load the preprocessed uint8 frames: (8, 640, 640, 3)
        data = np.load(src_path)
        frames = data['frames']
        
        ratios = []
        for frame in frames:
            results = model.predict(source=frame, conf=0.25, device='mps', verbose=False)
            boxes = results[0].boxes
            ratios.append(compute_vfoa_ratio(boxes))
            
        ratios_arr = np.array(ratios, dtype=np.float32).reshape(8, 1)
        np.save(dest_path, ratios_arr)

    print("Interaction feature extraction complete.")
    print("=" * 60)

if __name__ == "__main__":
    main()
