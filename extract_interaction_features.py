import os
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# VFOA "Instructional Zone" geometric boundary configs (left-most 27%)
ZONE_X_MIN = 0.0
ZONE_X_MAX = 0.27
ZONE_Y_MIN = 0.0
ZONE_Y_MAX = 1.0

MAX_STUDENTS_TRACKED = 5

def extract_frame_interaction_descriptor(boxes):
    """
    Extracts a rich 32-dimensional interaction descriptor from YOLO person detections:
      1. Global Classroom Geometry (7 dims):
         - Total person count (normalized / 10.0)
         - In-zone person count (normalized / 10.0)
         - VFOA facing ratio (in_zone / total)
         - Student centroid mean (cx_mean, cy_mean in [0, 1])
         - Student spatial spread/dispersion (std_x, std_y in [0, 1])
      2. Top-5 Student States (5 students x 5 dims = 25 dims):
         - For each student (sorted by bbox area): [cx, cy, w, h, in_zone_flag]
         - Zero-padded if fewer than 5 students detected.
      Total: 7 + 25 = 32 dimensions.
    """
    persons = []
    
    for box in boxes:
        cls_id = int(box.cls[0])
        if cls_id != 0:  # Class 0 is person
            continue
            
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = xyxy
        
        cx = ((x1 + x2) / 2.0) / 640.0
        cy = ((y1 + y2) / 2.0) / 640.0
        w = (x2 - x1) / 640.0
        h = (y2 - y1) / 640.0
        area = w * h
        
        in_zone = 1.0 if (ZONE_X_MIN <= cx <= ZONE_X_MAX and ZONE_Y_MIN <= cy <= ZONE_Y_MAX) else 0.0
        persons.append((area, cx, cy, w, h, in_zone))
        
    num_persons = len(persons)
    
    if num_persons == 0:
        # Fallback baseline when no people detected
        return np.zeros(32, dtype=np.float32)
        
    in_zone_count = sum(p[5] for p in persons)
    vfoa_ratio = in_zone_count / num_persons
    
    cxs = [p[1] for p in persons]
    cys = [p[2] for p in persons]
    
    cx_mean = float(np.mean(cxs))
    cy_mean = float(np.mean(cys))
    std_x = float(np.std(cxs)) if num_persons > 1 else 0.0
    std_y = float(np.std(cys)) if num_persons > 1 else 0.0
    
    # 7 Global geometry features
    global_features = [
        num_persons / 10.0,
        in_zone_count / 10.0,
        vfoa_ratio,
        cx_mean,
        cy_mean,
        std_x,
        std_y
    ]
    
    # Sort detected students by area (largest / closest first)
    persons.sort(key=lambda p: p[0], reverse=True)
    
    # 25 Top-5 Student states (5 * 5 = 25)
    student_features = []
    for i in range(MAX_STUDENTS_TRACKED):
        if i < num_persons:
            _, cx, cy, w, h, in_zone = persons[i]
            student_features.extend([cx, cy, w, h, in_zone])
        else:
            student_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            
    descriptor = np.array(global_features + student_features, dtype=np.float32)
    return descriptor

def main():
    print("=" * 60)
    print("Extracting Rich 32-dim Interaction Descriptors (YOLOv8)...")
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
    for split, cat, fname in tqdm(tasks, desc="Extracting 32-dim Interaction Features"):
        vname = os.path.splitext(fname)[0]
        
        # Source path
        src_path = os.path.join("preprocessed_data", "yolov5_640x640", split, cat, fname)
        
        # Destination path
        dest_dir = os.path.join(output_base, split, cat)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, f"{vname}.npy")

        # Load the preprocessed uint8 frames: (8, 640, 640, 3)
        data = np.load(src_path)
        frames = data['frames']
        
        # Batch inference on all 8 frames
        results = model.predict(source=list(frames), conf=0.25, device='mps', verbose=False)
        
        descriptors = [extract_frame_interaction_descriptor(res.boxes) for res in results]
        descriptors_arr = np.array(descriptors, dtype=np.float32)  # Shape: (8, 32)
        np.save(dest_path, descriptors_arr)

    print("32-dim Interaction feature extraction complete.")
    print("=" * 60)

if __name__ == "__main__":
    main()
