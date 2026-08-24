import os
import argparse
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Standard ImageNet normalization parameters for PyTorch / MobileNetV3
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
LABEL_BY_CATEGORY = {'low': 0, 'mid': 1, 'high': 2}

norm_transform = T.Compose([
    T.ToTensor(),  # Scales HWC uint8 [0, 255] -> CHW float [0.0, 1.0]
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


def get_default_video_root():
    """Find the dataset root containing videos/ near this script."""
    candidates = [SCRIPT_DIR, os.path.dirname(SCRIPT_DIR)]
    for candidate in candidates:
        if os.path.isdir(os.path.join(candidate, 'videos')):
            return candidate
    return SCRIPT_DIR

def resolve_video_path(video_path, video_root='.'):
    """Resolve a CSV video path without substituting data from another class."""
    candidate = os.path.join(video_root, video_path)
    if os.path.isfile(candidate):
        return candidate

    raise FileNotFoundError(
        f"Video file not found: {candidate}. "
        "Check --video_root and the path/capitalization in the CSV file."
    )


def extract_uniform_frames(video_path, num_frames=8):
    """
    Extracts exactly `num_frames` uniformly spaced frames from a video.
    Flips OpenCV BGR format to standard RGB format.
    Returns list of RGB numpy arrays with shape (H, W, 3).
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Could not read frame count for video: {video_path}")

    # Uniformly spaced target frame indices.
    # Decode sequentially instead of repeatedly seeking because random seeking
    # in H.264 videos can trigger decoder errors around damaged/non-key frames.
    target_indices = set(np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist())
    frames_by_index = {}
    frame_idx = 0

    while frame_idx < total_frames and len(frames_by_index) < len(target_indices):
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            cap.release()
            raise ValueError(
                f"H.264 decode failed at frame {frame_idx} in video: {video_path}. "
                "Re-encode or replace this video before preprocessing."
            )

        if frame_idx in target_indices:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames_by_index[frame_idx] = frame_rgb

        frame_idx += 1

    missing = sorted(target_indices.difference(frames_by_index.keys()))
    if missing:
        cap.release()
        raise ValueError(
            f"Could not decode required frames {missing} from video: {video_path}"
        )

    # Preserve the original uniform temporal order.
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames_rgb = [frames_by_index[int(idx)] for idx in indices]
    cap.release()
    return frames_rgb

def process_single_video(video_info, output_dir, num_frames=8, video_root='.'):
    """
    Processes one video into dual paths:
    1. MobileNetV3 Path: 160x160 PyTorch Tensor (Channels, Height, Width), Float32, ImageNet Normalized -> Saved as .pt
    2. YOLOv5 Path: 640x640 NumPy Array (Height, Width, Channels), Uint8 RGB [0, 255], Unnormalized -> Saved as .npy / .npz
    """
    rel_video_path, label, split_name = video_info
    vname = os.path.splitext(os.path.basename(rel_video_path))[0]
    category = rel_video_path.split('/')[1].lower() if '/' in rel_video_path else 'default'

    try:
        expected_label = LABEL_BY_CATEGORY.get(category)
        if expected_label is None:
            raise ValueError(f"Unknown engagement category '{category}' in {rel_video_path}")
        if int(label) != expected_label:
            raise ValueError(
                f"Label mismatch for {rel_video_path}: CSV label is {label}, "
                f"but category '{category}' requires {expected_label}"
            )

        video_path = resolve_video_path(rel_video_path, video_root=video_root)
        raw_rgb_frames = extract_uniform_frames(video_path, num_frames=num_frames)

        # -------------------------------------------------------------
        # 1. MobileNetV3 Path (160x160 PyTorch Tensor)
        # -------------------------------------------------------------
        mobilenet_tensors = []
        for frame_rgb in raw_rgb_frames:
            frame_160 = cv2.resize(frame_rgb, (160, 160), interpolation=cv2.INTER_LINEAR)
            # ToTensor converts (160, 160, 3) uint8 to (3, 160, 160) float32 in [0, 1]
            # Normalize applies ImageNet mean & std
            tensor_160 = norm_transform(frame_160) # Shape: (3, 160, 160)
            mobilenet_tensors.append(tensor_160)

        # Stack into (8, 3, 160, 160) tensor
        mobilenet_batch = torch.stack(mobilenet_tensors)

        # -------------------------------------------------------------
        # 2. YOLOv5 Path (640x640 NumPy Array)
        # -------------------------------------------------------------
        yolo_frames = []
        for frame_rgb in raw_rgb_frames:
            # Native HWC format (640, 640, 3), uint8 RGB
            frame_640 = cv2.resize(frame_rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
            yolo_frames.append(frame_640)

        # Stack into (8, 640, 640, 3) numpy array
        yolo_batch = np.array(yolo_frames, dtype=np.uint8)

        # -------------------------------------------------------------
        # Save Outputs
        # -------------------------------------------------------------
        mobilenet_dir = os.path.join(output_dir, 'mobilenetv3_160x160', split_name, category)
        yolo_dir = os.path.join(output_dir, 'yolov5_640x640', split_name, category)

        os.makedirs(mobilenet_dir, exist_ok=True)
        os.makedirs(yolo_dir, exist_ok=True)

        # Save MobileNetV3 PyTorch Tensor (.pt)
        torch.save({
            'frames': mobilenet_batch,       # Shape: (8, 3, 160, 160), float32, normalized
            'label': int(label),
            'video_name': vname,
            'category': category
        }, os.path.join(mobilenet_dir, f"{vname}.pt"))

        # Save YOLOv5 NumPy Array (.npz)
        np.savez_compressed(
            os.path.join(yolo_dir, f"{vname}.npz"),
            frames=yolo_batch,              # Shape: (8, 640, 640, 3), uint8, RGB [0, 255]
            label=int(label),
            video_name=vname,
            category=category
        )

        return True, vname
    except Exception as e:
        return False, f"{vname}: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Preprocess video dataset for MobileNetV3 (160x160 Tensor) and YOLOv5 (640x640 NumPy Array).")
    parser.add_argument('--csv_files', nargs='+', default=[
                            os.path.join(SCRIPT_DIR, 'train.csv'),
                            os.path.join(SCRIPT_DIR, 'val.csv'),
                            os.path.join(SCRIPT_DIR, 'test.csv'),
                        ],
                        help="List of CSV files to process.")
    parser.add_argument('--output_dir', default=os.path.join(SCRIPT_DIR, 'preprocessed_data'),
                        help="Root directory to save outputs.")
    parser.add_argument('--video_root', default=get_default_video_root(),
                        help="Directory relative to which CSV video paths are resolved.")
    parser.add_argument('--num_frames', type=int, default=8,
                        help="Number of uniformly spaced frames per video.")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="Number of parallel worker processes.")

    args = parser.parse_args()

    print("=" * 65)
    print("Dual-Model Video Preprocessing Pipeline")
    print(f"  • MobileNetV3 Path : 160x160 | PyTorch Tensor (C, H, W) | ImageNet Normalized | .pt")
    print(f"  • YOLOv5 Path     : 640x640 | NumPy Array (H, W, C)    | RGB uint8 [0, 255] | .npz")
    print(f"  • Frames / Video   : {args.num_frames}")
    print(f"  • Output Directory : {args.output_dir}")
    print("=" * 65)

    all_tasks = []
    for csv_file in args.csv_files:
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found. Skipping.")
            continue
        
        split_name = os.path.splitext(os.path.basename(csv_file))[0]
        with open(csv_file, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                video_path, label = parts[0], parts[1]
                all_tasks.append((video_path, label, split_name))

    print(f"Total videos to process: {len(all_tasks)}")
    if not all_tasks:
        raise FileNotFoundError(
            "No videos were scheduled. Check the CSV paths and current configuration."
        )

    success_count = 0
    failure_count = 0
    failures = []

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(
                process_single_video,
                task,
                args.output_dir,
                args.num_frames,
                args.video_root,
            )
            for task in all_tasks
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
            success, msg = future.result()
            if success:
                success_count += 1
            else:
                failure_count += 1
                failures.append(msg)

    print("\n" + "=" * 65)
    print("Preprocessing Complete!")
    print(f"  Successfully processed: {success_count}/{len(all_tasks)} videos")
    if failure_count > 0:
        print(f"  Failed: {failure_count} videos")
        for err in failures[:10]:
            print(f"    - {err}")
    print("=" * 65)
    if failure_count > 0:
        raise RuntimeError(
            f"Preprocessing failed for {failure_count} videos; no replacement "
            "frames or cross-class videos were fabricated."
        )

if __name__ == "__main__":
    main()
