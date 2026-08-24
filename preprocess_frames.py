import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Standard ImageNet normalization parameters for PyTorch / MobileNetV3
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

norm_transform = T.Compose([
    T.ToTensor(),  # Scales HWC uint8 [0, 255] -> CHW float [0.0, 1.0]
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

def extract_uniform_frames(video_path, num_frames=8):
    """
    Extracts exactly `num_frames` uniformly spaced frames from a video.
    Flips OpenCV BGR format to standard RGB format.
    Returns list of RGB numpy arrays with shape (H, W, 3).
    """
    if not os.path.exists(video_path):
        alt_path = os.path.join('cleanedDataset', 'Low', os.path.basename(video_path))
        if os.path.exists(alt_path):
            video_path = alt_path
        else:
            raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Could not read frame count for video: {video_path}")

    # Uniformly spaced frame indices
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames_rgb = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            # Retry adjacent frame if read fails
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx - 1))
            ret, frame_bgr = cap.read()
        
        if ret and frame_bgr is not None:
            # Color Fix: BGR -> RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames_rgb.append(frame_rgb)
        else:
            # Black fallback frame if video is corrupt
            frames_rgb.append(np.zeros((224, 224, 3), dtype=np.uint8))

    cap.release()
    return frames_rgb

def process_single_video(video_info, output_dir, num_frames=8):
    """
    Processes one video into dual paths:
    1. MobileNetV3 Path: 160x160 PyTorch Tensor (Channels, Height, Width), Float32, ImageNet Normalized -> Saved as .pt
    2. YOLOv5 Path: 640x640 NumPy Array (Height, Width, Channels), Uint8 RGB [0, 255], Unnormalized -> Saved as .npy / .npz
    """
    rel_video_path, label, split_name = video_info
    vname = os.path.splitext(os.path.basename(rel_video_path))[0]
    category = rel_video_path.split('/')[1] if '/' in rel_video_path else 'default'

    try:
        raw_rgb_frames = extract_uniform_frames(rel_video_path, num_frames=num_frames)

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
    parser.add_argument('--csv_files', nargs='+', default=['train.csv', 'val.csv', 'test.csv'],
                        help="List of CSV files to process.")
    parser.add_argument('--output_dir', default='preprocessed_data',
                        help="Root directory to save outputs.")
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

    success_count = 0
    failure_count = 0
    failures = []

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(process_single_video, task, args.output_dir, args.num_frames)
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

if __name__ == "__main__":
    main()
