#!/usr/bin/env python3
"""Visual Inspection Tool: Visualize YOLOv8-pose keypoints, student head/body orientation,
and the Instruction Zone attention vectors on classroom frames.

Usage:
    # Visualize a specific video
    python src/tools/visualize_interaction_orientation.py --video_path preprocessed_data/yolov5_640x640/test/high/view2228.npz

    # Automatically sample 1 High, 1 Mid, and 1 Low video and save full 8-frame montages
    python src/tools/visualize_interaction_orientation.py --sample
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.extract_interaction_features import (
    DEFAULT_TARGET,
    ZONE_X_MAX,
    ZONE_X_MIN,
    ZONE_Y_MAX,
    ZONE_Y_MIN,
    select_device,
)


def draw_interaction_frame(
    frame_rgb: np.ndarray,
    results,
    frame_idx: int,
    total_frames: int,
    video_name: str,
) -> np.ndarray:
    """Render bounding boxes, pose keypoints, head facing vectors, and instruction zone."""
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    # 1. Highlight Instruction Zone (Left 27%) with transparent tint
    overlay = img.copy()
    zone_x1 = int(ZONE_X_MIN * w)
    zone_x2 = int(ZONE_X_MAX * w)
    zone_y1 = int(ZONE_Y_MIN * h)
    zone_y2 = int(ZONE_Y_MAX * h)
    cv2.rectangle(overlay, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 200, 0), -1)
    alpha = 0.18
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.rectangle(img, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 200, 0), 2)
    cv2.putText(
        img,
        "Instruction Zone (Podium/Board)",
        (zone_x1 + 10, zone_y1 + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 220, 50),
        2,
        cv2.LINE_AA,
    )

    boxes = results.boxes
    if len(boxes) == 0:
        return img

    xyxy = boxes.xyxy.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy()
    kpts = results.keypoints.xy.detach().cpu().numpy() if results.keypoints is not None else None
    kpts_conf = (
        results.keypoints.conf.detach().cpu().numpy()
        if results.keypoints is not None and results.keypoints.conf is not None
        else None
    )

    # 2. Identify Teacher in the instruction zone
    teacher_pos = None
    teacher_box_idx = None
    for i, box in enumerate(xyxy):
        if int(classes[i]) != 0:
            continue
        cx = ((box[0] + box[2]) / 2.0) / float(w)
        cy = ((box[1] + box[3]) / 2.0) / float(h)
        if ZONE_X_MIN <= cx <= ZONE_X_MAX and ZONE_Y_MIN <= cy <= ZONE_Y_MAX:
            teacher_pos = np.array([cx * w, cy * h], dtype=np.float32)
            teacher_box_idx = i
            break

    target_px = teacher_pos if teacher_pos is not None else np.array([DEFAULT_TARGET[0] * w, DEFAULT_TARGET[1] * h])

    # Mark target center with crosshair
    tx, ty = int(target_px[0]), int(target_px[1])
    cv2.drawMarker(img, (tx, ty), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.putText(
        img,
        "Target Focus Point",
        (tx + 10, ty + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    student_attentions = []

    # 3. Draw each detected person
    for i, box in enumerate(xyxy):
        if int(classes[i]) != 0:
            continue
        x1, y1, x2, y2 = map(int, box[:4])
        cx_norm = ((x1 + x2) / 2.0) / float(w)
        cy_norm = ((y1 + y2) / 2.0) / float(h)
        cx_px = int((x1 + x2) / 2.0)
        cy_px = int((y1 + y2) / 2.0)

        # Is this person the teacher?
        if i == teacher_box_idx or cx_norm <= ZONE_X_MAX:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.putText(
                img,
                "Teacher / Instructor",
                (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 200, 0),
                2,
                cv2.LINE_AA,
            )
            continue

        # Target vector from student to instruction zone
        v_target = np.array([target_px[0] - cx_px, target_px[1] - cy_px], dtype=np.float32)
        dist_target = float(np.linalg.norm(v_target))
        u_target = v_target / dist_target if dist_target > 1e-4 else np.zeros(2, dtype=np.float32)

        # Facing vector from keypoints
        v_facing = None
        face_anchor_px = None
        zone_attention = 0.5

        if kpts is not None and i < len(kpts):
            p_kpts = kpts[i]
            p_conf = kpts_conf[i] if kpts_conf is not None else np.ones(17, dtype=np.float32)

            nose = p_kpts[0]
            l_eye, r_eye = p_kpts[1], p_kpts[2]
            l_sh, r_sh = p_kpts[5], p_kpts[6]

            # Draw keypoint dots
            for k_idx in [0, 1, 2, 5, 6]:
                if p_conf[k_idx] > 0.3:
                    kp_x, kp_y = int(p_kpts[k_idx, 0]), int(p_kpts[k_idx, 1])
                    color = (0, 255, 255) if k_idx == 0 else (0, 165, 255)
                    cv2.circle(img, (kp_x, kp_y), 4, color, -1)

            # Method 1: Head vector (mid-eyes to nose)
            if p_conf[0] > 0.3 and (p_conf[1] > 0.3 or p_conf[2] > 0.3):
                eye_mid = (
                    (l_eye + r_eye) / 2.0
                    if (p_conf[1] > 0.3 and p_conf[2] > 0.3)
                    else (l_eye if p_conf[1] > 0.3 else r_eye)
                )
                v_head = nose - eye_mid
                norm_h = float(np.linalg.norm(v_head))
                if norm_h > 1e-4:
                    v_facing = v_head / norm_h
                    face_anchor_px = nose

            # Method 2: Torso vector (orthogonal to shoulders)
            if v_facing is None and p_conf[5] > 0.3 and p_conf[6] > 0.3:
                v_sh = r_sh - l_sh
                v_torso = np.array([-v_sh[1], v_sh[0]], dtype=np.float32)
                norm_t = float(np.linalg.norm(v_torso))
                if norm_t > 1e-4:
                    v_facing = v_torso / norm_t
                    face_anchor_px = (l_sh + r_sh) / 2.0

        if v_facing is not None and dist_target > 1e-4:
            cos_sim = float(np.dot(v_facing, u_target))
            zone_attention = float(np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0))
        else:
            zone_attention = 0.5

        student_attentions.append(zone_attention)

        # Color coding: Green = Attentive, Yellow = Neutral, Red = Looking Away
        if zone_attention >= 0.60:
            box_color = (0, 255, 0)       # Green
            status_text = f"Attentive ({zone_attention*100:.0f}%)"
        elif zone_attention >= 0.45:
            box_color = (0, 215, 255)     # Yellow / Gold
            status_text = f"Neutral ({zone_attention*100:.0f}%)"
        else:
            box_color = (0, 0, 255)       # Red
            status_text = f"Looking Away ({zone_attention*100:.0f}%)"

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)

        # Draw Head Facing Vector Arrow
        if face_anchor_px is not None and v_facing is not None:
            ax, ay = int(face_anchor_px[0]), int(face_anchor_px[1])
            arrow_len = 45
            arrow_end = (int(ax + v_facing[0] * arrow_len), int(ay + v_facing[1] * arrow_len))
            cv2.arrowedLine(img, (ax, ay), arrow_end, (255, 255, 0), 2, tipLength=0.3)

        # Draw subtle target gaze guide line
        cv2.line(img, (cx_px, cy_px), (tx, ty), (128, 128, 128), 1, cv2.LINE_AA)

        # Badge pill above head
        badge_y = max(y1 - 10, 20)
        cv2.rectangle(img, (x1, badge_y - 16), (x1 + 140, badge_y + 4), (20, 20, 20), -1)
        cv2.putText(
            img,
            status_text,
            (x1 + 4, badge_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            box_color,
            1,
            cv2.LINE_AA,
        )

    # 4. Top HUD Banner
    hud_overlay = img.copy()
    cv2.rectangle(hud_overlay, (0, 0), (w, 42), (15, 15, 15), -1)
    img = cv2.addWeighted(hud_overlay, 0.85, img, 0.15, 0)

    mean_att = np.mean(student_attentions) if student_attentions else 0.5
    vfoa_focus = np.mean([a >= 0.55 for a in student_attentions]) if student_attentions else 0.0

    hud_text = (
        f"Video: {video_name} | Frame {frame_idx+1}/{total_frames} | "
        f"Students: {len(student_attentions)} | Zone Attention: {mean_att*100:.1f}% | Focus: {vfoa_focus*100:.0f}%"
    )
    cv2.putText(img, hud_text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def process_and_save_video(
    npz_path: Path,
    model: YOLO,
    device: str,
    output_dir: Path,
):
    """Generate visualized images and a composite contact sheet for a video."""
    video_name = npz_path.stem
    category = npz_path.parent.name
    split = npz_path.parent.parent.name

    with np.load(npz_path, allow_pickle=False) as data:
        frames_rgb = data["frames"]  # (8, 640, 640, 3)

    frames_bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames_rgb]
    results = model.predict(source=frames_bgr, conf=0.25, device=device, verbose=False)

    rendered_frames = []
    video_out_dir = output_dir / f"{split}_{category}_{video_name}"
    video_out_dir.mkdir(parents=True, exist_ok=True)

    for idx, (frame, res) in enumerate(zip(frames_rgb, results)):
        rendered = draw_interaction_frame(frame, res, idx, len(frames_rgb), f"{category.upper()}/{video_name}")
        rendered_frames.append(rendered)
        cv2.imwrite(str(video_out_dir / f"frame_{idx+1}.png"), rendered)

    # Create a 2x4 composite contact sheet
    top_row = np.hstack(rendered_frames[:4])
    bottom_row = np.hstack(rendered_frames[4:])
    montage = np.vstack([top_row, bottom_row])
    montage_resized = cv2.resize(montage, (1920, 960), interpolation=cv2.INTER_AREA)

    montage_path = output_dir / f"montage_{category}_{video_name}.png"
    cv2.imwrite(str(montage_path), montage_resized)
    print(f"✅ Saved 8-frame montage to: {montage_path}")
    return montage_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_path", type=Path, help="Path to specific .npz video file")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "audit_outputs" / "interaction_visualizations",
    )
    parser.add_argument("--model", default="yolov8n-pose.pt")
    parser.add_argument("--sample", action="store_true", help="Sample 1 High, 1 Mid, 1 Low video")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)

    if args.sample or not args.video_path:
        base_dir = PROJECT_ROOT / "preprocessed_data" / "yolov5_640x640" / "test"
        sample_videos = []
        for cat in ("high", "mid", "low"):
            cat_dir = base_dir / cat
            if cat_dir.is_dir():
                vids = sorted(cat_dir.glob("*.npz"))
                if vids:
                    sample_videos.append(vids[0])

        if not sample_videos:
            print("No test .npz files found under", base_dir)
            return

        print(f"Sampling {len(sample_videos)} test videos across High, Mid, and Low categories...")
        for vid in sample_videos:
            process_and_save_video(vid, model, device, args.output_dir)
    else:
        process_and_save_video(args.video_path.resolve(), model, device, args.output_dir)


if __name__ == "__main__":
    main()
