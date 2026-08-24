import os
import cv2
import numpy as np
from tqdm import tqdm
from emotiefflib.facial_analysis import EmotiEffLibRecognizer

# Neutral baseline emotion probability vector (8 classes)
# 0: Anger, 1: Contempt, 2: Disgust, 3: Fear, 4: Happiness, 5: Neutral, 6: Sadness, 7: Surprise
NEUTRAL_BASELINE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)

def main():
    print("=" * 60)
    print("Extracting Affect Features using YuNet + HSEmotion (ONNX)...")
    print("=" * 60)

    # Output directory
    output_base = os.path.join("preprocessed_features", "affect_features")
    os.makedirs(output_base, exist_ok=True)

    # Path to YuNet model (downloaded in previous test)
    yunet_model_path = "face_detection_yunet_2023mar.onnx"
    if not os.path.exists(yunet_model_path):
        # Fallback download if missing
        import urllib.request
        model_url = 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
        print("YuNet model missing. Downloading...")
        urllib.request.urlretrieve(model_url, yunet_model_path)

    # Initialize Face Detector (640x640 input shape)
    detector = cv2.FaceDetectorYN.create(yunet_model_path, '', (640, 640), score_threshold=0.5)

    # Initialize HSEmotion Recognizer with ONNX engine (fast, doesn't depend on torch GPU)
    recognizer = EmotiEffLibRecognizer(model_name='enet_b0_8_best_vgaf', engine='onnx')

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
    for split, cat, fname in tqdm(tasks, desc="Extracting Affect Features"):
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

        # Load the preprocessed uint8 RGB frames: (8, 640, 640, 3)
        data = np.load(src_path)
        frames = data['frames']
        
        video_probs = []
        
        for frame_rgb in frames:
            # Convert to BGR for OpenCV YuNet
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Detect faces
            _, faces = detector.detect(frame_bgr)
            
            if faces is not None and len(faces) > 0:
                face_crops = []
                for face in faces:
                    x, y, w, h = face[:4].astype(int)
                    
                    # Pad slightly for emotion recognition alignment
                    pad = int(max(w, h) * 0.1)
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(640, x + w + pad)
                    y2 = min(640, y + h + pad)
                    
                    crop = frame_rgb[y1:y2, x1:x2]
                    # HSEmotion expects non-empty face crop
                    if crop.size > 0:
                        face_crops.append(crop)
                
                if face_crops:
                    # Run batch inference on all cropped faces in the frame
                    try:
                        _, scores = recognizer.predict_emotions(face_crops)
                        
                        # Apply softmax to logits to get probabilities
                        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
                        
                        # Average probabilities across all faces in the frame
                        avg_frame_probs = probs.mean(axis=0)
                        video_probs.append(avg_frame_probs)
                    except Exception:
                        video_probs.append(NEUTRAL_BASELINE)
                else:
                    video_probs.append(NEUTRAL_BASELINE)
            else:
                video_probs.append(NEUTRAL_BASELINE)
                
        # Stack to (8, 8) shape
        video_probs_arr = np.array(video_probs, dtype=np.float32).reshape(8, 8)
        np.save(dest_path, video_probs_arr)

    print("Affect feature extraction complete.")
    print("=" * 60)

if __name__ == "__main__":
    main()
