"""
training/02_extract_landmarks.py
==================================
STAGE: Data Preprocessing (run on Google Colab).

Reads the raw dataset (folders of per-frame .jpg images, OR .mp4/.avi video
files - both supported below) and runs every frame through MediaPipe
Holistic to extract landmark keypoints. Each gesture sample (one full
sequence) becomes a single .npy file of shape (SEQUENCE_LENGTH, NUM_FEATURES).

Expected raw layout (from 01_collect_data.py):
    data/raw/<action>/<sequence_index>/<frame_index>.jpg

Output layout:
    data/landmarks/<action>/<sequence_index>.npy   # shape (SEQUENCE_LENGTH, NUM_FEATURES)

--------------------------------------------------------------------------
Colab usage:
    1. Mount Google Drive and unzip your uploaded raw dataset into
       /content/DSLR/data/raw (matching RAW_DATA_PATH in config.py), e.g.:

        from google.colab import drive
        drive.mount('/content/drive')
        !unzip -q "/content/drive/MyDrive/dslr_raw_dataset.zip" -d /content/DSLR/data/raw

    2. !pip install mediapipe opencv-python-headless
    3. !python training/02_extract_landmarks.py
--------------------------------------------------------------------------
"""

import os
import sys
import glob

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ACTIONS, SEQUENCE_LENGTH, RAW_DATA_PATH, LANDMARK_DATA_PATH  # noqa: E402
from utils.landmark_utils import make_holistic_model, mediapipe_detection, extract_keypoints  # noqa: E402


def frames_from_image_folder(sequence_dir: str):
    """Yields BGR frames in order for a sequence stored as N .jpg files."""
    frame_files = sorted(
        glob.glob(os.path.join(sequence_dir, "*.jpg")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
    )
    for path in frame_files:
        frame = cv2.imread(path)
        if frame is not None:
            yield frame


def frames_from_video(video_path: str):
    """Yields BGR frames from a single video file (.mp4/.avi/.mov)."""
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()


def sample_or_pad_sequence(keypoint_list, target_len: int, num_features: int):
    """
    Normalizes a variable-length list of per-frame keypoint vectors to exactly
    `target_len` frames:
      - if longer: uniformly sample `target_len` frames across the sequence
      - if shorter: pad with zero-vectors at the end
    This keeps every training sample the same shape, which LSTMs need for
    efficient batching (we still use a Masking layer downstream for safety).
    """
    n = len(keypoint_list)
    if n == 0:
        return np.zeros((target_len, num_features), dtype=np.float32)

    if n >= target_len:
        idx = np.linspace(0, n - 1, target_len).astype(int)
        return np.array([keypoint_list[i] for i in idx], dtype=np.float32)

    padded = list(keypoint_list) + [np.zeros(num_features, dtype=np.float32)] * (target_len - n)
    return np.array(padded, dtype=np.float32)


def process_dataset():
    holistic = make_holistic_model(static_image_mode=False, model_complexity=1)

    os.makedirs(LANDMARK_DATA_PATH, exist_ok=True)

    for action in ACTIONS:
        action_raw_dir = os.path.join(RAW_DATA_PATH, action)
        action_out_dir = os.path.join(LANDMARK_DATA_PATH, action)
        os.makedirs(action_out_dir, exist_ok=True)

        if not os.path.isdir(action_raw_dir):
            print(f"[skip] No raw data found for action '{action}' at {action_raw_dir}")
            continue

        # Each "sample" is either a subfolder of frames or a single video file.
        sample_entries = sorted(os.listdir(action_raw_dir))

        for entry in tqdm(sample_entries, desc=f"Processing '{action}'"):
            entry_path = os.path.join(action_raw_dir, entry)
            sample_id = os.path.splitext(entry)[0]
            out_path = os.path.join(action_out_dir, f"{sample_id}.npy")

            if os.path.isdir(entry_path):
                frame_iter = frames_from_image_folder(entry_path)
            elif entry.lower().endswith((".mp4", ".avi", ".mov")):
                frame_iter = frames_from_video(entry_path)
            else:
                continue  # not a recognized sample format

            keypoints_per_frame = []
            for frame in frame_iter:
                _, results = mediapipe_detection(frame, holistic)
                keypoints_per_frame.append(extract_keypoints(results))

            from config import NUM_FEATURES
            sequence_array = sample_or_pad_sequence(keypoints_per_frame, SEQUENCE_LENGTH, NUM_FEATURES)
            np.save(out_path, sequence_array)

    holistic.close()
    print(f"\nLandmark extraction complete. Saved under: {LANDMARK_DATA_PATH}")


if __name__ == "__main__":
    process_dataset()
