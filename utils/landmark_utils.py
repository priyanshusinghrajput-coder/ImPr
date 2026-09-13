"""
utils/landmark_utils.py
========================
Shared MediaPipe Holistic helpers. This module is imported by BOTH:
  - training/02_extract_landmarks.py   (running on Colab, over saved video/images)
  - inference/realtime_inference.py    (running on the Raspberry Pi, over live frames)

Keeping extraction logic in one place guarantees training and inference see
identical feature vectors.

Design choice for Pi performance: we use MediaPipe Holistic but explicitly
skip the face mesh (refine_face_landmarks is never enabled and we do not
read results.face_landmarks). Face landmarks are the most expensive part of
Holistic and are not required for hand/arm-driven sign language, so this
keeps CPU load low enough for real-time use on a Raspberry Pi.
"""

import numpy as np
import cv2
import mediapipe as mp

from config import POSE_LANDMARKS, HAND_LANDMARKS  # noqa: F401 (kept for external reference)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def make_holistic_model(static_image_mode: bool = False,
                         model_complexity: int = 1,
                         min_detection_confidence: float = 0.5,
                         min_tracking_confidence: float = 0.5):
    """
    Factory for a MediaPipe Holistic instance.

    model_complexity: 0 (lite), 1 (full), 2 (heavy). On Raspberry Pi, use 0
    for maximum speed if accuracy allows; use 1 on Colab / desktop for
    higher-quality landmarks during dataset creation.
    """
    return mp_holistic.Holistic(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def mediapipe_detection(frame_bgr, holistic_model):
    """
    Runs one frame through MediaPipe Holistic.

    Args:
        frame_bgr: a single OpenCV BGR frame.
        holistic_model: an active mp_holistic.Holistic() context.

    Returns:
        (processed_frame_bgr, results) - frame is returned unmodified (BGR)
        for display purposes; results is the raw MediaPipe output object.
    """
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False          # perf: avoid an internal copy
    results = holistic_model.process(image_rgb)
    image_rgb.flags.writeable = True
    return frame_bgr, results


def draw_landmarks(frame_bgr, results):
    """Overlay pose + hand landmarks on a frame. Useful for debugging/demo,
    not used in the headless production inference loop (drawing costs CPU)."""
    mp_drawing.draw_landmarks(frame_bgr, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(frame_bgr, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame_bgr, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    return frame_bgr


def extract_keypoints(results) -> np.ndarray:
    """
    Flattens a single frame's MediaPipe results into one feature vector.

    Layout (must match config.NUM_FEATURES):
        pose        : 33 landmarks * (x, y, z, visibility) = 132
        left hand   : 21 landmarks * (x, y, z)             = 63
        right hand  : 21 landmarks * (x, y, z)             = 63
        TOTAL                                              = 258

    Missing landmarks (e.g. a hand out of frame) are zero-filled so the
    output shape is always constant, which is required for LSTM batching.
    """
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                          for lm in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(POSE_LANDMARKS * 4)

    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z]
                        for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(HAND_LANDMARKS * 3)

    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z]
                        for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(HAND_LANDMARKS * 3)

    return np.concatenate([pose, lh, rh]).astype(np.float32)
