"""
config.py
=========
Single source of truth for constants shared between the Colab training
pipeline and the Raspberry Pi inference pipeline. Keeping this identical
on both sides is critical: if SEQUENCE_LENGTH or the keypoint layout
differs between training and inference, the model's input shape will not
match and predictions will silently be garbage.

Import this module from every other script instead of hard-coding values.
"""

import os

# --------------------------------------------------------------------------
# Gesture vocabulary
# --------------------------------------------------------------------------
# Each entry is a "sign" the model is trained to recognize. Extend this list
# to match your captured dataset. Order matters: the index of each action
# here becomes its integer class label during training.
ACTIONS = [
    "hello",
    "thanks",
    "please",
    "yes",
    "no",
    "help",
    "name",
    "sorry",
    "good",
    "bad",
    # Control tokens used by the sentence-formation layer at inference time.
    # Include these in your dataset as their own gesture classes if you want
    # the signer to be able to control sentence assembly (space/clear/end).
    "space",
    "delete",
    "done",
]

# --------------------------------------------------------------------------
# Sequence / windowing parameters
# --------------------------------------------------------------------------
SEQUENCE_LENGTH = 30      # number of frames that make up one gesture sample
FRAMES_PER_SECOND = 15    # target capture FPS on the Pi (keep modest for CPU headroom)

# --------------------------------------------------------------------------
# Landmark layout (MUST match utils/landmark_utils.extract_keypoints)
# --------------------------------------------------------------------------
# We deliberately EXCLUDE MediaPipe's 468-point face mesh: it is by far the
# most expensive part of Holistic to compute and adds little value for sign
# language, which is carried by hand shape, hand trajectory and pose/arms.
# Dropping it roughly triples inference FPS on a Raspberry Pi.
POSE_LANDMARKS = 33
HAND_LANDMARKS = 21

POSE_FEATURES = POSE_LANDMARKS * 4   # x, y, z, visibility
HAND_FEATURES = HAND_LANDMARKS * 3   # x, y, z (per hand)

NUM_FEATURES = POSE_FEATURES + HAND_FEATURES + HAND_FEATURES  # pose + left + right

# --------------------------------------------------------------------------
# Filesystem layout
# --------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")             # captured photos/videos
LANDMARK_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "landmarks")  # extracted .npy sequences

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, "dslr_lstm_model.keras")
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "dslr_lstm_model.tflite")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")
TRAINING_HISTORY_PLOT = os.path.join(MODEL_DIR, "training_history.png")

# --------------------------------------------------------------------------
# Inference thresholds
# --------------------------------------------------------------------------
PREDICTION_CONFIDENCE_THRESHOLD = 0.80   # ignore predictions below this softmax score
STABLE_FRAMES_REQUIRED = 8               # consecutive matching predictions before accepting a word
COOLDOWN_FRAMES_AFTER_ACCEPT = 15        # frames to ignore after a word is accepted (avoid repeats)
