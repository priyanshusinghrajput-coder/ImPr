"""
inference/realtime_inference.py
==================================
STAGE: Real-Time Edge Inference (run on the Raspberry Pi).

Pipeline per frame:
    camera frame --(OpenCV)--> MediaPipe Holistic --> keypoint vector
        --> rolling window of SEQUENCE_LENGTH frames
        --> LSTM model --> class probabilities
        --> SentenceBuilder (debounce + word formation)
        --> pyttsx3 (spoken output, non-blocking)

Model backend:
    Prefers TensorFlow Lite (models/dslr_lstm_model.tflite) via
    `tflite-runtime` for speed and low memory use on the Pi. Falls back to
    the full Keras model (models/dslr_lstm_model.keras) + TensorFlow if the
    .tflite file or tflite-runtime isn't available - useful during
    development/testing on a laptop before deploying to the Pi.

Run:
    source venv/bin/activate
    python inference/realtime_inference.py [--camera 0] [--headless]
"""

import os
import sys
import json
import time
import argparse
from collections import deque

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (SEQUENCE_LENGTH, NUM_FEATURES, KERAS_MODEL_PATH, TFLITE_MODEL_PATH,  # noqa: E402
                     LABEL_MAP_PATH, PREDICTION_CONFIDENCE_THRESHOLD, FRAMES_PER_SECOND)
from utils.landmark_utils import make_holistic_model, mediapipe_detection, extract_keypoints, draw_landmarks  # noqa: E402
from inference.sentence_builder import SentenceBuilder  # noqa: E402
from inference.tts_engine import TTSEngine  # noqa: E402


# ============================================================================
# Model wrapper - abstracts over TFLite vs full Keras so the main loop below
# doesn't need to care which backend is active.
# ============================================================================
class GestureModel:
    def __init__(self):
        self.backend = None
        self._load()

    def _load(self):
        if os.path.exists(TFLITE_MODEL_PATH):
            try:
                try:
                    import tflite_runtime.interpreter as tflite
                except ImportError:
                    from tensorflow.lite.python import interpreter as tflite  # fallback if only full TF is installed
                self.interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.backend = "tflite"
                print("[model] Loaded TFLite model (fast path).")
                return
            except Exception as e:
                print(f"[model] TFLite load failed ({e}); falling back to Keras model.")

        if os.path.exists(KERAS_MODEL_PATH):
            import tensorflow as tf
            self.keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH)
            self.backend = "keras"
            print("[model] Loaded full Keras model.")
            return

        raise FileNotFoundError(
            "No model found. Expected a .tflite or .keras file under models/. "
            "Train and export a model first (see training/ scripts)."
        )

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        """sequence: shape (SEQUENCE_LENGTH, NUM_FEATURES). Returns class probabilities."""
        batch = np.expand_dims(sequence, axis=0).astype(np.float32)

        if self.backend == "tflite":
            self.interpreter.set_tensor(self.input_details[0]["index"], batch)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]["index"])
            return output[0]

        # keras backend
        output = self.keras_model.predict(batch, verbose=0)
        return output[0]


def load_label_map():
    with open(LABEL_MAP_PATH, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def open_camera(camera_index: int):
    """
    Opens the camera via OpenCV. On a Raspberry Pi with the official camera
    module and libcamera, index 0 typically works through the V4L2 backend
    once `sudo raspi-config` -> camera is enabled and
    `sudo apt-get install v4l-utils` / libcamera stack is set up. For USB
    webcams, index 0/1 selects the device as usual.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. On the Pi, verify the camera "
            "is enabled (raspi-config) and check `ls /dev/video*`."
        )
    # Keep resolution modest for CPU headroom on the Pi.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, FRAMES_PER_SECOND)
    return cap


def main():
    parser = argparse.ArgumentParser(description="DSLR real-time inference")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--headless", action="store_true",
                         help="Run without opening a preview window (recommended for a Pi with no monitor)")
    parser.add_argument("--model-complexity", type=int, default=0, choices=[0, 1, 2],
                         help="MediaPipe Holistic model complexity; 0=lite is fastest, best default for Pi")
    args = parser.parse_args()

    label_map = load_label_map()
    model = GestureModel()
    tts = TTSEngine()
    sentence = SentenceBuilder()

    holistic = make_holistic_model(static_image_mode=False, model_complexity=args.model_complexity)
    cap = open_camera(args.camera)

    keypoint_window = deque(maxlen=SEQUENCE_LENGTH)
    frame_interval = 1.0 / FRAMES_PER_SECOND

    print("=== DSLR real-time inference running. Press Ctrl+C (or 'q' if not headless) to stop. ===")
    tts.speak("Sign language recognition ready.")

    try:
        last_tick = time.time()
        while True:
            # Simple frame-rate limiter so we don't spend more CPU than needed
            # capturing frames faster than the model realistically needs.
            elapsed = time.time() - last_tick
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            last_tick = time.time()

            ret, frame = cap.read()
            if not ret:
                print("[warn] Camera frame grab failed, retrying...")
                continue

            frame, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            keypoint_window.append(keypoints)

            predicted_label, confidence = None, 0.0

            if len(keypoint_window) == SEQUENCE_LENGTH:
                sequence = np.array(keypoint_window, dtype=np.float32)
                probabilities = model.predict(sequence)
                class_idx = int(np.argmax(probabilities))
                confidence = float(probabilities[class_idx])
                predicted_label = label_map.get(class_idx, "unknown")

                accepted_word = sentence.update(predicted_label, confidence,
                                                 PREDICTION_CONFIDENCE_THRESHOLD)
                if accepted_word:
                    print(f"[accepted] '{accepted_word}'  (sentence so far: '{sentence.get_sentence()}')")

                if sentence.finalized:
                    phrase = sentence.get_sentence().replace(" done", "").strip()
                    if phrase:
                        print(f"[speaking] '{phrase}'")
                        tts.speak(phrase)
                    sentence.reset()
                    keypoint_window.clear()

            if not args.headless:
                display = draw_landmarks(frame.copy(), results)
                status = f"{predicted_label or '...'} ({confidence:.2f})"
                cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display, f"Sentence: {sentence.get_sentence()}", (10, 460),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.imshow("DSLR Inference", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C received)...")
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        holistic.close()
        tts.flush()
        tts.shutdown()


if __name__ == "__main__":
    main()
