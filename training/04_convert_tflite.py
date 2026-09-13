"""
training/04_convert_tflite.py
===============================
STAGE: Export optimization (run on Google Colab, right after training).

Converts the trained Keras model to TensorFlow Lite. This is the single
biggest lever for Raspberry Pi performance:
  - `tflite-runtime` is a small ARM-native wheel (~a few MB) versus full
    TensorFlow (several hundred MB) - much faster to load and lower memory.
  - Dynamic-range quantization shrinks the model further and speeds up
    matrix multiplications on the Pi's CPU, at a small, usually negligible,
    accuracy cost for a model this size.

Usage on Colab:
    !python training/04_convert_tflite.py
Commit the resulting models/dslr_lstm_model.tflite to GitHub alongside the
.keras file. The Pi inference script will prefer the .tflite file if
tflite-runtime is installed, and fall back to the full .keras model +
TensorFlow otherwise.
"""

import os
import sys

import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import KERAS_MODEL_PATH, TFLITE_MODEL_PATH  # noqa: E402


def convert():
    if not os.path.exists(KERAS_MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {KERAS_MODEL_PATH}. Run 03_train_model.py first."
        )

    model = tf.keras.models.load_model(KERAS_MODEL_PATH)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # dynamic-range quantization
    # LSTMs need this to convert reliably with TFLite's built-in ops:
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    tflite_model = converter.convert()

    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)

    orig_mb = os.path.getsize(KERAS_MODEL_PATH) / 1e6
    new_mb = os.path.getsize(TFLITE_MODEL_PATH) / 1e6
    print(f"Converted. {KERAS_MODEL_PATH} ({orig_mb:.2f} MB) -> {TFLITE_MODEL_PATH} ({new_mb:.2f} MB)")


if __name__ == "__main__":
    convert()
