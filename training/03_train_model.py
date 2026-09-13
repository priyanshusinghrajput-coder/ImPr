"""
training/03_train_model.py
============================
STAGE: Model Training (run on Google Colab, ideally with a GPU runtime:
Runtime > Change runtime type > GPU).

Loads the .npy landmark sequences produced by 02_extract_landmarks.py,
builds an LSTM-based sequence classifier in Keras, trains it, and exports:

    models/dslr_lstm_model.keras   - full model (architecture + weights)
    models/label_map.json          - {class_index: action_name} for inference
    models/training_history.png    - accuracy/loss curves

--------------------------------------------------------------------------
Colab usage:
    !python training/03_train_model.py
Then commit the `models/` folder to your GitHub repo (or download it) so
the Raspberry Pi can pull it down in the inference stage.
--------------------------------------------------------------------------
"""

import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe for Colab/servers
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (ACTIONS, SEQUENCE_LENGTH, NUM_FEATURES, LANDMARK_DATA_PATH,  # noqa: E402
                     MODEL_DIR, KERAS_MODEL_PATH, LABEL_MAP_PATH, TRAINING_HISTORY_PLOT)


def load_dataset():
    """Loads every .npy sequence file into X (features) / y (integer labels)."""
    X, y = [], []
    label_map = {action: idx for idx, action in enumerate(ACTIONS)}

    for action in ACTIONS:
        action_dir = os.path.join(LANDMARK_DATA_PATH, action)
        if not os.path.isdir(action_dir):
            print(f"[warn] no landmark data for '{action}', skipping")
            continue
        for fname in sorted(os.listdir(action_dir)):
            if fname.endswith(".npy"):
                seq = np.load(os.path.join(action_dir, fname))
                if seq.shape == (SEQUENCE_LENGTH, NUM_FEATURES):
                    X.append(seq)
                    y.append(label_map[action])
                else:
                    print(f"[warn] shape mismatch in {fname}: {seq.shape}, skipping")

    if not X:
        raise RuntimeError(
            "No training data found. Run 02_extract_landmarks.py first and "
            f"confirm files exist under {LANDMARK_DATA_PATH}."
        )

    X = np.array(X, dtype=np.float32)
    y = to_categorical(np.array(y), num_classes=len(ACTIONS)).astype(np.float32)
    return X, y, label_map


def build_model(num_classes: int) -> Sequential:
    """
    Sequential LSTM classifier.

    - Masking: ignores zero-padded frames (from shorter sequences) so they
      don't skew the LSTM's hidden state.
    - Stacked LSTM layers capture short-term (hand shape changes) and
      longer-term (overall gesture trajectory) temporal patterns.
    - Dropout between layers combats overfitting, especially important with
      small custom sign-language datasets.
    """
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
        LSTM(64, return_sequences=True, activation="tanh"),
        Dropout(0.3),
        LSTM(128, return_sequences=True, activation="tanh"),
        Dropout(0.3),
        LSTM(64, return_sequences=False, activation="tanh"),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    return model


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="train_loss")
    axes[0].plot(history.history["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history.history["categorical_accuracy"], label="train_acc")
    axes[1].plot(history.history["val_categorical_accuracy"], label="val_acc")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(TRAINING_HISTORY_PLOT)
    print(f"Saved training curves to {TRAINING_HISTORY_PLOT}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading dataset...")
    X, y, label_map = load_dataset()
    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y.argmax(axis=1)
    )

    model = build_model(num_classes=len(ACTIONS))
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        ModelCheckpoint(KERAS_MODEL_PATH, monitor="val_categorical_accuracy",
                         save_best_only=True, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=200,
        batch_size=16,
        callbacks=callbacks,
        verbose=1,
    )

    plot_history(history)

    # --- Evaluation ---
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=ACTIONS, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    # --- Export artifacts for the Raspberry Pi ---
    model.save(KERAS_MODEL_PATH)  # ensures final (not just best-checkpoint) state is saved too
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump({str(idx): action for action, idx in label_map.items()}, f, indent=2)

    print(f"\nModel saved to:      {KERAS_MODEL_PATH}")
    print(f"Label map saved to:  {LABEL_MAP_PATH}")
    print("\nNext: commit the models/ folder to your GitHub repo, then run")
    print("training/04_convert_tflite.py (optional but recommended for Raspberry Pi speed).")


if __name__ == "__main__":
    main()
