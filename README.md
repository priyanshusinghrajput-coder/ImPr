# DSLR - Dynamic Sign Language Recognition System

A two-stage system that recognizes continuous sign-language gestures from
video and speaks the resulting phrase aloud. Training happens on Google
Colab (GPU); real-time inference runs on a Raspberry Pi (CPU, camera, aux
audio output).

```
DSLR/
├── config.py                     # shared constants (BOTH stages import this)
├── requirements.txt
├── utils/
│   └── landmark_utils.py         # MediaPipe Holistic wrapper, shared by both stages
├── training/                     # ---- runs on Google Colab ----
│   ├── 01_collect_data.py        # (optional, run locally) webcam -> raw dataset
│   ├── 02_extract_landmarks.py   # raw dataset -> .npy landmark sequences
│   ├── 03_train_model.py         # .npy sequences -> trained LSTM (.keras)
│   └── 04_convert_tflite.py      # .keras -> .tflite (recommended for the Pi)
├── inference/                    # ---- runs on Raspberry Pi ----
│   ├── setup_pi.sh               # clone repo + install deps on a fresh Pi
│   ├── realtime_inference.py     # camera -> MediaPipe -> LSTM -> sentence -> TTS
│   ├── sentence_builder.py       # debouncing + word/phrase formation logic
│   └── tts_engine.py             # non-blocking pyttsx3 wrapper
├── data/
│   ├── raw/                      # captured photos/videos (git-ignored, large)
│   └── landmarks/                # extracted .npy sequences (git-ignored, large)
└── models/
    ├── dslr_lstm_model.keras     # trained model (commit this)
    ├── dslr_lstm_model.tflite    # optimized model for the Pi (commit this)
    └── label_map.json            # class index -> gesture name (commit this)
```

## Stage 1 - Training (Google Colab)

1. **Collect data.** Either run `training/01_collect_data.py` locally on a
   machine with a webcam (saves `data/raw/<action>/<sequence>/<frame>.jpg`),
   or supply your own `.mp4`/`.avi` clips per gesture. Zip `data/raw/` and
   upload it to Google Drive.

2. **Open a Colab notebook**, mount Drive, unzip your dataset into
   `data/raw/`, then install deps and run the pipeline:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   !git clone https://github.com/<you>/<your-dslr-repo>.git
   %cd your-dslr-repo
   !unzip -q "/content/drive/MyDrive/dslr_raw_dataset.zip" -d data/raw

   !pip install -q mediapipe opencv-python-headless tqdm

   !python training/02_extract_landmarks.py
   !python training/03_train_model.py
   !python training/04_convert_tflite.py
   ```

3. **Push the trained artifacts to GitHub** (or download them):
   `models/dslr_lstm_model.keras`, `models/dslr_lstm_model.tflite`,
   `models/label_map.json`.

   ```bash
   git add models/
   git commit -m "Add trained DSLR model"
   git push
   ```

## Stage 2 - Real-time inference (Raspberry Pi)

1. **Flash Raspberry Pi OS**, enable the camera (`sudo raspi-config` ->
   Interface Options -> Camera), and connect a speaker/headphones to the
   3.5mm aux jack (or configure USB/HDMI audio).

2. **Clone the repo and install dependencies** with the provided setup
   script, which installs `tflite-runtime` for a lightweight, fast model
   backend instead of full TensorFlow:

   ```bash
   chmod +x inference/setup_pi.sh
   ./inference/setup_pi.sh https://github.com/<you>/<your-dslr-repo>.git
   source ~/DSLR/venv/bin/activate
   ```

3. **Run inference:**

   ```bash
   python inference/realtime_inference.py --headless
   ```

   Drop `--headless` if a monitor is attached and you want a live landmark
   preview. Recognized gestures assemble into a sentence; signing the
   `"done"` control gesture speaks the finished phrase aloud through the
   aux output. Signing `"space"` or `"delete"` edits the in-progress
   sentence without speaking anything.

## Extending the gesture vocabulary

Add new words to `ACTIONS` in `config.py`, capture matching training data
for them, and re-run the Stage 1 pipeline. `config.py` is imported by every
script in both stages, so it's the only place you need to edit vocabulary
or sequence-length settings.

## Performance notes for the Pi

- `utils/landmark_utils.py` intentionally skips MediaPipe's 468-point face
  mesh (not read from `results.face_landmarks`) since it's the most
  expensive part of Holistic and isn't needed for hand/arm-driven signs.
- `realtime_inference.py` defaults to MediaPipe's lite model complexity
  (`--model-complexity 0`) for maximum frame rate on Pi-class CPUs.
- Prefer the `.tflite` model export (`training/04_convert_tflite.py`) over
  the full `.keras`/TensorFlow path on-device; the inference script picks
  it automatically when present.
- Camera capture is limited to 640x480 and a modest target FPS
  (`config.FRAMES_PER_SECOND`) to keep CPU headroom for MediaPipe + the
  LSTM.
