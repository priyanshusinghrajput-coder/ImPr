"""
training/01_collect_data.py
============================
STAGE: Data Collection (run locally on a laptop/desktop with a webcam, OR
on the Raspberry Pi itself if that's your only camera). This is NOT a Colab
script (Colab has no access to your webcam) - it produces the raw dataset
that you then zip and upload to Google Drive / Colab for steps 02 and 03.

For each action in config.ACTIONS, this script records NO_SEQUENCES separate
gesture repetitions, each SEQUENCE_LENGTH frames long, and saves every frame
as a .jpg image under:

    data/raw/<action>/<sequence_index>/<frame_index>.jpg

That folder-of-frames layout is what training/02_extract_landmarks.py
expects. If you already have your own videos, skip this script and instead
adapt 02_extract_landmarks.py's loader to read video files directly (a
helper for that is included there).

Controls while running:
    's' - start recording the next sequence for the current action
    'q' - quit early
"""

import os
import time
import cv2

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ACTIONS, SEQUENCE_LENGTH, RAW_DATA_PATH  # noqa: E402

NO_SEQUENCES = 30  # how many repetitions to record per action - more = better model


def ensure_dirs():
    for action in ACTIONS:
        for seq in range(NO_SEQUENCES):
            os.makedirs(os.path.join(RAW_DATA_PATH, action, str(seq)), exist_ok=True)


def main():
    ensure_dirs()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera index / permissions.")

    print("=== DSLR Data Collection ===")
    print(f"Actions to record: {ACTIONS}")
    print(f"{NO_SEQUENCES} sequences x {SEQUENCE_LENGTH} frames each.\n")

    for action in ACTIONS:
        for sequence in range(NO_SEQUENCES):
            # Countdown / ready screen so the signer can position themselves
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                display = frame.copy()
                cv2.putText(display, f'Action: "{action}"  Sequence: {sequence}',
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display, "Press 's' to start, 'q' to quit",
                            (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("DSLR Data Collection", display)
                key = cv2.waitKey(10) & 0xFF
                if key == ord('s'):
                    break
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # Record SEQUENCE_LENGTH frames for this sequence
            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret:
                    continue

                display = frame.copy()
                cv2.putText(display, f'RECORDING "{action}" seq {sequence} frame {frame_num}',
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("DSLR Data Collection", display)

                out_path = os.path.join(RAW_DATA_PATH, action, str(sequence), f"{frame_num}.jpg")
                cv2.imwrite(out_path, frame)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            time.sleep(0.3)  # brief pause between sequences

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Raw dataset saved under: {RAW_DATA_PATH}")
    print("Zip this folder and upload it to Google Drive for the Colab training steps.")


if __name__ == "__main__":
    main()
